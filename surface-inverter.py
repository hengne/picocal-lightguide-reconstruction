"""
Inverse Surface Deformation Solver for Conical Light Guides

Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>

Implements a three-stage low-memory (< 500 MB) inverse optical reconstruction architecture:
  Stage 1: Axisymmetric generatrix profile calibration deltaR_axi(z)
  Stage 2: SO(2) rotational-equivariant 2D Jacobian assembly (576 bicubic B-spline DOFs)
  Stage 3: Nonlinear exact specular ray-tracing Gauss-Newton defect correction with backtracking line search
  Stage 4: Full-resolution 100x100 high-statistics independent Sobol validation
"""

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.optimize import least_squares
from scipy.special import erf

__author__ = "Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>"


@dataclass
class ScanCalibrationData:
    """
    Calibrated 2D XY diffuse laser scan grid and aperture parameters.
    """

    rawDf: pd.DataFrame
    xCoordsMm: np.ndarray
    yCoordsMm: np.ndarray
    rawVoltGrid: np.ndarray
    netMvGrid: np.ndarray
    centerXMm: float
    centerYMm: float
    effectiveRadiusMm: float
    beamSigmaMm: float
    backgroundVolt: float
    noiseStdMv: float
    noiseFloorPct: float


@dataclass
class InversionResult:
    """
    Complete output container from the multi-stage inverse surface reconstruction.
    """

    axiProfileMm: np.ndarray
    nonAxiGridMm: np.ndarray
    reflectivity: float
    cosineExponent: float
    gainScaleMv: float
    simFullNetMv: np.ndarray
    idealFullNetMv: np.ndarray
    iterHistoryRmsePct: list[float]
    metrics: dict[str, float]


def loadAndCalibrateScanCsv(csvPath: Path) -> ScanCalibrationData:
    """
    Load `scan_summary.csv` and perform electronic baseline subtraction, sub-pixel circle-center
    fitting, laser spot Gaussian sigma calibration, and discrete Laplacian Poisson noise estimation.
    """
    logger = logging.getLogger(__name__)
    df = pd.read_csv(csvPath)
    nx = int(df["idx_x"].max()) + 1
    ny = int(df["idx_y"].max()) + 1

    voltGrid = np.zeros((ny, nx), dtype=np.float32)
    for row in df.itertuples(index=False):
        voltGrid[int(row.idx_y), int(row.idx_x)] = float(row.probe_V)

    xCoords = np.sort(df["pos_x_mm"].unique()).astype(np.float32)
    yCoords = np.sort(df["pos_y_mm"].unique()).astype(np.float32)
    xx, yy = np.meshgrid(xCoords, yCoords)

    # NOTE: Use the median of low-voltage exterior pixels as a robust initial electronic baseline.
    bgQuantile = float(np.percentile(voltGrid, 18.0))
    bgMask = voltGrid <= bgQuantile + 0.0012
    bgInit = float(np.median(voltGrid[bgMask]))

    halfVal = bgInit + 0.5 * (float(np.max(voltGrid)) - bgInit)
    roughIn = voltGrid > halfVal
    cxInit = float(np.mean(xx[roughIn]))
    cyInit = float(np.mean(yy[roughIn]))

    dxStep = abs(float(xCoords[1] - xCoords[0])) if len(xCoords) > 1 else 0.2
    dyStep = abs(float(yCoords[1] - yCoords[0])) if len(yCoords) > 1 else 0.2
    rAreaInit = math.sqrt(max(1.0, float(np.sum(roughIn)) * dxStep * dyStep / math.pi))

    rRough = np.sqrt((xx - cxInit) ** 2 + (yy - cyInit) ** 2)
    edgeRing = (rRough > rAreaInit - 0.85) & (rRough < rAreaInit + 1.25)

    def edgeResidual(params: np.ndarray) -> np.ndarray:
        cx, cy, rEff, sigmaB, bgV, edgeAmp = params
        rad = np.sqrt((xx[edgeRing] - cx) ** 2 + (yy[edgeRing] - cy) ** 2)
        pred = bgV + edgeAmp * 0.5 * (1.0 - erf((rad - rEff) / (math.sqrt(2.0) * max(0.05, sigmaB))))
        return (pred - voltGrid[edgeRing]).astype(np.float64)

    ampInit = max(0.005, float(np.percentile(voltGrid[roughIn], 75.0)) - bgInit)
    fitRes = least_squares(
        edgeResidual,
        x0=np.array([cxInit, cyInit, rAreaInit, 0.22, bgInit, ampInit], dtype=np.float64),
        bounds=(
            [cxInit - 1.2, cyInit - 1.2, rAreaInit - 0.8, 0.08, bgInit - 0.003, ampInit * 0.4],
            [cxInit + 1.2, cyInit + 1.2, rAreaInit + 0.8, 0.70, bgInit + 0.003, ampInit * 1.8],
        ),
    )
    cxFit, cyFit, rEffFit, sigmaFit, bgFit, _ = [float(v) for v in fitRes.x]

    # NOTE: Convert net optical signal to millivolts (mV) for numerical conditioning and reporting.
    netMv = (voltGrid - bgFit) * 1000.0
    rExact = np.sqrt((xx - cxFit) ** 2 + (yy - cyFit) ** 2)

    # NOTE: Estimate single-pixel statistical Poisson noise via normalized 2D discrete Laplacian.
    laplacian = (
        netMv[1:-1, 2:] + netMv[1:-1, :-2] + netMv[2:, 1:-1] + netMv[:-2, 1:-1] - 4.0 * netMv[1:-1, 1:-1]
    ) / math.sqrt(20.0)
    innerCore = rExact[1:-1, 1:-1] < (rEffFit * 0.88)
    noiseStdMv = float(np.std(laplacian[innerCore]))
    innerMeanMv = float(np.mean(netMv[rExact < (rEffFit * 0.88)]))
    noiseFloorPct = (noiseStdMv / max(1e-6, innerMeanMv)) * 100.0

    logger.info(
        "Calibrated scan CSV: center=(%.4f, %.4f) mm, R_eff=%.3f mm, sigma_beam=%.3f mm, baseline=%.3f mV, noise_floor=%.2f%%",
        cxFit,
        cyFit,
        rEffFit,
        sigmaFit,
        bgFit * 1000.0,
        noiseFloorPct,
    )

    return ScanCalibrationData(
        rawDf=df,
        xCoordsMm=xCoords,
        yCoordsMm=yCoords,
        rawVoltGrid=voltGrid,
        netMvGrid=netMv,
        centerXMm=cxFit,
        centerYMm=cyFit,
        effectiveRadiusMm=rEffFit,
        beamSigmaMm=sigmaFit,
        backgroundVolt=bgFit,
        noiseStdMv=noiseStdMv,
        noiseFloorPct=noiseFloorPct,
    )


def computeAzimuthalResidualPct(
    netMv2d: np.ndarray,
    radius2d: np.ndarray,
    normMeanMv: float,
) -> np.ndarray:
    """
    Subtract concentric ring azimuthal averages to isolate pure non-axisymmetric bump variations (%).
    """
    residual = np.zeros_like(netMv2d, dtype=np.float32)
    rBins = np.linspace(0.0, 8.5, 44)
    for k in range(len(rBins) - 1):
        ringMask = (radius2d >= rBins[k]) & (radius2d < rBins[k + 1])
        if np.any(ringMask):
            residual[ringMask] = ((netMv2d[ringMask] - float(np.mean(netMv2d[ringMask]))) / normMeanMv) * 100.0
    return residual


class LightGuideSurfaceInverter:
    """
    Multi-stage nonlinear inverse solver reconstructing 3D inner surface deformations from 2D PMT scans.
    """

    def __init__(
        self,
        geometry: Any,
        surfaceClass: Any,
        tracerClass: Any,
        rayFactory: Any,
        memoryGuard: Any,
        device: Any,
        numThetaCtrl: int = 36,
        numZCtrl: int = 16,
        chunkPoints: int = 768,
    ) -> None:
        """
        Configure control-grid resolution and hardware batching parameters.
        """
        self.geometry: Any = geometry
        self.surfaceClass: Any = surfaceClass
        self.tracerClass: Any = tracerClass
        self.rayFactory: Any = rayFactory
        self.memoryGuard: Any = memoryGuard
        self.device: Any = device
        self.numThetaCtrl: int = numThetaCtrl
        self.numZCtrl: int = numZCtrl
        self.chunkPoints: int = chunkPoints
        self.logger: logging.Logger = logging.getLogger(__name__)

    def reconstructSurface(self, scanData: ScanCalibrationData) -> InversionResult:
        """
        Execute the full 4-stage surface reconstruction and 100x100 validation pipeline.
        """
        tStart = time.time()
        nTh = self.numThetaCtrl
        nZ = self.numZCtrl

        # Initial uniform inward offset from reflective mirror film thickness (~ -0.31 mm)
        filmOffsetInit = float(scanData.effectiveRadiusMm - self.geometry.inputRadiusMm)
        axiProfile = np.full(nZ, filmOffsetInit, dtype=np.float32)
        nonAxiGrid = np.zeros((nTh, nZ), dtype=np.float32)

        reflectivity = 0.955
        cosineExponent = 1.0

        surface = self.surfaceClass(self.geometry, axiProfile, nonAxiGrid, self.device)
        tracer = self.tracerClass(surface, self.memoryGuard, maxBounces=20, chunkPoints=self.chunkPoints)

        xxFull, yyFull = np.meshgrid(
            scanData.xCoordsMm - scanData.centerXMm,
            scanData.yCoordsMm - scanData.centerYMm,
        )
        rrFull = np.sqrt(xxFull * xxFull + yyFull * yyFull)

        # ==================== Stage 1: Axisymmetric Generatrix Profile Calibration ====================
        rMaxScan = round(scanData.effectiveRadiusMm + 0.66, 2)
        r1dDense = np.linspace(0.0, rMaxScan, 183, dtype=np.float32)
        r1d = np.linspace(0.0, rMaxScan, 46, dtype=np.float32)
        measRadial1d = np.zeros_like(r1d)
        for idx, rv in enumerate(r1d):
            ring = (rrFull >= max(0.0, float(rv) - 0.16)) & (rrFull < float(rv) + 0.16)
            measRadial1d[idx] = float(np.mean(scanData.netMvGrid[ring])) if np.any(ring) else 0.0

        bundleTrain = self.rayFactory(16384, self.device, cosineExponent=cosineExponent, seed=42)

        def evalRadial1d(candAxi: np.ndarray) -> np.ndarray:
            surface.updateControlGrid(candAxi, np.zeros((nTh, nZ), dtype=np.float32))
            effDense = tracer.tracePointCloud(r1dDense, np.zeros_like(r1dDense), bundleTrain, reflectivity=reflectivity)
            stepDense = float(r1dDense[1] - r1dDense[0])
            blurredDense = gaussian_filter(effDense, sigma=scanData.beamSigmaMm / stepDense)
            return np.interp(r1d, r1dDense, blurredDense).astype(np.float32)

        gainScaleMv = 160.0
        coreLimit = scanData.effectiveRadiusMm - 0.54
        for axiIter in range(4):
            base1d = evalRadial1d(axiProfile)
            core1d = r1d < coreLimit
            gainScaleMv = float(
                np.dot(base1d[core1d], measRadial1d[core1d]) / (np.dot(base1d[core1d], base1d[core1d]) + 1e-12)
            )
            jacAxi = np.zeros((len(r1d), nZ), dtype=np.float32)
            epsAxi = 0.04
            for j in range(nZ):
                pertP = axiProfile.copy()
                pertP[j] += epsAxi
                pertM = axiProfile.copy()
                pertM[j] -= epsAxi
                jacAxi[:, j] = gainScaleMv * (evalRadial1d(pertP) - evalRadial1d(pertM)) / (2.0 * epsAxi)

            d2Mat = (np.eye(nZ, k=1) - 2.0 * np.eye(nZ) + np.eye(nZ, k=-1))[1:-1]
            regAxi = 25.0 * (d2Mat.T @ d2Mat) + 3.0 * np.eye(nZ)
            regAxi[0, 0] += 40.0  # Anchor entrance rim z=0 to the calibrated clear aperture radius R_eff
            rhsAxi = jacAxi.T @ (measRadial1d - gainScaleMv * base1d) - regAxi @ (axiProfile - filmOffsetInit)
            deltaAxi = np.linalg.solve(jacAxi.T @ jacAxi + regAxi, rhsAxi).astype(np.float32)
            axiProfile = np.clip(axiProfile + 0.7 * deltaAxi, filmOffsetInit - 0.45, filmOffsetInit + 0.35)
            self.logger.debug("Axisymmetric calibration step %d completed (gain=%.2f mV)", axiIter + 1, gainScaleMv)

        snap1 = self.memoryGuard.inspectMemory("stage1_axisymmetric")
        self.logger.info(
            "[%.1fs] Stage 1 axisymmetric profile calibrated: deltaR_axi in [%.1f, %.1f] um | RSS=%.0f MB",
            time.time() - tStart,
            float(axiProfile.min() * 1000.0),
            float(axiProfile.max() * 1000.0),
            snap1.rssMb,
        )

        # ==================== Stage 2: SO(2) Equivariant 2D Jacobian Construction ====================
        workN = 42
        workHalfSpan = scanData.effectiveRadiusMm + 0.38
        workAxis = np.linspace(-workHalfSpan, workHalfSpan, workN, dtype=np.float32)
        workStep = float(workAxis[1] - workAxis[0])
        xxWork, yyWork = np.meshgrid(workAxis, workAxis)
        rrWork = np.sqrt(xxWork * xxWork + yyWork * yyWork)

        measInterp = RegularGridInterpolator(
            (scanData.yCoordsMm - scanData.centerYMm, scanData.xCoordsMm - scanData.centerXMm),
            scanData.netMvGrid,
            bounds_error=False,
            fill_value=0.0,
        )
        measWork = measInterp(np.stack([yyWork, xxWork], axis=-1)).astype(np.float32)

        def simulateWorkGrid(candNonAxi: np.ndarray, rayBdl: Any = bundleTrain) -> np.ndarray:
            surface.updateControlGrid(axiProfile, candNonAxi)
            rawEff = tracer.tracePointCloud(
                xxWork.reshape(-1), yyWork.reshape(-1), rayBdl, reflectivity=reflectivity
            ).reshape(workN, workN)
            blurred = gaussian_filter(rawEff, sigma=scanData.beamSigmaMm / workStep)
            return (gainScaleMv * blurred).astype(np.float32)

        epsBasis = 0.065
        basisResponses = np.zeros((nZ, workN, workN), dtype=np.float32)
        for j in range(nZ):
            gridPlus = np.zeros((nTh, nZ), dtype=np.float32)
            gridPlus[0, j] = +epsBasis
            gridMinus = np.zeros((nTh, nZ), dtype=np.float32)
            gridMinus[0, j] = -epsBasis
            respPlus = simulateWorkGrid(gridPlus)
            respMinus = simulateWorkGrid(gridMinus)
            basisResponses[j] = (respPlus - respMinus) / (2.0 * epsBasis)

        fitMaskWork = rrWork < (scanData.effectiveRadiusMm - 0.16)
        numObs = int(np.sum(fitMaskWork))
        numParams = nTh * nZ
        jacFull = np.zeros((numObs, numParams), dtype=np.float32)

        centerPx = 0.5 * (workN - 1.0)
        centerVec = np.array([centerPx, centerPx], dtype=np.float32)

        for i in range(nTh):
            angleRad = (2.0 * math.pi * i) / nTh
            cosA = math.cos(angleRad)
            sinA = math.sin(angleRad)
            rotMat = np.array([[cosA, -sinA], [sinA, cosA]], dtype=np.float32)
            offsetVec = centerVec - rotMat @ centerVec
            for j in range(nZ):
                rotatedResp = affine_transform(
                    basisResponses[j], rotMat, offset=offsetVec, order=3, mode="nearest"
                )
                jacFull[:, i * nZ + j] = rotatedResp[fitMaskWork]

        snap2 = self.memoryGuard.inspectMemory("stage2_jacobian")
        self.logger.info(
            "[%.1fs] Stage 2 SO(2) sensitivity matrix built (%dx%d=%d DOFs) | RSS=%.0f MB",
            time.time() - tStart,
            nTh,
            nZ,
            numParams,
            snap2.rssMb,
        )

        # ==================== Stage 3: Nonlinear Defect-Correction Gauss-Newton ====================
        lapTheta = np.zeros((numParams, numParams), dtype=np.float32)
        lapZ = np.zeros((numParams, numParams), dtype=np.float32)
        for i in range(nTh):
            iPrev = (i - 1) % nTh
            iNext = (i + 1) % nTh
            for j in range(nZ):
                pIdx = i * nZ + j
                lapTheta[pIdx, pIdx] = -2.0
                lapTheta[pIdx, iPrev * nZ + j] = 1.0
                lapTheta[pIdx, iNext * nZ + j] = 1.0
                if 0 < j < nZ - 1:
                    lapZ[pIdx, pIdx] = -2.0
                    lapZ[pIdx, i * nZ + (j - 1)] = 1.0
                    lapZ[pIdx, i * nZ + (j + 1)] = 1.0

        # NOTE: Balance azimuthal/axial biharmonic curvature smoothness against local slope expressivity.
        regMatrix = (
            1.6 * (lapTheta.T @ lapTheta)
            + 2.6 * (lapZ.T @ lapZ)
            + 0.5 * np.eye(numParams, dtype=np.float32)
        )
        jtjInv = np.linalg.inv(jacFull.T @ jacFull + regMatrix).astype(np.float32)

        innerWorkMask = rrWork < (scanData.effectiveRadiusMm - 0.48)
        meanInnerMeas = float(np.mean(measWork[innerWorkMask]))
        iterHistoryRmsePct: list[float] = []

        def evalCandidateWork(candGrid: np.ndarray, currGain: float) -> tuple[np.ndarray, float, float, float, float]:
            simRaw = simulateWorkGrid(candGrid)
            gUpd = float(
                np.dot(simRaw[innerWorkMask], measWork[innerWorkMask])
                / (np.dot(simRaw[innerWorkMask], simRaw[innerWorkMask]) + 1e-12)
            )
            newGain = currGain * gUpd
            simScaled = simRaw * gUpd
            rMv = float(np.sqrt(np.mean((measWork[innerWorkMask] - simScaled[innerWorkMask]) ** 2)))
            rPct = (rMv / meanInnerMeas) * 100.0
            cVal = float(np.corrcoef(measWork[innerWorkMask], simScaled[innerWorkMask])[0, 1])
            return simScaled, newGain, rMv, rPct, cVal

        simCurr, gainScaleMv, rmseMv, rmsePct, corrWork = evalCandidateWork(nonAxiGrid, gainScaleMv)
        iterHistoryRmsePct.append(rmsePct)
        self.logger.info(
            "[%.1fs] Iteration 1/9: inner RMSE=%.3f mV (%.2f%%), r=%.4f, peak_deltaR=%.0f um",
            time.time() - tStart,
            rmseMv,
            rmsePct,
            corrWork,
            float(np.max(np.abs(nonAxiGrid)) * 1000.0),
        )

        for iterIdx in range(1, 9):
            residualVec = (measWork - simCurr)[fitMaskWork]
            rhsVec = jacFull.T @ residualVec - 0.35 * (regMatrix @ nonAxiGrid.reshape(-1))
            stepGrid = (jtjInv @ rhsVec).reshape(nTh, nZ)
            stepGrid -= np.mean(stepGrid, axis=0, keepdims=True)

            # NOTE: Backtracking line search using exact nonlinear specular ray tracing guarantees monotonic descent.
            for alpha in (0.65, 0.38, 0.20, 0.10):
                candGrid = np.clip(nonAxiGrid + alpha * stepGrid, -0.55, 0.55)
                candSim, candGain, candRmseMv, candRmsePct, candCorr = evalCandidateWork(candGrid, gainScaleMv)
                if candRmsePct < rmsePct - 1e-4:
                    nonAxiGrid = candGrid
                    simCurr = candSim
                    gainScaleMv = candGain
                    rmseMv = candRmseMv
                    rmsePct = candRmsePct
                    corrWork = candCorr
                    break

            iterHistoryRmsePct.append(rmsePct)
            self.logger.info(
                "[%.1fs] Iteration %d/9: inner RMSE=%.3f mV (%.2f%%), r=%.4f, peak_deltaR=%.0f um",
                time.time() - tStart,
                iterIdx + 1,
                rmseMv,
                rmsePct,
                corrWork,
                float(np.max(np.abs(nonAxiGrid)) * 1000.0),
            )

        # ==================== Stage 4: Full 100x100 Independent Sobol Validation ====================
        bundleVal = self.rayFactory(32768, self.device, cosineExponent=cosineExponent, seed=20261111)

        surface.updateControlGrid(np.zeros(nZ, dtype=np.float32), np.zeros((nTh, nZ), dtype=np.float32))
        idealRaw = tracer.tracePointCloud(
            xxFull.reshape(-1), yyFull.reshape(-1), bundleTrain, reflectivity=reflectivity
        ).reshape(xxFull.shape)
        scanStepMm = float(scanData.xCoordsMm[1] - scanData.xCoordsMm[0])
        idealBlur = gaussian_filter(idealRaw, sigma=scanData.beamSigmaMm / scanStepMm)
        idealScale = float(
            np.mean(scanData.netMvGrid[rrFull < 5.0]) / max(1e-9, float(np.mean(idealBlur[rrFull < 5.0])))
        )
        idealFullNetMv = (idealScale * idealBlur).astype(np.float32)

        surface.updateControlGrid(axiProfile, nonAxiGrid)
        simFullRaw = tracer.tracePointCloud(
            xxFull.reshape(-1), yyFull.reshape(-1), bundleVal, reflectivity=reflectivity
        ).reshape(xxFull.shape)
        simFullBlur = gaussian_filter(simFullRaw, sigma=scanData.beamSigmaMm / scanStepMm)

        innerFullMask = rrFull < (scanData.effectiveRadiusMm - 0.48)
        finalGain = float(
            np.dot(simFullBlur[innerFullMask], scanData.netMvGrid[innerFullMask])
            / (np.dot(simFullBlur[innerFullMask], simFullBlur[innerFullMask]) + 1e-12)
        )
        simFullNetMv = (finalGain * simFullBlur).astype(np.float32)

        finalRmseMv = float(np.sqrt(np.mean((scanData.netMvGrid[innerFullMask] - simFullNetMv[innerFullMask]) ** 2)))
        meanInnerFull = float(np.mean(scanData.netMvGrid[innerFullMask]))
        finalRmsePct = (finalRmseMv / meanInnerFull) * 100.0

        validCircleMask = rrFull < (scanData.effectiveRadiusMm + 0.15)
        totalCorr = float(np.corrcoef(scanData.netMvGrid[validCircleMask], simFullNetMv[validCircleMask])[0, 1])

        measBumpPct = computeAzimuthalResidualPct(scanData.netMvGrid, rrFull, meanInnerFull)
        simBumpPct = computeAzimuthalResidualPct(simFullNetMv, rrFull, meanInnerFull)
        measBumpSmooth = gaussian_filter(measBumpPct, sigma=1.2)
        nonAxiCorrRaw = float(np.corrcoef(measBumpPct[innerFullMask], simBumpPct[innerFullMask])[0, 1])
        nonAxiCorrSmooth = float(np.corrcoef(measBumpSmooth[innerFullMask], simBumpPct[innerFullMask])[0, 1])

        snap4 = self.memoryGuard.inspectMemory("stage4_full_validation")
        self.logger.info(
            "[%.1fs] Full 100x100 validation done! Inner RMSE=%.3f mV (%.2f%%), total r=%.4f, bump r=%.4f (raw r=%.4f) | Peak RSS=%.0f MB",
            time.time() - tStart,
            finalRmseMv,
            finalRmsePct,
            totalCorr,
            nonAxiCorrSmooth,
            nonAxiCorrRaw,
            snap4.rssMb,
        )

        metrics = {
            "innerRmseMv": finalRmseMv,
            "innerRmsePct": finalRmsePct,
            "noiseFloorPct": scanData.noiseFloorPct,
            "totalCorr": totalCorr,
            "nonAxiCorr": nonAxiCorrSmooth,
            "nonAxiCorrRaw": nonAxiCorrRaw,
            "beamSigmaMm": scanData.beamSigmaMm,
            "effectiveRadiusMm": scanData.effectiveRadiusMm,
            "peakBulgeUm": float(np.min(nonAxiGrid) * 1000.0),
            "peakDentUm": float(np.max(nonAxiGrid) * 1000.0),
            "peakRssMb": snap4.rssMb,
        }

        return InversionResult(
            axiProfileMm=axiProfile,
            nonAxiGridMm=nonAxiGrid,
            reflectivity=reflectivity,
            cosineExponent=cosineExponent,
            gainScaleMv=finalGain,
            simFullNetMv=simFullNetMv,
            idealFullNetMv=idealFullNetMv,
            iterHistoryRmsePct=iterHistoryRmsePct,
            metrics=metrics,
        )
