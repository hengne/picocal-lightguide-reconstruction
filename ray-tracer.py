"""
Hybrid GPU/CPU Exact Specular Multi-Bounce Ray-Tracing Module

Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>

Simulates diffuse point light sources across all (X, Y) scan positions inside the circular
entrance window, tracing up to 20 exact specular reflections off the deformed bicubic B-spline
inner mirror wall and recording light collection efficiency at the exit PMT aperture.
Supports transparent three-tier hardware dispatch:
  1. Apple Silicon Metal GPU Compute Shader (`libmetal_raytracer.dylib`)
  2. Cross-platform C++17 Multi-Core Thread Pool (`libraytracer.{dylib,so,dll}`)
  3. Chunked PyTorch Vectorized Fallback (CUDA / MPS / CPU)
"""

import ctypes
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import qmc

__author__ = "Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>"


def _loadMetalLibrary() -> Any:
    """
    Attempt to load the compiled Apple Metal GPU ray-tracing shared library on macOS.
    """
    if sys.platform != "darwin":
        return None
    metalPath = Path(__file__).resolve().parent / "libmetal_raytracer.dylib"
    if not metalPath.exists():
        return None
    try:
        mlib = ctypes.CDLL(str(metalPath))
        fp = ctypes.POINTER(ctypes.c_float)
        mlib.tracePointsMetalGpu.argtypes = [
            ctypes.c_int, fp, fp, ctypes.c_int, fp, fp, fp,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_int, ctypes.c_int, fp, ctypes.c_float, ctypes.c_int, fp,
        ]
        mlib.tracePointsMetalGpu.restype = ctypes.c_int
        return mlib
    except OSError:
        return None


def _loadNativeCpuLibrary() -> Any:
    """
    Attempt to load the compiled cross-platform C++17 multi-core ray-tracing library.
    """
    baseDir = Path(__file__).resolve().parent
    candidates = [
        baseDir / "libraytracer.dylib",
        baseDir / "libraytracer.so",
        baseDir / "libraytracer.dll",
    ]
    for cand in candidates:
        if cand.exists():
            try:
                lib = ctypes.CDLL(str(cand))
                fp = ctypes.POINTER(ctypes.c_float)
                lib.tracePointsParallel.argtypes = [
                    ctypes.c_int, fp, fp, ctypes.c_int, fp, fp, fp,
                    ctypes.c_float, ctypes.c_float, ctypes.c_float,
                    ctypes.c_int, ctypes.c_int, fp, ctypes.c_float, ctypes.c_int, ctypes.c_int, fp,
                ]
                return lib
            except OSError:
                continue
    return None


_METAL_LIB: Any = _loadMetalLibrary()
_NATIVE_LIB: Any = _loadNativeCpuLibrary()


def reloadAcceleratorLibraries() -> None:
    """
    Reload native GPU/CPU shared libraries after on-the-fly compilation.
    """
    global _METAL_LIB, _NATIVE_LIB
    _METAL_LIB = _loadMetalLibrary()
    _NATIVE_LIB = _loadNativeCpuLibrary()


@dataclass(frozen=True)
class RayBundle:
    """
    Quasi-Monte Carlo (scrambled 2D Sobol sequence) diffuse initial ray direction bundle.
    """

    dx: torch.Tensor
    dy: torch.Tensor
    dz: torch.Tensor


@dataclass(frozen=True)
class SampledTrajectory:
    """
    Single multi-bounce 3D photon trajectory sampled for interactive visual rendering.
    """

    pointsMm: list[tuple[float, float, float]]
    reachedPmt: bool
    bounceCount: int


def createDiffuseRayBundle(
    numRays: int,
    device: torch.device,
    cosineExponent: float = 1.0,
    seed: int = 42,
) -> RayBundle:
    """
    Generate low-discrepancy hemispherical diffuse ray directions via scrambled Sobol sampling.
    """
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    uv = sampler.random(numRays).astype(np.float32)
    phi = (2.0 * math.pi) * uv[:, 0]
    cosTheta = np.power(np.clip(1.0 - uv[:, 1], 1e-7, 1.0), 1.0 / (cosineExponent + 1.0)).astype(np.float32)
    sinTheta = np.sqrt(np.clip(1.0 - cosTheta * cosTheta, 0.0, 1.0)).astype(np.float32)
    dxNp = (sinTheta * np.cos(phi)).astype(np.float32)
    dyNp = (sinTheta * np.sin(phi)).astype(np.float32)
    return RayBundle(
        dx=torch.from_numpy(dxNp).to(device=device),
        dy=torch.from_numpy(dyNp).to(device=device),
        dz=torch.from_numpy(cosTheta).to(device=device),
    )


class SpecularLightGuideTracer:
    """
    Exact multi-bounce specular ray tracer for deformed conical light guides.

    Uses a concentric local reference cone through the current wall hit radius (making quadratic
    constant C == 0 identically) to eliminate self-intersection roots in outward-dented regions,
    followed by Newton refinement onto the exact bicubic B-spline surface.
    """

    def __init__(
        self,
        surfaceModel: Any,
        memoryGuard: Any,
        maxBounces: int = 22,
        chunkPoints: int = 256,
    ) -> None:
        """
        Initialize the ray tracer with a surface model and memory safety watchdog.
        """
        self.surfaceModel: Any = surfaceModel
        self.memoryGuard: Any = memoryGuard
        self.maxBounces: int = maxBounces
        self.chunkPoints: int = chunkPoints

    @torch.no_grad()
    def tracePointCloud(
        self,
        sourceXMm: np.ndarray,
        sourceYMm: np.ndarray,
        rayBundle: RayBundle,
        reflectivity: float = 0.955,
    ) -> np.ndarray:
        """
        Compute PMT light collection efficiency for an array of entrance (X, Y) points.
        """
        if _METAL_LIB is not None or _NATIVE_LIB is not None:
            fp = ctypes.POINTER(ctypes.c_float)
            pxArr = np.ascontiguousarray(sourceXMm, dtype=np.float32)
            pyArr = np.ascontiguousarray(sourceYMm, dtype=np.float32)
            dxArr = np.ascontiguousarray(rayBundle.dx.cpu().numpy(), dtype=np.float32)
            dyArr = np.ascontiguousarray(rayBundle.dy.cpu().numpy(), dtype=np.float32)
            dzArr = np.ascontiguousarray(rayBundle.dz.cpu().numpy(), dtype=np.float32)
            ctrlArr = np.ascontiguousarray(
                self.surfaceModel.nonAxiGridMm + self.surfaceModel.axiProfileMm[None, :], dtype=np.float32
            )
            outEff = np.zeros_like(pxArr, dtype=np.float32)

            if _METAL_LIB is not None:
                rc = _METAL_LIB.tracePointsMetalGpu(
                    int(pxArr.shape[0]),
                    pxArr.ctypes.data_as(fp),
                    pyArr.ctypes.data_as(fp),
                    int(dxArr.shape[0]),
                    dxArr.ctypes.data_as(fp),
                    dyArr.ctypes.data_as(fp),
                    dzArr.ctypes.data_as(fp),
                    float(self.surfaceModel.geometry.inputRadiusMm),
                    float(self.surfaceModel.geometry.coneSlope),
                    float(self.surfaceModel.geometry.coneLengthMm),
                    int(self.surfaceModel.numTheta),
                    int(self.surfaceModel.numZ),
                    ctrlArr.ctypes.data_as(fp),
                    float(reflectivity),
                    int(self.maxBounces),
                    outEff.ctypes.data_as(fp),
                )
                if rc == 0:
                    self.memoryGuard.inspectMemory("tracePointCloudMetalGpu")
                    return outEff

            if _NATIVE_LIB is not None:
                numThreads = min(16, max(2, (os.cpu_count() or 8) - 2))
                _NATIVE_LIB.tracePointsParallel(
                    int(pxArr.shape[0]),
                    pxArr.ctypes.data_as(fp),
                    pyArr.ctypes.data_as(fp),
                    int(dxArr.shape[0]),
                    dxArr.ctypes.data_as(fp),
                    dyArr.ctypes.data_as(fp),
                    dzArr.ctypes.data_as(fp),
                    float(self.surfaceModel.geometry.inputRadiusMm),
                    float(self.surfaceModel.geometry.coneSlope),
                    float(self.surfaceModel.geometry.coneLengthMm),
                    int(self.surfaceModel.numTheta),
                    int(self.surfaceModel.numZ),
                    ctrlArr.ctypes.data_as(fp),
                    float(reflectivity),
                    int(self.maxBounces),
                    int(numThreads),
                    outEff.ctypes.data_as(fp),
                )
                self.memoryGuard.inspectMemory("tracePointCloudNative")
                return outEff

        device = self.surfaceModel.device
        coneLen = self.surfaceModel.geometry.coneLengthMm
        slope = self.surfaceModel.geometry.coneSlope

        pxAll = torch.tensor(sourceXMm, dtype=torch.float32, device=device)
        pyAll = torch.tensor(sourceYMm, dtype=torch.float32, device=device)
        totalPoints = int(pxAll.shape[0])

        r0 = torch.sqrt(pxAll * pxAll + pyAll * pyAll)
        th0 = torch.atan2(pyAll, pxAll)
        rInLimit, _, _ = self.surfaceModel.evaluateRadiusAndGradients(th0, torch.zeros_like(r0))

        # NOTE: Only trace source points inside the clear entrance pupil; exterior points yield zero efficiency.
        insideMask = r0 < (rInLimit - 1e-4)
        validIndices = torch.nonzero(insideMask, as_tuple=False).squeeze(-1)
        numValid = int(validIndices.shape[0])
        numRays = int(rayBundle.dx.shape[0])

        efficiencies = torch.zeros(totalPoints, dtype=torch.float32, device=device)

        for startIdx in range(0, numValid, self.chunkPoints):
            subIdx = validIndices[startIdx : min(numValid, startIdx + self.chunkPoints)]
            batchSize = int(subIdx.shape[0])

            x = pxAll[subIdx, None].expand(batchSize, numRays).clone()
            y = pyAll[subIdx, None].expand(batchSize, numRays).clone()
            z = torch.zeros((batchSize, numRays), dtype=torch.float32, device=device)
            dx = rayBundle.dx[None, :].expand(batchSize, numRays).clone()
            dy = rayBundle.dy[None, :].expand(batchSize, numRays).clone()
            dz = rayBundle.dz[None, :].expand(batchSize, numRays).clone()

            weight = torch.ones((batchSize, numRays), dtype=torch.float32, device=device)
            alive = torch.ones((batchSize, numRays), dtype=torch.bool, device=device)
            collected = torch.zeros((batchSize, numRays), dtype=torch.float32, device=device)

            for bounce in range(self.maxBounces):
                if not alive.any():
                    break

                tzOut = torch.where(dz > 1e-6, (coneLen - z) / dz, torch.full_like(z, 1e9))
                tzIn = torch.where((dz < -1e-6) & (bounce > 0), -z / dz, torch.full_like(z, 1e9))

                if bounce == 0:
                    rzStart, _, _ = self.surfaceModel.evaluateRadiusAndGradients(torch.atan2(y, x), z)
                    quadA = dx * dx + dy * dy - (slope * slope) * dz * dz
                    quadB = 2.0 * (x * dx + y * dy - rzStart * slope * dz)
                    quadC = x * x + y * y - rzStart * rzStart
                    disc = torch.clamp(quadB * quadB - 4.0 * quadA * quadC, min=0.0)
                    tEst = (-quadB + torch.sqrt(disc)) / (
                        2.0 * torch.where(torch.abs(quadA) > 1e-6, quadA, torch.ones_like(quadA))
                    )
                else:
                    # NOTE: Construct a concentric reference cone passing through rCurr so C == 0 identically.
                    rCurr = torch.sqrt(x * x + y * y + 1e-10)
                    quadA = dx * dx + dy * dy - (slope * slope) * dz * dz
                    quadB = 2.0 * (x * dx + y * dy - rCurr * slope * dz)
                    tEst = -quadB / torch.where(torch.abs(quadA) > 1e-6, quadA, torch.ones_like(quadA))

                tVal = torch.clamp(tEst, min=0.05, max=150.0)

                for _ in range(4):
                    xt = x + tVal * dx
                    yt = y + tVal * dy
                    zt = torch.clamp(z + tVal * dz, 0.0, coneLen)
                    rt = torch.sqrt(xt * xt + yt * yt + 1e-12)
                    tht = torch.atan2(yt, xt)
                    rSurf, drDth, drDz = self.surfaceModel.evaluateRadiusAndGradients(tht, zt)
                    fVal = rt - rSurf
                    dfDt = (xt * dx + yt * dy) / rt - drDth * ((xt * dy - yt * dx) / (rt * rt)) - drDz * dz
                    denom = torch.where(torch.abs(dfDt) > 1e-5, dfDt, torch.ones_like(dfDt))
                    tVal = torch.clamp(tVal - fVal / denom, min=1e-3, max=160.0)

                zWall = z + tVal * dz
                xt = x + tVal * dx
                yt = y + tVal * dy
                rCheck, _, _ = self.surfaceModel.evaluateRadiusAndGradients(
                    torch.atan2(yt, xt), torch.clamp(zWall, 0.0, coneLen)
                )
                validWall = (
                    (tVal > 2e-3)
                    & (zWall > 0.0)
                    & (zWall < coneLen)
                    & (torch.abs(torch.sqrt(xt * xt + yt * yt) - rCheck) < 0.025)
                )
                tWall = torch.where(validWall, tVal, torch.full_like(z, 1e9))

                hitOut = alive & (tzOut <= tWall) & (tzOut <= tzIn) & (tzOut < 1e8)
                hitIn = alive & (~hitOut) & (tzIn <= tWall) & (tzIn < 1e8)
                hitWall = alive & (~hitOut) & (~hitIn) & (tWall < 1e8)

                xOut = x + tzOut * dx
                yOut = y + tzOut * dy
                rEnd, _, _ = self.surfaceModel.evaluateRadiusAndGradients(
                    torch.atan2(yOut, xOut), torch.full_like(xOut, coneLen)
                )
                inAperture = (xOut * xOut + yOut * yOut) <= (rEnd + 0.025) ** 2
                collected = torch.where(hitOut & inAperture, collected + weight, collected)

                alive = alive & hitWall
                if not alive.any():
                    break

                x = torch.where(hitWall, x + tWall * dx, x)
                y = torch.where(hitWall, y + tWall * dy, y)
                z = torch.where(hitWall, z + tWall * dz, z)
                weight = torch.where(hitWall, weight * reflectivity, weight)

                rHit = torch.sqrt(x * x + y * y + 1e-12)
                thHit = torch.atan2(y, x)
                _, drDth, drDz = self.surfaceModel.evaluateRadiusAndGradients(thHit, z)
                gx = x / rHit - drDth * (-y / (rHit * rHit))
                gy = y / rHit - drDth * (x / (rHit * rHit))
                gz = -drDz
                gNorm = torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
                nx, ny, nz = -gx / gNorm, -gy / gNorm, -gz / gNorm

                dotProd = dx * nx + dy * ny + dz * nz
                dx = torch.where(hitWall, dx - 2.0 * dotProd * nx, dx)
                dy = torch.where(hitWall, dy - 2.0 * dotProd * ny, dy)
                dz = torch.where(hitWall, dz - 2.0 * dotProd * nz, dz)

            efficiencies[subIdx] = collected.mean(dim=1)

        self.memoryGuard.inspectMemory("tracePointCloud")
        return efficiencies.cpu().numpy()

    @torch.no_grad()
    def sampleRayTrajectories(
        self,
        sourcePointXY: tuple[float, float],
        numSampleRays: int = 32,
        seed: int = 108,
    ) -> list[SampledTrajectory]:
        """
        Sample representative 3D photon bounce trajectories from a given entrance (X, Y) coordinate.
        """
        device = self.surfaceModel.device
        coneLen = self.surfaceModel.geometry.coneLengthMm
        slope = self.surfaceModel.geometry.coneSlope

        bundle = createDiffuseRayBundle(numSampleRays, device=device, cosineExponent=1.2, seed=seed)
        trajectories: list[SampledTrajectory] = []

        for idx in range(numSampleRays):
            currX = float(sourcePointXY[0])
            currY = float(sourcePointXY[1])
            currZ = 0.0
            currDx = float(bundle.dx[idx].item())
            currDy = float(bundle.dy[idx].item())
            currDz = float(bundle.dz[idx].item())

            pts: list[tuple[float, float, float]] = [(currX, currY, currZ)]
            reached = False
            bounces = 0

            for bounce in range(self.maxBounces):
                tzOut = (coneLen - currZ) / currDz if currDz > 1e-6 else 1e9
                tzIn = -currZ / currDz if (currDz < -1e-6 and bounce > 0) else 1e9

                xT = torch.tensor([currX], dtype=torch.float32, device=device)
                yT = torch.tensor([currY], dtype=torch.float32, device=device)
                zT = torch.tensor([currZ], dtype=torch.float32, device=device)

                if bounce == 0:
                    rzStart, _, _ = self.surfaceModel.evaluateRadiusAndGradients(torch.atan2(yT, xT), zT)
                    rInit = float(rzStart.item())
                    qa = currDx * currDx + currDy * currDy - slope * slope * currDz * currDz
                    qb = 2.0 * (currX * currDx + currY * currDy - rInit * slope * currDz)
                    qc = currX * currX + currY * currY - rInit * rInit
                    disc = max(0.0, qb * qb - 4.0 * qa * qc)
                    tVal = (-qb + math.sqrt(disc)) / (2.0 * (qa if abs(qa) > 1e-6 else 1.0))
                else:
                    rCurr = math.hypot(currX, currY)
                    qa = currDx * currDx + currDy * currDy - slope * slope * currDz * currDz
                    qb = 2.0 * (currX * currDx + currY * currDy - rCurr * slope * currDz)
                    tVal = -qb / (qa if abs(qa) > 1e-6 else 1.0)

                tVal = max(0.05, min(150.0, tVal))
                for _ in range(5):
                    xt = currX + tVal * currDx
                    yt = currY + tVal * currDy
                    zt = max(0.0, min(coneLen, currZ + tVal * currDz))
                    rt = math.hypot(xt, yt) + 1e-12
                    rS, drDth, drDz = self.surfaceModel.evaluateRadiusAndGradients(
                        torch.tensor([math.atan2(yt, xt)], dtype=torch.float32, device=device),
                        torch.tensor([zt], dtype=torch.float32, device=device),
                    )
                    fVal = rt - float(rS.item())
                    dfDt = (
                        (xt * currDx + yt * currDy) / rt
                        - float(drDth.item()) * ((xt * currDy - yt * currDx) / (rt * rt))
                        - float(drDz.item()) * currDz
                    )
                    if abs(dfDt) > 1e-6:
                        tVal = max(1e-3, min(160.0, tVal - fVal / dfDt))

                zWall = currZ + tVal * currDz
                tWall = tVal if (tVal > 2e-3 and 0.0 < zWall < coneLen) else 1e9

                if tzOut <= tWall and tzOut <= tzIn and tzOut < 1e8:
                    pts.append((currX + tzOut * currDx, currY + tzOut * currDy, coneLen))
                    reached = True
                    break
                if tzIn <= tWall and tzIn < 1e8:
                    pts.append((currX + tzIn * currDx, currY + tzIn * currDy, 0.0))
                    reached = False
                    break
                if tWall >= 1e8:
                    break

                currX += tWall * currDx
                currY += tWall * currDy
                currZ += tWall * currDz
                pts.append((currX, currY, currZ))
                bounces += 1

                rHit = math.hypot(currX, currY) + 1e-12
                _, drDth, drDz = self.surfaceModel.evaluateRadiusAndGradients(
                    torch.tensor([math.atan2(currY, currX)], dtype=torch.float32, device=device),
                    torch.tensor([currZ], dtype=torch.float32, device=device),
                )
                gx = currX / rHit - float(drDth.item()) * (-currY / (rHit * rHit))
                gy = currY / rHit - float(drDth.item()) * (currX / (rHit * rHit))
                gz = -float(drDz.item())
                gNorm = math.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
                nx, ny, nz = -gx / gNorm, -gy / gNorm, -gz / gNorm
                dot = currDx * nx + currDy * ny + currDz * nz
                currDx -= 2.0 * dot * nx
                currDy -= 2.0 * dot * ny
                currDz -= 2.0 * dot * nz

            trajectories.append(SampledTrajectory(pointsMm=pts, reachedPmt=reached, bounceCount=bounces))

        return trajectories
