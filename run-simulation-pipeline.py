"""
Main End-to-End Simulation & Inverse Surface Reconstruction Pipeline for LHCb PicoCal Light Guides.

Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
Affiliation: South China Normal University (SCNU) / LHCb Collaboration, CERN

Executes the complete closed-loop scientific workflow:
1. Cross-platform memory and thread-cap safety initialization.
2. On-demand compilation of Apple Metal GPU / multi-core C++17 ray-tracing accelerators.
3. Automatic STEP CAD geometry parsing and 2D XY laser scan calibration.
4. Multi-stage nonlinear specular ray-tracing inverse reconstruction of inner-wall deformations.
5. Export of 3D watertight binary STL mesh, unwrapped surface grid CSV, and 10,000-point simulated scan CSV.
6. Portable auto-detection, compilation, and multi-process execution of Geant4 (G4OpticalPhysics) cross-validation.
7. Generation of bilingual (Chinese & English) zoomable interactive vector SVG reports.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

__author__ = "Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>"


def loadLocalModule(moduleAlias: str, filePath: Path) -> Any:
    """
    Dynamically import a local kebab-case Python module by file path.
    """
    spec = importlib.util.spec_from_file_location(moduleAlias, filePath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module file: {filePath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[moduleAlias] = module
    spec.loader.exec_module(module)
    return module


def ensureAcceleratorsCompiled(projectDir: Path, rtMod: Any, logger: logging.Logger) -> None:
    """
    Ensure that native C++17 and/or Apple Metal GPU ray-tracing shared libraries exist and are
    up-to-date with their C++/Objective-C++ sources, invoking compile-accelerators.py when needed.
    """
    metalLib = projectDir / "libmetal_raytracer.dylib"
    metalSrc = projectDir / "ray-tracer-metal.mm"
    nativeSrc = projectDir / "ray-tracer-native.cc"
    nativeLibs = [projectDir / name for name in ("libraytracer.dylib", "libraytracer.so", "libraytracer.dll")]

    metalUpToDate = (
        sys.platform != "darwin"
        or (metalLib.exists() and (not metalSrc.exists() or metalLib.stat().st_mtime >= metalSrc.stat().st_mtime))
    )
    existingNative = next((p for p in nativeLibs if p.exists()), None)
    nativeUpToDate = (
        existingNative is not None
        and (not nativeSrc.exists() or existingNative.stat().st_mtime >= nativeSrc.stat().st_mtime)
    )
    if metalUpToDate and nativeUpToDate:
        return

    compilerScript = projectDir / "compile-accelerators.py"
    if not compilerScript.exists():
        return

    logger.info("Compiling native hardware ray-tracing accelerators via %s ...", compilerScript.name)
    try:
        compMod = loadLocalModule("compile_accelerators", compilerScript)
        compMod.compileAllAccelerators(projectDir)
        if hasattr(rtMod, "reloadAcceleratorLibraries"):
            rtMod.reloadAcceleratorLibraries()
    except Exception as exc:
        logger.warning("Native accelerator build skipped (%s); falling back to PyTorch engine.", exc)


def discoverGeant4Toolchain() -> tuple[Path | None, str | None]:
    """
    Portably auto-discover `geant4-config` and a C++17 compiler across PATH, CONDA_PREFIX,
    and standard Conda/Miniforge environments without any hardcoded usernames.
    """
    candidates: list[Path] = []

    envCfg = os.environ.get("GEANT4_CONFIG", "").strip()
    if envCfg:
        candidates.append(Path(envCfg))

    whichCfg = shutil.which("geant4-config")
    if whichCfg:
        candidates.append(Path(whichCfg))

    condaPrefix = os.environ.get("CONDA_PREFIX", "").strip()
    if condaPrefix:
        candidates.append(Path(condaPrefix) / "bin" / "geant4-config")

    # Search conda base environments if conda is installed
    condaRoots: list[Path] = []
    condaExe = shutil.which("conda")
    if condaExe:
        try:
            baseStr = subprocess.check_output([condaExe, "info", "--base"], text=True, timeout=5).strip()
            if baseStr:
                condaRoots.append(Path(baseStr))
        except (subprocess.SubprocessError, OSError):
            pass

    homeDir = Path.home()
    for folderName in ("miniconda3", "miniforge3", "mambaforge", "anaconda3"):
        candRoot = homeDir / folderName
        if candRoot.is_dir() and candRoot not in condaRoots:
            condaRoots.append(candRoot)

    for root in condaRoots:
        envsDir = root / "envs"
        if not envsDir.is_dir():
            continue
        # Prioritize environments named g4 or geant4
        for preferred in ("g4", "geant4"):
            prefCfg = envsDir / preferred / "bin" / "geant4-config"
            if prefCfg.exists():
                candidates.append(prefCfg)
        for subEnv in sorted(envsDir.iterdir()):
            subCfg = subEnv / "bin" / "geant4-config"
            if subCfg.exists() and subCfg not in candidates:
                candidates.append(subCfg)

    g4ConfigBin: Path | None = next((p for p in candidates if p.is_file()), None)

    cxxEnv = os.environ.get("CXX", "").strip()
    cxxBin: str | None = None
    if platform.system() == "Darwin" and Path("/usr/bin/clang++").is_file():
        # Prefer Apple's native system clang++ on macOS so headers match MacOSX.sdk builtins
        cxxBin = "/usr/bin/clang++"
    elif cxxEnv and shutil.which(cxxEnv):
        cxxBin = shutil.which(cxxEnv)
    else:
        preferredCompilers = ("clang++", "g++", "c++") if platform.system() == "Darwin" else ("g++", "clang++", "c++")
        for comp in preferredCompilers:
            found = shutil.which(comp)
            if found:
                cxxBin = found
                break

    return g4ConfigBin, cxxBin


def runGeant4Validation(
    projectDir: Path,
    outputsDir: Path,
    reflectivity: float,
    cosineExponent: float,
    gainScaleMv: float,
    sliceYLocalMm: float,
    photonsPerPt: int = 8000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compile and run independent multi-process Geant4 optical-photon Monte Carlo validation
    along a representative cross-section slice across the reconstructed deformed mesh.
    """
    logger = logging.getLogger("agy.pipeline")

    sliceXPoints = np.linspace(-7.0, 7.0, 15, dtype=np.float32)
    ptsFile = outputsDir / "geant4-slice-points.txt"
    ptsLines = [f"{float(xv):.4f} {sliceYLocalMm:.4f}" for xv in sliceXPoints]
    ptsFile.write_text("\n".join(ptsLines) + "\n", encoding="utf-8")

    g4ConfigBin, cxxBin = discoverGeant4Toolchain()
    if g4ConfigBin is None or cxxBin is None:
        logger.warning("Geant4 toolchain not found on this machine; skipping optional Geant4 cross-validation.")
        return np.array([]), np.array([]), np.array([])

    g4Prefix = g4ConfigBin.parent.parent
    srcFile = projectDir / "geant4-validator" / "geant4-light-guide.cc"
    binFile = projectDir / "geant4-validator" / "geant4-light-guide-bin"
    gridFile = outputsDir / "geant4-surface-grid.txt"
    outCsv = outputsDir / "geant4-slice-validation.csv"

    try:
        if not binFile.exists() or binFile.stat().st_mtime < srcFile.stat().st_mtime:
            # Collapse multi-line geant4-config flags into a single whitespace-separated list
            rawFlags = subprocess.check_output(
                [str(g4ConfigBin), "--cflags", "--libs"], text=True, timeout=15
            )
            flagsOut = " ".join(rawFlags.split())
            rpathFlag = f"-Wl,-rpath,'{g4Prefix / 'lib'}'" if platform.system() in ("Darwin", "Linux") else ""
            compileCmd = (
                f"'{cxxBin}' -O3 -std=c++17 '{srcFile}' {flagsOut} "
                f"{rpathFlag} -o '{binFile}'"
            )
            compileEnv = dict(os.environ)
            if platform.system() == "Darwin" and cxxBin == "/usr/bin/clang++":
                compileEnv.pop("CONDA_BUILD_SYSROOT", None)
                compileEnv.pop("SDKROOT", None)
            subprocess.run(compileCmd, shell=True, check=True, timeout=90, env=compileEnv)
            logger.info("Compiled Geant4 optical validator binary: %s", binFile)

        # Resolve Geant4 physics dataset directories, handling conda-forge stripped 'G4' prefixes
        dsOut = subprocess.check_output([str(g4ConfigBin), "--datasets"], text=True, timeout=10)
        runEnv = dict(os.environ)
        dataRoot = g4Prefix / "share" / "Geant4" / "data"
        existingDirs = [p for p in dataRoot.iterdir() if p.is_dir()] if dataRoot.exists() else []
        for line in dsOut.splitlines():
            tokens = line.strip().split()
            if len(tokens) == 3:
                envVar, rawPathStr = tokens[1], tokens[2]
                candPath = Path(rawPathStr)
                if not candPath.exists() and dataRoot.exists():
                    strippedName = candPath.name[2:] if candPath.name.startswith("G4") else candPath.name
                    altPath = dataRoot / strippedName
                    if altPath.exists():
                        candPath = altPath
                    else:
                        alphaPrefix = "".join(ch for ch in strippedName if ch.isalpha())
                        matched = [d for d in existingDirs if d.name.startswith(alphaPrefix)]
                        if matched:
                            candPath = matched[0]
                runEnv[envVar] = str(candPath)

        # Partition 15 test points across 5 parallel worker processes to saturate multi-core CPUs
        numShards = 5
        procs: list[tuple[int, subprocess.Popen[bytes], Path]] = []
        for sIdx in range(numShards):
            subPts = sliceXPoints[sIdx * 3 : (sIdx + 1) * 3]
            subPtsFile = outputsDir / f"geant4-pts-shard-{sIdx}.txt"
            subOutCsv = outputsDir / f"geant4-out-shard-{sIdx}.csv"
            subPtsFile.write_text(
                "\n".join(f"{float(xv):.4f} {sliceYLocalMm:.4f}" for xv in subPts) + "\n", encoding="utf-8"
            )
            proc = subprocess.Popen(
                [
                    str(binFile),
                    str(gridFile),
                    str(subPtsFile),
                    str(subOutCsv),
                    f"{reflectivity:.5f}",
                    f"{cosineExponent:.4f}",
                    str(photonsPerPt),
                ],
                env=runEnv,
            )
            procs.append((sIdx, proc, subOutCsv))

        shardDfs: list[pd.DataFrame] = []
        for _, proc, subOutCsv in procs:
            proc.wait(timeout=120)
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, str(binFile))
            shardDfs.append(pd.read_csv(subOutCsv))

        g4Df = pd.concat(shardDfs, ignore_index=True)
        g4Eff = g4Df["g4_efficiency"].to_numpy(dtype=np.float32)
        g4N = g4Df["n_photons"].to_numpy(dtype=np.float32)

        g4Mv = g4Eff * gainScaleMv
        g4ErrEff = np.sqrt(np.maximum(g4Eff * (1.0 - g4Eff), 1e-6) / g4N)
        g4ErrMv = g4ErrEff * gainScaleMv
        g4Df["g4_mv"] = np.round(g4Mv, 4)
        g4Df["g4_err_mv"] = np.round(g4ErrMv, 4)
        g4Df.to_csv(outCsv, index=False)
        logger.info(
            "Geant4 optical photon validation complete (%d points x %d photons/pt), mean signal=%.2f mV",
            len(sliceXPoints),
            photonsPerPt,
            float(np.mean(g4Mv)),
        )
        return sliceXPoints, g4Mv, g4ErrMv
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        logger.warning("Geant4 validation encountered an error (%s); continuing pipeline.", exc)
        return np.array([]), np.array([]), np.array([])


def resolveDefaultDataFile(projectDir: Path, filename: str) -> Path:
    """
    Resolve default bundled input files inside `data/<filename>` first, falling back to parent folder.
    """
    bundled = projectDir / "data" / filename
    if bundled.exists():
        return bundled
    fallback = projectDir.parent / filename
    if fallback.exists():
        return fallback
    return bundled


def parseCliArgs(projectDir: Path) -> argparse.Namespace:
    """
    Parse command-line options for reproducible runs and custom experimental scan datasets.
    """
    parser = argparse.ArgumentParser(
        description="LHCb PicoCal Conical Light-Guide Optical Simulation & Inverse Surface Reconstruction Pipeline"
    )
    parser.add_argument(
        "--step-file",
        type=Path,
        default=resolveDefaultDataFile(projectDir, "4cells_17.5mm_8mm_circle_to_circle_100mm-Cut.step"),
        help="Path to input STEP CAD geometry file (default: data/4cells_17.5mm_8mm_circle_to_circle_100mm-Cut.step)",
    )
    parser.add_argument(
        "--scan-csv",
        type=Path,
        default=resolveDefaultDataFile(projectDir, "scan_summary.csv"),
        help="Path to measured 2D XY laser scan CSV file (default: data/scan_summary.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=projectDir / "outputs",
        help="Directory for reconstructed STL, CSVs, metrics JSON, and logs (default: outputs/)",
    )
    parser.add_argument(
        "--skip-geant4",
        action="store_true",
        help="Skip optional Geant4 optical-photon cross-validation",
    )
    parser.add_argument(
        "--photons-per-point",
        type=int,
        default=8000,
        help="Number of G4OpticalPhotons per slice point for Geant4 validation (default: 8000)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the complete inverse surface reconstruction, dual-engine validation, and bilingual SVG report generation.
    """
    projectDir = Path(__file__).resolve().parent
    args = parseCliArgs(projectDir)
    outputsDir: Path = args.output_dir.resolve()
    outputsDir.mkdir(parents=True, exist_ok=True)

    mgMod = loadLocalModule("memory_guard", projectDir / "memory-guard.py")
    adaptiveProfile = mgMod.configureProcessLimits()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(outputsDir / "simulation-pipeline.log", mode="w", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("agy.pipeline")

    import torch

    torch.set_num_threads(1)
    device = torch.device("cpu")

    smMod = loadLocalModule("surface_model", projectDir / "surface-model.py")
    rtMod = loadLocalModule("ray_tracer", projectDir / "ray-tracer.py")
    invMod = loadLocalModule("surface_inverter", projectDir / "surface-inverter.py")
    svgMod = loadLocalModule("svg_builder", projectDir / "svg-builder.py")

    ensureAcceleratorsCompiled(projectDir, rtMod, logger)

    accelName = (
        "Apple Metal GPU"
        if getattr(rtMod, "_METAL_LIB", None) is not None
        else ("Native C++17 Multi-Core" if getattr(rtMod, "_NATIVE_LIB", None) is not None else "PyTorch Vectorized")
    )
    logger.info(
        "Starting PicoCal Light-Guide Pipeline | Engine: %s | Free RAM: %.1f GB",
        accelName,
        adaptiveProfile.freeMemoryGb,
    )

    memoryGuard = mgMod.MemoryGuard(maxMemoryMb=adaptiveProfile.memoryCapMb)

    stepPath = args.step_file.resolve()
    csvPath = args.scan_csv.resolve()
    if not stepPath.exists():
        raise FileNotFoundError(f"STEP geometry file not found: {stepPath}")
    if not csvPath.exists():
        raise FileNotFoundError(f"Scan CSV file not found: {csvPath}")

    geometry = smMod.parseStepGeometry(stepPath)
    scanData = invMod.loadAndCalibrateScanCsv(csvPath)

    inverter = invMod.LightGuideSurfaceInverter(
        geometry=geometry,
        surfaceClass=smMod.BicubicCylindricalSurface,
        tracerClass=rtMod.SpecularLightGuideTracer,
        rayFactory=rtMod.createDiffuseRayBundle,
        memoryGuard=memoryGuard,
        device=device,
        numThetaCtrl=36,
        numZCtrl=16,
        chunkPoints=adaptiveProfile.assignedChunkPoints,
    )
    invResult = inverter.reconstructSurface(scanData)

    # Construct final optimal freeform surface and export 3D STL, CSV, and Geant4 tessellation grid
    finalSurface = smMod.BicubicCylindricalSurface(
        geometry, invResult.axiProfileMm, invResult.nonAxiGridMm, device
    )
    thetaRad1d, zMm1d, totalRadiusMm, nonAxiDeltaRMm = finalSurface.sampleDenseGrid(resTheta=180, resZ=201)

    stlPath = outputsDir / "reconstructed-light-guide.stl"
    finalSurface.exportBinaryStl(stlPath, resTheta=120, resZ=121, wallThicknessMm=1.5)
    g4GridPath = outputsDir / "geant4-surface-grid.txt"
    finalSurface.exportGeant4GridFile(g4GridPath, resTheta=60, resZ=61)
    logger.info("Exported reconstructed 3D binary STL mesh: %s", stlPath)

    # Export unwrapped inner-surface deformation grid CSV
    thGrid, zGrid = np.meshgrid(np.degrees(thetaRad1d), zMm1d, indexing="ij")
    surfDf = pd.DataFrame(
        {
            "theta_deg": thGrid.reshape(-1),
            "z_mm": zGrid.reshape(-1),
            "radius_mm": totalRadiusMm.reshape(-1),
            "delta_r_non_axi_um": (nonAxiDeltaRMm * 1000.0).reshape(-1),
        }
    )
    surfDf.to_csv(outputsDir / "reconstructed-surface-grid.csv", index=False)

    # Export 10,000-point simulated XY scan CSV aligned with scan_summary.csv
    simScanDf = scanData.rawDf[["idx_x", "idx_y", "pos_x_mm", "pos_y_mm", "probe_V", "passed"]].copy()
    simVoltGrid = scanData.backgroundVolt + invResult.simFullNetMv / 1000.0
    simProbeList = [
        float(simVoltGrid[int(row.idx_y), int(row.idx_x)]) for row in simScanDf.itertuples(index=False)
    ]
    simScanDf["simulated_probe_V"] = np.round(simProbeList, 6)
    simScanDf.to_csv(outputsDir / "simulated-scan-summary.csv", index=False)

    # Run independent Geant4 optical-photon slice validation unless skipped
    sliceYLocal = 1.60  # Cross-section passing through the prominent upper-left valley (Y' = +1.6 mm)
    if args.skip_geant4:
        logger.info("Skipping Geant4 cross-validation (--skip-geant4 specified).")
        g4X, g4Mv, g4ErrMv = np.array([]), np.array([]), np.array([])
        cachedG4Csv = outputsDir / "geant4-slice-validation.csv"
        if cachedG4Csv.exists():
            cachedDf = pd.read_csv(cachedG4Csv)
            if {"x_local_mm", "g4_mv", "g4_err_mv"}.issubset(cachedDf.columns):
                g4X = cachedDf["x_local_mm"].to_numpy(dtype=np.float32)
                g4Mv = cachedDf["g4_mv"].to_numpy(dtype=np.float32)
                g4ErrMv = cachedDf["g4_err_mv"].to_numpy(dtype=np.float32)
    else:
        g4X, g4Mv, g4ErrMv = runGeant4Validation(
            projectDir=projectDir,
            outputsDir=outputsDir,
            reflectivity=invResult.reflectivity,
            cosineExponent=invResult.cosineExponent,
            gainScaleMv=invResult.gainScaleMv,
            sliceYLocalMm=sliceYLocal,
            photonsPerPt=args.photons_per_point,
        )

    # Sample representative 3D multi-bounce trajectories for Panel B visualization
    finalTracer = rtMod.SpecularLightGuideTracer(finalSurface, memoryGuard, maxBounces=20, chunkPoints=128)
    trajectories = finalTracer.sampleRayTrajectories(sourcePointXY=(-1.2, 1.8), numSampleRays=32, seed=314)

    # Assemble visualization payload for bilingual interactive SVGs
    scanXLocal = scanData.xCoordsMm - scanData.centerXMm
    scanYLocal = scanData.yCoordsMm - scanData.centerYMm
    xxLocal, yyLocal = np.meshgrid(scanXLocal, scanYLocal)
    rrLocal = np.sqrt(xxLocal * xxLocal + yyLocal * yyLocal)
    insideMask2d = rrLocal < (invResult.metrics["effectiveRadiusMm"] + 0.08)

    innerMeanMv = float(np.mean(scanData.netMvGrid[rrLocal < (invResult.metrics["effectiveRadiusMm"] - 0.50)]))
    measNonAxiPct = invMod.computeAzimuthalResidualPct(scanData.netMvGrid, rrLocal, innerMeanMv)
    simNonAxiPct = invMod.computeAzimuthalResidualPct(invResult.simFullNetMv, rrLocal, innerMeanMv)

    rBins = np.linspace(0.0, 9.2, 47)
    rCenters = 0.5 * (rBins[:-1] + rBins[1:])
    radialMeas = np.zeros_like(rCenters)
    radialSim = np.zeros_like(rCenters)
    radialIdeal = np.zeros_like(rCenters)
    for k in range(len(rCenters)):
        ring = (rrLocal >= rBins[k]) & (rrLocal < rBins[k + 1])
        if np.any(ring):
            radialMeas[k] = float(np.mean(scanData.netMvGrid[ring]))
            radialSim[k] = float(np.mean(invResult.simFullNetMv[ring]))
            radialIdeal[k] = float(np.mean(invResult.idealFullNetMv[ring]))

    measInterp = RegularGridInterpolator((scanYLocal, scanXLocal), scanData.netMvGrid, bounds_error=False, fill_value=np.nan)
    simInterp = RegularGridInterpolator((scanYLocal, scanXLocal), invResult.simFullNetMv, bounds_error=False, fill_value=np.nan)
    sliceXDense = np.linspace(-8.2, 8.2, 85)
    slicePts = np.stack([np.full_like(sliceXDense, sliceYLocal), sliceXDense], axis=-1)
    sliceMeasMv = measInterp(slicePts)
    sliceSimMv = simInterp(slicePts)

    axiZDense = zMm1d
    axiTotalDelta = np.mean(totalRadiusMm - (geometry.inputRadiusMm + geometry.coneSlope * zMm1d[None, :]), axis=0)

    metricsPath = outputsDir / "inversion-metrics.json"
    metricsPath.write_text(json.dumps(invResult.metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    svgPayload = {
        "thetaRad1d": thetaRad1d,
        "zMm1d": zMm1d,
        "nonAxiDeltaRMm": nonAxiDeltaRMm,
        "axiProfileZ": axiZDense,
        "axiProfileMm": axiTotalDelta,
        "scanXLocalMm": scanXLocal,
        "scanYLocalMm": scanYLocal,
        "measNetMv": scanData.netMvGrid,
        "simNetMv": invResult.simFullNetMv,
        "measNonAxiPct": measNonAxiPct,
        "simNonAxiPct": simNonAxiPct,
        "insideMask2d": insideMask2d,
        "radialR": rCenters,
        "radialIdealMv": radialIdeal,
        "radialMeasMv": radialMeas,
        "radialSimMv": radialSim,
        "sliceXMm": sliceXDense,
        "sliceMeasMv": sliceMeasMv,
        "sliceSimMv": sliceSimMv,
        "g4PointsXMm": g4X,
        "g4PointsMv": g4Mv,
        "g4PointsErrMv": g4ErrMv,
        "iterHistoryRmsePct": invResult.iterHistoryRmsePct,
        "trajectories": trajectories,
        "metrics": invResult.metrics,
    }

    svgZhPath = projectDir / "light-guide-imperfection-simulation.svg"
    svgEnPath = projectDir / "light-guide-imperfection-simulation-en.svg"
    svgMod.buildInteractiveSvg(outputSvgPath=svgZhPath, payload=svgPayload, lang="zh")
    svgMod.buildInteractiveSvg(outputSvgPath=svgEnPath, payload=svgPayload, lang="en")

    finalSnap = memoryGuard.inspectMemory("pipeline_complete")
    logger.info(
        "Pipeline completed successfully | Peak RSS: %.1f MB, MPS Driver: %.1f MB",
        finalSnap.rssMb,
        finalSnap.mpsDriverMb,
    )


if __name__ == "__main__":
    main()
