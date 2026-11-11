"""
End-to-end orchestrator for the bilingual presentation, interactive 3D, and webpage suite.

Loads the reconstructed B-spline surface grid, 100x100 measured and simulated
scan matrices, Geant4 11.1.3 optical validation data, and inversion metrics from
Agy_Carbon/outputs/, then builds both Simplified Chinese ('zh') and English ('en')
editions of all visual and interactive deliverables:
  1. Animated photon propagation SVGs (photon-propagation-animated[-en].svg)
  2. 1920x1080 zoomable research summary slide posters (research-summary-slides[-en].svg)
  3. 3D interactive photon-to-PMT HTML5 simulators (interactive-3d-photon-simulator[-en].html)
  4. Project summary portals (index.html, index-en.html)
  5. Zero-dependency single-file standalone HTML bundles (*-standalone-zh.html, *-standalone-en.html)

Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

__author__ = "Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>"


def loadPresentationModule(moduleAlias: str, filePath: Path) -> Any:
    """
    Dynamically load a kebab-case local builder module from its filesystem path.
    """
    spec = importlib.util.spec_from_file_location(moduleAlias, filePath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load presentation module from path: {filePath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[moduleAlias] = module
    spec.loader.exec_module(module)
    return module


def buildFullPresentationSuite(projectDir: Path) -> dict[str, Path]:
    """
    Execute the complete bilingual presentation, portal, and standalone bundle build.
    """
    logger = logging.getLogger("agy.presentation")
    outputsDir = projectDir / "outputs"
    presDir = projectDir / "presentation"
    presDir.mkdir(parents=True, exist_ok=True)

    surfaceCsv = outputsDir / "reconstructed-surface-grid.csv"
    scanCsv = outputsDir / "simulated-scan-summary.csv"
    metricsJson = outputsDir / "inversion-metrics.json"
    g4Csv = outputsDir / "geant4-slice-validation.csv"

    if not (surfaceCsv.exists() and scanCsv.exists() and metricsJson.exists()):
        raise FileNotFoundError(
            f"Missing required pipeline outputs in {outputsDir}; run run-simulation-pipeline.py first."
        )

    surfaceDf = pd.read_csv(surfaceCsv)
    scanDf = pd.read_csv(scanCsv)
    metrics: dict[str, Any] = json.loads(metricsJson.read_text(encoding="utf-8"))
    geant4Df: pd.DataFrame | None = pd.read_csv(g4Csv) if g4Csv.exists() else None

    animMod = loadPresentationModule("pres_anim_svg", projectDir / "presentation-animated-svg.py")
    slidesMod = loadPresentationModule("pres_slides_svg", projectDir / "presentation-slides-svg.py")
    htmlMod = loadPresentationModule("pres_3d_html", projectDir / "presentation-3d-html.py")
    webMod = loadPresentationModule("pres_summary_web", projectDir / "build-summary-webpage.py")
    standaloneMod = loadPresentationModule("pres_standalone_html", projectDir / "build-standalone-report.py")

    animZh = presDir / "photon-propagation-animated.svg"
    animEn = presDir / "photon-propagation-animated-en.svg"
    animMod.buildAnimatedPropagationSvg(outputPath=animZh, metrics=metrics, lang="zh")
    animMod.buildAnimatedPropagationSvg(outputPath=animEn, metrics=metrics, lang="en")

    slidesZh = presDir / "research-summary-slides.svg"
    slidesEn = presDir / "research-summary-slides-en.svg"
    slidesMod.buildResearchSummarySlidesSvg(
        outputPath=slidesZh, surfaceDf=surfaceDf, scanDf=scanDf, metrics=metrics, geant4Df=geant4Df, lang="zh"
    )
    slidesMod.buildResearchSummarySlidesSvg(
        outputPath=slidesEn, surfaceDf=surfaceDf, scanDf=scanDf, metrics=metrics, geant4Df=geant4Df, lang="en"
    )

    sim3dZh = presDir / "interactive-3d-photon-simulator.html"
    sim3dEn = presDir / "interactive-3d-photon-simulator-en.html"
    htmlMod.buildInteractive3dSimulatorHtml(
        outputPath=sim3dZh, surfaceDf=surfaceDf, scanDf=scanDf, metrics=metrics, lang="zh"
    )
    htmlMod.buildInteractive3dSimulatorHtml(
        outputPath=sim3dEn, surfaceDf=surfaceDf, scanDf=scanDf, metrics=metrics, lang="en"
    )

    indexZh, indexEn = webMod.buildProjectSummaryWebpage(projectDir=projectDir)
    standaloneZh, standaloneEn = standaloneMod.buildStandaloneReportHtml(projectDir=projectDir)

    logger.info("Completed full bilingual presentation and standalone suite in %s", projectDir)
    return {
        "animZh": animZh,
        "animEn": animEn,
        "slidesZh": slidesZh,
        "slidesEn": slidesEn,
        "sim3dZh": sim3dZh,
        "sim3dEn": sim3dEn,
        "indexZh": indexZh,
        "indexEn": indexEn,
        "standaloneZh": standaloneZh,
        "standaloneEn": standaloneEn,
    }


def main() -> None:
    """
    CLI entry point building all bilingual presentation SVGs, 3D simulators, portals, and standalone HTMLs.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    buildFullPresentationSuite(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
