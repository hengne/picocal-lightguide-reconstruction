"""
Zoomable 1920x1080 widescreen research summary slide poster SVG generator.

Loads the reconstructed B-spline surface grid, 100x100 measured and simulated
laser scan matrices, Geant4 11.1.3 optical validation slice points, and
quantitative inversion metrics to render a vector research poster supporting
smooth wheel zoom and click-drag panning in both Chinese and English.

Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__author__ = "Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>"

SLIDE_WIDTH: int = 1920
SLIDE_HEIGHT: int = 1080


def divergingColorHex(normVal: float) -> str:
    """
    Map a normalized scalar in [-1, 1] onto a blue-white-crimson diverging palette.
    """
    t = max(-1.0, min(1.0, normVal))
    if t < 0.0:
        u = -t
        r = int(241 * (1.0 - u) + 37 * u)
        g = int(245 * (1.0 - u) + 99 * u)
        b = int(249 * (1.0 - u) + 235 * u)
    else:
        u = t
        r = int(241 * (1.0 - u) + 225 * u)
        g = int(245 * (1.0 - u) + 29 * u)
        b = int(249 * (1.0 - u) + 72 * u)
    return f"#{r:02x}{g:02x}{b:02x}"


def viridisLikeHex(normVal: float) -> str:
    """
    Map a normalized intensity in [0, 1] onto a high-contrast perceptual heatmap palette.
    """
    stops = [
        (0.00, (15, 23, 42)),
        (0.25, (30, 64, 175)),
        (0.50, (13, 148, 136)),
        (0.75, (132, 204, 22)),
        (1.00, (253, 224, 71)),
    ]
    t = max(0.0, min(1.0, normVal))
    for idx in range(len(stops) - 1):
        t0, c0 = stops[idx]
        t1, c1 = stops[idx + 1]
        if t <= t1:
            w = (t - t0) / max(1e-6, t1 - t0)
            r = int(c0[0] * (1.0 - w) + c1[0] * w)
            g = int(c0[1] * (1.0 - w) + c1[1] * w)
            b = int(c0[2] * (1.0 - w) + c1[2] * w)
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#fde047"


def getSlideTexts(
    lang: str,
    noisePct: float,
    nonAxiCorr: float,
    peakBulgeUm: float,
    peakDentUm: float,
    maxAbsUm: float,
) -> dict[str, Any]:
    """
    Build localized text strings for Chinese ('zh') or English ('en') summary posters.
    """
    if lang == "en":
        return {
            "fontStack": "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif",
            "title": "LHCb PicoCal Conical Light-Guide Inner-Surface Reconstruction &amp; Optical Simulation",
            "titleSize": "23.5",
            "subtitle": "Bicubic B-Spline Freeform Surface + Apple Metal GPU Ray Tracing + Geant4 11.1.3 Validation (Wheel Zoom &amp; Drag Pan)",
            "author": "Author: Hengne Li | Hengne.Li@cern.ch | Hengne.Li@m.scnu.edu.cn · SCNU / CERN LHCb",
            "kpi1": "Inner Relative RMSE",
            "kpi2": "Full-Map Pearson r",
            "kpi3": "Peak Memory Footprint",
            "p1Title": "1. Optical Physics Diagnosis &amp; Inverse Reconstruction Pipeline",
            "s1Head": "① Calibration &amp; CAD Baseline",
            "s1L1": "• STEP Cone: Ø17.5 → Ø8.0 mm, L=100 mm",
            "s1L2": "• Calibrated Aperture R_eff = 8.439 mm",
            "s1L3": "• Reflective Film ~311 μm, Beam σ=0.22 mm",
            "s2Head": "② Periodic Bicubic B-Spline",
            "s2L1": "• 36 Azimuthal × 16 Axial = 576 DOFs",
            "s2L2": "• Exact C² Analytic Normal n(θ, z)",
            "s2L3": "• Specular Law d' = d - 2(d·n)n",
            "s3Head": "③ Metal GPU + Geant4 Loop",
            "s3L1": "• M3 Max GPU ~300M Multi-Bounce Rays/s",
            "s3L2": "• SO(2) Equivariant Jacobian + Line Search",
            "s3L3": "• 5-Worker Geant4 11.1.3 Cross-Check",
            "findTitle": "Key Physical Mechanism: Why Micron-Scale Inner Bulges Induce Strong 2D Scan Ripples",
            "find1": "1. Etendue Limit: (4.0 / 8.75)² ≈ 20.9%; >99% of guided diffuse photons undergo 1–14 specular wall bounces;",
            "find2": "2. Critical-Angle Lever: ~78% of rays turn around; a 150–200 μm bulge steepens upstream slope &amp; triggers escape;",
            "find3": "3. Axial Projection: Front bulges modulate outer rings; mid-rear bulges (z ∈ [40, 85] mm) shape central pits &amp; crescents.",
            "find4": (
                f"Quantitative Match: Shot-noise floor {noisePct:.2f}%, non-axisymmetric texture r = {nonAxiCorr:.4f}, "
                f"peak bulge {peakBulgeUm:.0f} μm / dent +{peakDentUm:.0f} μm."
            ),
            "p2Title": "2. Reconstructed Surface Map ΔR_non-axi(θ, z)",
            "z0": "z = 0 mm (Entrance)",
            "z50": "Axial Depth z = 50 mm",
            "z100": "z = 100 mm (PMT Exit)",
            "cbLeft": f"Bulge -{maxAbsUm:.0f} μm",
            "cbRight": f"Dent +{maxAbsUm:.0f} μm",
            "p3Title": "3. Entrance 100×100 Diffuse Laser Scan Comparison (Net Signal mV)",
            "p3Left": "Measured Laser Scan (scan_summary.csv)",
            "p3Right": "Reconstructed Surface Sim (32,768 Rays/Pt)",
            "p3Foot": "Dashed Circle: Effective Aperture R_eff = 8.439 mm | Upper-left dark pit &amp; right bright crescent reproduced pixel-by-pixel",
            "p4Title": "4. Characteristic Cross-Section Slice &amp; Geant4 11.1.3 Optical Validation",
            "leg1": "Measured Laser Scan",
            "leg2X": 170,
            "leg2": "Metal GPU Bicubic Spline Ray Tracing",
            "leg3X": 438,
            "leg3": "Geant4 11.1.3 Validation (G4OpticalPhoton)",
            "xAxis": "Local Transverse Coordinate X' (mm) — Scroll wheel to inspect sub-mV details, double-click to reset",
            "hint": "Tip: Scroll mouse wheel to zoom (0.6× – 16×) around cursor; hold left button to pan",
        }
    return {
        "fontStack": "-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif",
        "title": "LHCb PicoCal 锥形光导内表面不均匀性逆向重构与光学模拟研究总结",
        "titleSize": "27.5",
        "subtitle": "双三次 B 样条自由曲面 + Apple M3 Max 原生 Metal GPU 镜面光线追踪 + Geant4 11.1.3 独立交叉验证（支持滚轮无级缩放）",
        "author": "作者：Hengne Li (李衡讷) | Hengne.Li@cern.ch | Hengne.Li@m.scnu.edu.cn · SCNU / CERN LHCb",
        "kpi1": "圆内相对拟合误差",
        "kpi2": "全图皮尔逊相关系数",
        "kpi3": "峰值内存安全占用",
        "p1Title": "1. 物理机制解析与逆向面形拟合技术路线",
        "s1Head": "① 实验测量与几何标定",
        "s1L1": "• STEP 标称锥口 Ø17.5 → Ø8.0 mm",
        "s1L2": "• 实测有效孔径 R_eff = 8.439 mm",
        "s1L3": "• 反射膜厚内缩约 311 μm，光斑 σ=0.22mm",
        "s2Head": "② 双三次 B 样条自由曲面",
        "s2L1": "• 周向 36 周期节 × 轴向 16 节 = 576 维",
        "s2L2": "• C² 连续精确解析法向量 n(θ, z)",
        "s2L3": "• 严格镜面反射 d' = d - 2(d·n)n",
        "s3Head": "③ Metal GPU + Geant4 闭环",
        "s3L1": "• M3 Max GPU ~3 亿条多跳光线/秒",
        "s3L2": "• SO(2) 等变雅可比 + 线搜索修正",
        "s3L3": "• 5 进程并行 Geant4 11.1.3 核验",
        "findTitle": "核心光学物理发现：为什么微米级内壁凸起会造成显著的二维光产额凹凸图？",
        "find1": "1. 光学扩展量极限：(4.0 / 8.75)² ≈ 20.9%，超过 99% 被收集漫散射光子需在锥壁经历 1 ~ 14 次镜面反射；",
        "find2": "2. 临界角放大效应：约 78% 光子处于调头临界区，仅 150~200 μm 局部内凸鼓包即可令迎光侧临界光线提前折返逃逸；",
        "find3": "3. 轴向投影映射：前段形变局域投射至外环；中后段形变 (z ∈ [40, 85] mm) 经多跳耦合直接调制圆心，复现左上暗坑与新月亮带。",
        "find4": f"定量对比：单点统计噪声极限 {noisePct:.2f}%，重构后凹凸纹理相关系数 r = {nonAxiCorr:.4f}，峰值内凸 {peakBulgeUm:.0f} μm / 外凹 +{peakDentUm:.0f} μm。",
        "p2Title": "2. 逆向重构出的内表面形变展开图 ΔR_non-axi(θ, z)",
        "z0": "z = 0 mm (入射口)",
        "z50": "轴向深度 z = 50 mm",
        "z100": "z = 100 mm (PMT 出口)",
        "cbLeft": f"内凸 -{maxAbsUm:.0f} μm",
        "cbRight": f"外凹 +{maxAbsUm:.0f} μm",
        "p3Title": "3. 入射圆口 100×100 二维漫散射扫描光响应对比（净信号 mV）",
        "p3Left": "实测扫描响应 (scan_summary.csv)",
        "p3Right": "重构内表面模拟响应 (32,768 光线/点)",
        "p3Foot": "白色虚线圆：标定有效通光孔径 R_eff = 8.439 mm | 中心偏左上暗坑与右侧高响应新月带在模拟图中得到像素级复现",
        "p4Title": "4. 特征截面定量吻合度（穿过凹坑区切片）与 Geant4 独立光学验证",
        "leg1": "实验激光扫描读数",
        "leg2X": 180,
        "leg2": "Metal GPU 双三次样条光线追踪",
        "leg3X": 445,
        "leg3": "Geant4 11.1.3 独立光学光子模拟 (G4OpticalPhoton)",
        "xAxis": "相对圆心局部横向坐标 X' (mm) — 滚轮可放大检视微伏级细节，双击画面复位",
        "hint": "提示：本 SVG 支持鼠标滚轮以光标为中心缩放 (0.6× ~ 16×) 与按住左键拖拽平移",
    }


def buildResearchSummarySlidesSvg(
    outputPath: Path,
    surfaceDf: pd.DataFrame,
    scanDf: pd.DataFrame,
    metrics: dict[str, Any],
    geant4Df: pd.DataFrame | None = None,
    lang: str = "zh",
) -> None:
    """
    Generate a 1920x1080 interactive zoomable research summary slide SVG.
    """
    logger = logging.getLogger(__name__)
    outputPath.parent.mkdir(parents=True, exist_ok=True)

    pivSurf = surfaceDf.pivot(index="theta_deg", columns="z_mm", values="delta_r_non_axi_um")
    surfArr = pivSurf.to_numpy(dtype=np.float32)
    stepTh = max(1, surfArr.shape[0] // 60)
    stepZ = max(1, surfArr.shape[1] // 67)
    surfSub = surfArr[::stepTh, ::stepZ]
    nThSub, nZSub = surfSub.shape
    maxAbsUm = max(120.0, float(np.percentile(np.abs(surfSub), 99.2)))

    surfCells: list[str] = []
    mapX0, mapY0, mapW, mapH = 1015.0, 225.0, 825.0, 265.0
    cellW, cellH = mapW / nZSub, mapH / nThSub
    for i in range(nThSub):
        for j in range(nZSub):
            valUm = float(surfSub[nThSub - 1 - i, j])
            col = divergingColorHex(valUm / maxAbsUm)
            surfCells.append(
                f'<rect x="{mapX0 + j * cellW:.2f}" y="{mapY0 + i * cellH:.2f}" '
                f'width="{cellW + 0.35:.2f}" height="{cellH + 0.35:.2f}" fill="{col}" />'
            )

    measGrid = np.zeros((100, 100), dtype=np.float32)
    simGrid = np.zeros((100, 100), dtype=np.float32)
    for row in scanDf.itertuples(index=False):
        ix, iy = int(row.idx_x), int(row.idx_y)
        measGrid[iy, ix] = float(row.probe_V) * 1000.0
        simGrid[iy, ix] = float(row.simulated_probe_V) * 1000.0

    bgMv = float(np.percentile(measGrid, 8.0))
    measNet = np.maximum(0.0, measGrid - bgMv)
    simNet = np.maximum(0.0, simGrid - bgMv)
    vLow, vHigh = 20.4, 27.5

    def renderScanHeatmap(arr100: np.ndarray, ox: float, oy: float, sizePx: float) -> str:
        sub = arr100[::2, ::2]
        nSide = sub.shape[0]
        pxStep = sizePx / nSide
        rects: list[str] = []
        for rIdx in range(nSide):
            yMm = -14.5 + (nSide - 1 - rIdx) * 0.4
            for cIdx in range(nSide):
                xMm = -7.5 + cIdx * 0.4
                rDist = float(np.hypot(xMm - 2.3665, yMm - (-4.3788)))
                vVal = float(sub[nSide - 1 - rIdx, cIdx])
                col = "#091022" if (rDist > 8.75 or vVal < 8.0) else viridisLikeHex((vVal - vLow) / (vHigh - vLow))
                rects.append(
                    f'<rect x="{ox + cIdx * pxStep:.2f}" y="{oy + rIdx * pxStep:.2f}" '
                    f'width="{pxStep + 0.35:.2f}" height="{pxStep + 0.35:.2f}" fill="{col}" />'
                )
        cx = ox + sizePx * ((2.3665 - (-7.5)) / 19.8)
        cy = oy + sizePx * (1.0 - ((-4.3788 - (-14.5)) / 19.8))
        radPx = sizePx * (8.44 / 19.8)
        rects.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radPx:.1f}" fill="none" '
            f'stroke="#f8fafc" stroke-width="1.4" stroke-dasharray="5,4" stroke-opacity="0.8" />'
        )
        return "".join(rects)

    measHeatmapSvg = renderScanHeatmap(measNet, 85.0, 635.0, 340.0)
    simHeatmapSvg = renderScanHeatmap(simNet, 495.0, 635.0, 340.0)

    sliceRow = 64
    xAxisVals = np.linspace(-7.5, 12.3, 100) - 2.3665
    measSlice = measNet[sliceRow, :]
    simSlice = simNet[sliceRow, :]
    plotX0, plotY0, plotW, plotH = 1020.0, 645.0, 815.0, 295.0

    def mapPlot(xLocal: float, mvVal: float) -> tuple[float, float]:
        px = plotX0 + ((xLocal + 9.0) / 18.0) * plotW
        py = plotY0 + plotH - (max(0.0, min(33.0, mvVal)) / 33.0) * plotH
        return px, py

    measPts = [f"{mapPlot(float(xv), float(mv))[0]:.1f},{mapPlot(float(xv), float(mv))[1]:.1f}" for xv, mv in zip(xAxisVals, measSlice) if -8.8 <= xv <= 8.8]
    simPts = [f"{mapPlot(float(xv), float(mv))[0]:.1f},{mapPlot(float(xv), float(mv))[1]:.1f}" for xv, mv in zip(xAxisVals, simSlice) if -8.8 <= xv <= 8.8]

    g4OverlaySvg: list[str] = []
    if geant4Df is not None and not geant4Df.empty and "g4_mv" in geant4Df.columns:
        for gRow in geant4Df.itertuples(index=False):
            gx, gy = mapPlot(float(gRow.x_local_mm), float(gRow.g4_mv))
            errPx = (float(getattr(gRow, "g4_err_mv", 0.45)) / 33.0) * plotH
            g4OverlaySvg.append(
                f'<line x1="{gx:.1f}" y1="{gy - errPx:.1f}" x2="{gx:.1f}" y2="{gy + errPx:.1f}" stroke="#34d399" stroke-width="1.8" />'
                f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="4.5" fill="#10b981" stroke="#ecfdf5" stroke-width="1.3" />'
            )

    rmseMv = float(metrics.get("innerRmseMv", 0.397))
    rmsePct = float(metrics.get("innerRmsePct", 1.56))
    noisePct = float(metrics.get("noiseFloorPct", 0.81))
    totalCorr = float(metrics.get("totalCorr", 0.9750))
    nonAxiCorr = float(metrics.get("nonAxiCorr", 0.8785))
    peakBulgeUm = float(metrics.get("peakBulgeUm", -187.0))
    peakDentUm = float(metrics.get("peakDentUm", 146.0))
    peakRssMb = float(metrics.get("peakRssMb", 332.0))

    tx = getSlideTexts(lang, noisePct, nonAxiCorr, peakBulgeUm, peakDentUm, maxAbsUm)
    l2x = int(tx["leg2X"])
    l3x = int(tx["leg3X"])

    svgContent = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SLIDE_WIDTH} {SLIDE_HEIGHT}" width="100%" height="100%" style="background:#020617;font-family:{tx['fontStack']};user-select:none;">
  <defs>
    <linearGradient id="headerAccent" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0ea5e9" /><stop offset="50%" stop-color="#6366f1" /><stop offset="100%" stop-color="#10b981" />
    </linearGradient>
    <linearGradient id="divScaleBar" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#2563eb" /><stop offset="50%" stop-color="#f1f5f9" /><stop offset="100%" stop-color="#e11d48" />
    </linearGradient>
  </defs>
  <g id="slideViewport">
    <rect x="40" y="28" width="1840" height="106" rx="16" fill="#0f172a" stroke="#1e293b" stroke-width="2" />
    <rect x="40" y="28" width="1840" height="5" rx="3" fill="url(#headerAccent)" />
    <text x="74" y="66" fill="#f8fafc" font-size="{tx['titleSize']}" font-weight="800">{tx['title']}</text>
    <text x="74" y="94" fill="#94a3b8" font-size="14">{tx['subtitle']}</text>
    <text x="74" y="119" fill="#38bdf8" font-size="13" font-weight="600">{tx['author']}</text>
    <g transform="translate(1265, 48)">
      <rect x="0" y="0" width="185" height="66" rx="10" fill="#020617" stroke="#0ea5e9" stroke-width="1.5" />
      <text x="92.5" y="26" fill="#7dd3fc" font-size="12.5" text-anchor="middle">{tx['kpi1']}</text>
      <text x="92.5" y="53" fill="#f8fafc" font-size="21" font-weight="800" text-anchor="middle">{rmsePct:.2f}% ({rmseMv:.2f} mV)</text>
      <rect x="198" y="0" width="185" height="66" rx="10" fill="#020617" stroke="#10b981" stroke-width="1.5" />
      <text x="290.5" y="26" fill="#6ee7b7" font-size="12.5" text-anchor="middle">{tx['kpi2']}</text>
      <text x="290.5" y="53" fill="#f8fafc" font-size="21" font-weight="800" text-anchor="middle">r = {totalCorr:.4f}</text>
      <rect x="396" y="0" width="185" height="66" rx="10" fill="#020617" stroke="#a855f7" stroke-width="1.5" />
      <text x="488.5" y="26" fill="#d8b4fe" font-size="12.5" text-anchor="middle">{tx['kpi3']}</text>
      <text x="488.5" y="53" fill="#f8fafc" font-size="21" font-weight="800" text-anchor="middle">{peakRssMb:.0f} MB</text>
    </g>
    <rect x="40" y="154" width="900" height="390" rx="16" fill="#0f172a" stroke="#1e293b" stroke-width="1.8" />
    <text x="72" y="194" fill="#f8fafc" font-size="20" font-weight="700">{tx['p1Title']}</text>
    <g transform="translate(72, 220)">
      <rect x="0" y="0" width="252" height="128" rx="12" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5" />
      <text x="16" y="30" fill="#38bdf8" font-size="15" font-weight="700">{tx['s1Head']}</text>
      <text x="16" y="56" fill="#cbd5e1" font-size="12.5">{tx['s1L1']}</text>
      <text x="16" y="80" fill="#cbd5e1" font-size="12.5">{tx['s1L2']}</text>
      <text x="16" y="104" fill="#cbd5e1" font-size="12.5">{tx['s1L3']}</text>
      <line x1="254" y1="64" x2="288" y2="64" stroke="#64748b" stroke-width="2.2" />
      <polygon points="288,64 278,58 278,70" fill="#64748b" />
      <rect x="292" y="0" width="258" height="128" rx="12" fill="#1e293b" stroke="#818cf8" stroke-width="1.5" />
      <text x="308" y="30" fill="#a5b4fc" font-size="15" font-weight="700">{tx['s2Head']}</text>
      <text x="308" y="56" fill="#cbd5e1" font-size="12.5">{tx['s2L1']}</text>
      <text x="308" y="80" fill="#cbd5e1" font-size="12.5">{tx['s2L2']}</text>
      <text x="308" y="104" fill="#cbd5e1" font-size="12.5">{tx['s2L3']}</text>
      <line x1="552" y1="64" x2="586" y2="64" stroke="#64748b" stroke-width="2.2" />
      <polygon points="586,64 576,58 576,70" fill="#64748b" />
      <rect x="590" y="0" width="246" height="128" rx="12" fill="#1e293b" stroke="#34d399" stroke-width="1.5" />
      <text x="606" y="30" fill="#6ee7b7" font-size="15" font-weight="700">{tx['s3Head']}</text>
      <text x="606" y="56" fill="#cbd5e1" font-size="12.5">{tx['s3L1']}</text>
      <text x="606" y="80" fill="#cbd5e1" font-size="12.5">{tx['s3L2']}</text>
      <text x="606" y="104" fill="#cbd5e1" font-size="12.5">{tx['s3L3']}</text>
    </g>
    <rect x="72" y="368" width="836" height="152" rx="12" fill="#020617" stroke="#334155" stroke-width="1.4" />
    <text x="90" y="396" fill="#fbbf24" font-size="14.2" font-weight="700">{tx['findTitle']}</text>
    <text x="90" y="424" fill="#e2e8f0" font-size="12.2">{tx['find1']}</text>
    <text x="90" y="451" fill="#e2e8f0" font-size="12.2">{tx['find2']}</text>
    <text x="90" y="478" fill="#e2e8f0" font-size="12.2">{tx['find3']}</text>
    <text x="90" y="505" fill="#94a3b8" font-size="12">{tx['find4']}</text>
    <rect x="960" y="154" width="920" height="390" rx="16" fill="#0f172a" stroke="#1e293b" stroke-width="1.8" />
    <text x="992" y="192" fill="#f8fafc" font-size="18.5" font-weight="700">{tx['p2Title']}</text>
    <g>{"".join(surfCells)}</g>
    <rect x="{mapX0:.1f}" y="{mapY0:.1f}" width="{mapW:.1f}" height="{mapH:.1f}" fill="none" stroke="#64748b" stroke-width="1.4" />
    <text x="{mapX0:.1f}" y="{mapY0 + mapH + 22:.1f}" fill="#94a3b8" font-size="12.5">{tx['z0']}</text>
    <text x="{mapX0 + mapW * 0.5:.1f}" y="{mapY0 + mapH + 22:.1f}" fill="#94a3b8" font-size="12.5" text-anchor="middle">{tx['z50']}</text>
    <text x="{mapX0 + mapW:.1f}" y="{mapY0 + mapH + 22:.1f}" fill="#94a3b8" font-size="12.5" text-anchor="end">{tx['z100']}</text>
    <text x="{mapX0 - 10:.1f}" y="{mapY0 + 12:.1f}" fill="#94a3b8" font-size="12" text-anchor="end">+180°</text>
    <text x="{mapX0 - 10:.1f}" y="{mapY0 + mapH * 0.5 + 4:.1f}" fill="#94a3b8" font-size="12" text-anchor="end">0°</text>
    <text x="{mapX0 - 10:.1f}" y="{mapY0 + mapH:.1f}" fill="#94a3b8" font-size="12" text-anchor="end">-180°</text>
    <rect x="1575" y="178" width="185" height="13" rx="4" fill="url(#divScaleBar)" stroke="#475569" stroke-width="1" />
    <text x="1567" y="189" fill="#60a5fa" font-size="11.5" text-anchor="end">{tx['cbLeft']}</text>
    <text x="1768" y="189" fill="#fb7185" font-size="11.5">{tx['cbRight']}</text>
    <rect x="40" y="564" width="900" height="484" rx="16" fill="#0f172a" stroke="#1e293b" stroke-width="1.8" />
    <text x="72" y="602" fill="#f8fafc" font-size="20" font-weight="700">{tx['p3Title']}</text>
    <text x="255" y="626" fill="#38bdf8" font-size="14" font-weight="700" text-anchor="middle">{tx['p3Left']}</text>
    <text x="665" y="626" fill="#34d399" font-size="14" font-weight="700" text-anchor="middle">{tx['p3Right']}</text>
    <g>{measHeatmapSvg}</g>
    <g>{simHeatmapSvg}</g>
    <rect x="85" y="635" width="340" height="340" fill="none" stroke="#475569" stroke-width="1.4" />
    <rect x="495" y="635" width="340" height="340" fill="none" stroke="#475569" stroke-width="1.4" />
    <text x="460" y="1012" fill="#cbd5e1" font-size="12.8" text-anchor="middle">{tx['p3Foot']}</text>
    <rect x="960" y="564" width="920" height="484" rx="16" fill="#0f172a" stroke="#1e293b" stroke-width="1.8" />
    <text x="992" y="602" fill="#f8fafc" font-size="20" font-weight="700">{tx['p4Title']}</text>
    <rect x="{plotX0:.1f}" y="{plotY0:.1f}" width="{plotW:.1f}" height="{plotH:.1f}" fill="#020617" stroke="#334155" stroke-width="1.4" />
    <line x1="{plotX0:.1f}" y1="{plotY0 + plotH * 0.5:.1f}" x2="{plotX0 + plotW:.1f}" y2="{plotY0 + plotH * 0.5:.1f}" stroke="#1e293b" stroke-width="1.2" stroke-dasharray="4,4" />
    <polyline points="{" ".join(measPts)}" fill="none" stroke="#94a3b8" stroke-width="2.2" stroke-opacity="0.85" />
    <polyline points="{" ".join(simPts)}" fill="none" stroke="#38bdf8" stroke-width="3.2" />
    <g>{"".join(g4OverlaySvg)}</g>
    <g transform="translate(1035, 662)">
      <line x1="0" y1="8" x2="28" y2="8" stroke="#94a3b8" stroke-width="2.4" />
      <text x="36" y="13" fill="#cbd5e1" font-size="12.5">{tx['leg1']}</text>
      <line x1="{l2x}" y1="8" x2="{l2x + 28}" y2="8" stroke="#38bdf8" stroke-width="3.2" />
      <text x="{l2x + 36}" y="13" fill="#38bdf8" font-size="12.5">{tx['leg2']}</text>
      <circle cx="{l3x}" cy="8" r="4.5" fill="#10b981" stroke="#ecfdf5" stroke-width="1.2" />
      <text x="{l3x + 12}" y="13" fill="#6ee7b7" font-size="12.5">{tx['leg3']}</text>
    </g>
    <text x="{plotX0 + plotW * 0.5:.1f}" y="{plotY0 + plotH + 28:.1f}" fill="#94a3b8" font-size="13" text-anchor="middle">{tx['xAxis']}</text>
    <text x="1420" y="1018" fill="#64748b" font-size="12.5" text-anchor="middle">{tx['hint']}</text>
  </g>
  <script type="application/ecmascript"><![CDATA[
    (function() {{
      const svgEl = document.documentElement;
      const vp = document.getElementById("slideViewport");
      let scale = 1.0, tx = 0.0, ty = 0.0, dragging = false, startX = 0, startY = 0;
      function applyTransform() {{
        vp.setAttribute("transform", "translate(" + tx.toFixed(2) + "," + ty.toFixed(2) + ") scale(" + scale.toFixed(4) + ")");
      }}
      svgEl.addEventListener("wheel", function(ev) {{
        ev.preventDefault();
        const pt = svgEl.createSVGPoint();
        pt.x = ev.clientX; pt.y = ev.clientY;
        const ctm = svgEl.getScreenCTM();
        if (!ctm) return;
        const cursor = pt.matrixTransform(ctm.inverse());
        const factor = ev.deltaY < 0 ? 1.14 : 0.88;
        const nextScale = Math.max(0.6, Math.min(16.0, scale * factor));
        tx = cursor.x - (cursor.x - tx) * (nextScale / scale);
        ty = cursor.y - (cursor.y - ty) * (nextScale / scale);
        scale = nextScale;
        applyTransform();
      }}, {{ passive: false }});
      svgEl.addEventListener("mousedown", function(ev) {{ dragging = true; startX = ev.clientX; startY = ev.clientY; }});
      window.addEventListener("mousemove", function(ev) {{
        if (!dragging) return;
        const ctm = svgEl.getScreenCTM();
        if (!ctm) return;
        tx += (ev.clientX - startX) / ctm.a;
        ty += (ev.clientY - startY) / ctm.d;
        startX = ev.clientX; startY = ev.clientY;
        applyTransform();
      }});
      window.addEventListener("mouseup", function() {{ dragging = false; }});
      svgEl.addEventListener("dblclick", function() {{ scale = 1.0; tx = 0.0; ty = 0.0; applyTransform(); }});
    }})();
  ]]></script>
</svg>
"""
    outputPath.write_text(svgContent, encoding="utf-8")
    logger.info("Generated research summary slide SVG (%s): %s (%.1f KB)", lang, outputPath, outputPath.stat().st_size / 1024.0)
