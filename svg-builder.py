"""
Zoomable Interactive Vector Report Builder (SVG) for LHCb PicoCal Light-Guide Reconstruction.

Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
Affiliation: South China Normal University (SCNU) / LHCb Collaboration, CERN

Packages the reconstructed 3D imperfect inner-surface deformation map, multi-bounce
3D specular ray trajectories, 100x100 XY scan light-yield comparison grids,
Marching Squares micrometer contour lines, and Geant4 11.1.3 optical-photon
cross-validation curves into a zero-dependency interactive SVG supporting smooth
cursor-centered wheel zoom, drag panning, layer toggles, and live hover readouts
in both Simplified Chinese (lang="zh") and English (lang="en").
"""

from __future__ import annotations

import html
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

__author__ = "Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>"

CANVAS_WIDTH: int = 2240
CANVAS_HEIGHT: int = 1480


@dataclass(frozen=True)
class DetectedFeature:
    """
    Dominant local bulge or dent detected on the unwrapped inner surface.
    """

    featureId: int
    kindLabel: str
    thetaDeg: float
    zMm: float
    amplitudeUm: float


def colormapTurboHex(normVal: float) -> str:
    """
    Map a normalized [0, 1] scalar onto a high-contrast perceptual Turbo-like palette.
    """
    x = max(0.0, min(1.0, float(normVal)))
    stops = [
        (0.00, (48, 18, 59)),
        (0.18, (65, 98, 216)),
        (0.36, (36, 188, 213)),
        (0.52, (112, 235, 130)),
        (0.68, (226, 220, 56)),
        (0.84, (251, 128, 34)),
        (1.00, (165, 18, 18)),
    ]
    for idx in range(len(stops) - 1):
        t0, c0 = stops[idx]
        t1, c1 = stops[idx + 1]
        if x <= t1 or idx == len(stops) - 2:
            w = (x - t0) / max(1e-9, t1 - t0)
            r = int(round(c0[0] + w * (c1[0] - c0[0])))
            g = int(round(c0[1] + w * (c1[1] - c0[1])))
            b = int(round(c0[2] + w * (c1[2] - c0[2])))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#a51212"


def colormapDivergingHex(normSigned: float) -> str:
    """
    Map a symmetric [-1, 1] residual onto a blue-white-red diverging palette.
    Negative (blue) denotes an inward bulge or deficit; positive (red) denotes an outward dent or surplus.
    """
    v = max(-1.0, min(1.0, float(normSigned)))
    stops = [
        (-1.0, (22, 62, 140)),
        (-0.5, (72, 138, 210)),
        (0.0, (246, 247, 250)),
        (0.5, (232, 118, 84)),
        (1.0, (162, 22, 34)),
    ]
    for idx in range(len(stops) - 1):
        t0, c0 = stops[idx]
        t1, c1 = stops[idx + 1]
        if v <= t1 or idx == len(stops) - 2:
            w = (v - t0) / max(1e-9, t1 - t0)
            r = int(round(c0[0] + w * (c1[0] - c0[0])))
            g = int(round(c0[1] + w * (c1[1] - c0[1])))
            b = int(round(c0[2] + w * (c1[2] - c0[2])))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#a21622"


def extractMarchingSquaresPaths(
    field2d: np.ndarray,
    xCoords: np.ndarray,
    yCoords: np.ndarray,
    level: float,
    xToPx: Any,
    yToPx: Any,
) -> str:
    """
    Extract exact iso-contour line segments from a 2D scalar grid via Marching Squares
    and serialize them as a compact SVG path data string.
    """
    ny, nx = field2d.shape
    segments: list[str] = []

    def interp(v1: float, v2: float, p1: float, p2: float) -> float:
        denom = v2 - v1
        if abs(denom) < 1e-12:
            return 0.5 * (p1 + p2)
        t = max(0.0, min(1.0, (level - v1) / denom))
        return p1 + t * (p2 - p1)

    for iy in range(ny - 1):
        y0, y1 = float(yCoords[iy]), float(yCoords[iy + 1])
        for ix in range(nx - 1):
            x0, x1 = float(xCoords[ix]), float(xCoords[ix + 1])
            v00 = float(field2d[iy, ix])
            v10 = float(field2d[iy, ix + 1])
            v11 = float(field2d[iy + 1, ix + 1])
            v01 = float(field2d[iy + 1, ix])
            if not (math.isfinite(v00) and math.isfinite(v10) and math.isfinite(v11) and math.isfinite(v01)):
                continue

            mask = 0
            if v00 >= level:
                mask |= 1
            if v10 >= level:
                mask |= 2
            if v11 >= level:
                mask |= 4
            if v01 >= level:
                mask |= 8
            if mask == 0 or mask == 15:
                continue

            eBottom = (xToPx(interp(v00, v10, x0, x1)), yToPx(y0))
            eRight = (xToPx(x1), yToPx(interp(v10, v11, y0, y1)))
            eTop = (xToPx(interp(v01, v11, x0, x1)), yToPx(y1))
            eLeft = (xToPx(x0), yToPx(interp(v00, v01, y0, y1)))

            table: dict[int, list[tuple[tuple[float, float], tuple[float, float]]]] = {
                1: [(eLeft, eBottom)],
                2: [(eBottom, eRight)],
                3: [(eLeft, eRight)],
                4: [(eRight, eTop)],
                5: [(eLeft, eTop), (eBottom, eRight)],
                6: [(eBottom, eTop)],
                7: [(eLeft, eTop)],
                8: [(eTop, eLeft)],
                9: [(eTop, eBottom)],
                10: [(eBottom, eLeft), (eRight, eTop)],
                11: [(eTop, eRight)],
                12: [(eRight, eLeft)],
                13: [(eBottom, eRight)],
                14: [(eLeft, eBottom)],
            }
            for pStart, pEnd in table.get(mask, []):
                segments.append(f"M{pStart[0]:.1f},{pStart[1]:.1f}L{pEnd[0]:.1f},{pEnd[1]:.1f}")

    return "".join(segments)


def detectDominantSurfaceShapes(
    thetaDeg1d: np.ndarray,
    zMm1d: np.ndarray,
    deltaRUm: np.ndarray,
    maxFeatures: int = 6,
    lang: str = "zh",
) -> list[DetectedFeature]:
    """
    Identify well-separated local extrema (major inward bulges and outward dents)
    on the unwrapped non-axisymmetric deformation map for visual callout badges.
    """
    isEn = lang.lower().startswith("en")
    neighborhood = (21, 21)
    localMax = maximum_filter(deltaRUm, size=neighborhood, mode=("wrap", "nearest"))
    localMin = minimum_filter(deltaRUm, size=neighborhood, mode=("wrap", "nearest"))

    candidates: list[tuple[float, int, int]] = []
    for i in range(deltaRUm.shape[0]):
        for j in range(4, deltaRUm.shape[1] - 4):
            val = float(deltaRUm[i, j])
            if abs(val) < 35.0:
                continue
            if val == float(localMax[i, j]) or val == float(localMin[i, j]):
                candidates.append((abs(val), i, j))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[DetectedFeature] = []
    for _, iIdx, jIdx in candidates:
        thVal = float(thetaDeg1d[iIdx])
        zVal = float(zMm1d[jIdx])
        # Suppress spatially adjacent duplicate extrema so each badge marks a distinct feature
        tooClose = False
        for prev in selected:
            dTh = abs(((thVal - prev.thetaDeg + 180.0) % 360.0) - 180.0)
            dZ = abs(zVal - prev.zMm)
            if dTh < 38.0 and dZ < 16.0:
                tooClose = True
                break
        if tooClose:
            continue

        amp = float(deltaRUm[iIdx, jIdx])
        if isEn:
            label = "Inward Bulge" if amp < 0.0 else "Outward Dent"
        else:
            label = "内凸鼓包" if amp < 0.0 else "外扩凹陷"
        selected.append(
            DetectedFeature(
                featureId=len(selected) + 1,
                kindLabel=label,
                thetaDeg=thVal,
                zMm=zVal,
                amplitudeUm=amp,
            )
        )
        if len(selected) >= maxFeatures:
            break

    return selected


def projectPoint3d(
    xMm: float,
    yMm: float,
    zMm: float,
    centerPx: tuple[float, float],
    scalePx: float = 5.4,
) -> tuple[float, float, float]:
    """
    Project 3D light-guide coordinates (x, y, z) onto the 2D panel plane and return depth
    for back-to-front painter's algorithm sorting.
    """
    # Center z at 50 mm and foreshorten axial length so both circular apertures remain legible
    zCentered = (zMm - 50.0) * 0.62
    yaw = math.radians(-34.0)
    pitch = math.radians(21.0)

    u3 = zCentered * math.cos(yaw) + xMm * math.sin(yaw)
    v3 = -zCentered * math.sin(yaw) + xMm * math.cos(yaw)
    w3 = yMm

    screenX = centerPx[0] + scalePx * u3
    screenY = centerPx[1] - scalePx * (w3 * math.cos(pitch) - v3 * math.sin(pitch))
    depth = v3 * math.cos(pitch) + w3 * math.sin(pitch)
    return screenX, screenY, depth


def _getLocalizedStrings(lang: str, metrics: dict[str, float]) -> dict[str, Any]:
    """
    Return the complete dictionary of UI, axis, legend, and summary strings for the requested language.
    """
    if lang.lower().startswith("en"):
        return {
            "bannerTitle": "LHCb Upgrade PicoCal Conical Light-Guide Surface Reconstruction & Specular Ray-Tracing Report",
            "authorTag": "Author: Hengne Li | Hengne.Li@cern.ch",
            "bannerSub": (
                f"CAD Baseline: Entry Ø17.50 mm (R=8.75 mm) → Exit Ø8.00 mm (R=4.00 mm), Length L=100.0 mm | "
                f"Inner RMSE = {metrics['innerRmsePct']:.2f}% (Poisson floor ≈ {metrics['noiseFloorPct']:.2f}%), "
                f"Full-map r = {metrics['totalCorr']:.4f}, Non-axi texture r_bump = {metrics['nonAxiCorr']:.4f}"
            ),
            "paTitle": "A. Reconstructed Inner-Surface Deformation Map ΔR_non-axi(θ, z) [μm]",
            "paSub": "Blue (ΔR &lt; 0): inward reflective-foil bulge toward axis; Red (ΔR &gt; 0): outward dent; right inset shows axisymmetric ΔR_axi(z)",
            "paAxisX": "Axial Depth z (mm) [z=0 Entry Aperture → z=100 PMT Exit Window]",
            "paAxisY": "Azimuthal Angle θ (deg)",
            "paAxiTitle": "Axisymmetric Profile ΔR_axi(z)",
            "pbTitle": "B. 3D Deformed Light-Guide Surface (8× Exaggerated) & Multi-Bounce Specular Trajectories",
            "pbSub": "Teal rays: collected at z=100 mm PMT window after specular bounces; Coral dashed rays: supercritical turnaround loss",
            "ring0": "Diffuse Scan Entry Window (z=0, Ø17.5mm CAD / Ø16.9mm Clear Foil)",
            "ring100": "Exit Aperture → PMT Readout Window (z=100mm, Ø8.0mm CAD)",
            "ring110": "STEP PMT Mounting Counterbore (z=110mm, Ø16.36mm)",
            "pbLeg1": "Collected rays (transmitted through z=100 mm PMT window via specular bounces)",
            "pbLeg2": "Turnaround loss rays (polar angle exceeds 90° due to cone taper or inward bulge)",
            "pcTitle": "C. 2D XY Entry Scan Measured vs. Reconstructed Simulation Comparison (100 × 100 Full Grid)",
            "pcSub": "Identical diffuse light source scanned across (X, Y) at z=0 with specular inner-wall reflections (hover for exact coordinates & mV)",
            "c1Title": "C1. Measured Net Signal V_meas (mV)",
            "c2Title": "C2. Reconstructed Sim V_sim (mV)",
            "c3Title": "C3. Measured Non-Axi Residual (%)",
            "c4Title": "C4. Simulated Non-Axi Residual (%)",
            "pcNoteTitle": "Physical Mechanism & Inverse Reconstruction Summary:",
            "pcNotes": [
                f"1. Effective Clear Aperture & Roll-off: Measured R_eff ≈ 8.45 mm (~0.30 mm inward offset vs. 8.75 mm CAD from foil thickness), laser spot σ_beam ≈ {metrics['beamSigmaMm']:.2f} mm.",
                "2. Upper-Left Central Valley (X'≈-0.8 mm, Y'≈+1.8 mm, -5.5% dip): Caused by mid-to-rear (z ≈ 35–75 mm) asymmetric inward bulges steepening near-critical grazing bounces.",
                "3. Upper Crescent Ridge (+3.5% to +5.5%) & Bottom Notch (-6.0%): Driven by front-to-mid (z ≈ 8–32 mm) hundred-micron foil bulges/dents modulating local slope angles.",
                f"4. Fit Residual vs. Shot Noise: Full-aperture RMSE = {metrics['innerRmseMv']:.3f} mV ({metrics['innerRmsePct']:.2f}%), approaching the n_events=1000 statistical floor ({metrics['noiseFloorPct']:.2f}%).",
            ],
            "pdTitle": "D. Radial Uniformity Profile, Cross-Section Slice & Geant4 11.1.3 Optical Validation",
            "d1Title": "D1. Azimuthal-Mean Radial Response V(r): Ideal STEP Cone vs. Measured vs. Reconstructed",
            "d1AxisX": "Radial Distance from Center r (mm)",
            "d1AxisY": "Net Signal (mV)",
            "d1LegIdeal": "Ideal STEP Cone Baseline",
            "d1LegMeas": "Measured Scan (scan_summary.csv)",
            "d1LegSim": "Reconstructed Deformed Surface Sim",
            "d2Title": "D2. Cross-Section Slice (Y'=+1.6 mm) & Geant4 11.1.3 Optical Check",
            "d2AxisX": "Local Slice Coordinate X' (mm)",
            "d2LegG4": "Geant4 11.1.3 (G4TessellatedSolid + G4OpticalPhysics)",
            "d3Title": "D3. Inversion Convergence",
            "d3Noise": f"Noise Floor ({metrics['noiseFloorPct']:.2f}%)",
            "d3AxisX": "Optimization Step (RMSE %)",
            "btnZoomIn": "＋ Zoom In",
            "btnZoomOut": "－ Zoom Out",
            "btnReset": "⟲ 1:1 Reset",
            "btnContourOn": "✓ Contours: ON",
            "btnContourOff": "○ Contours: OFF",
            "btnAnnotOn": "✓ Callouts: ON",
            "btnAnnotOff": "○ Callouts: OFF",
            "btnRaysOn": "✓ 3D Rays: ON",
            "btnRaysOff": "○ 3D Rays: OFF",
            "hudDefault": "Controls: Cursor-centered wheel zoom (0.6×–24×), left-drag pan, double-click reset, hover Panels A/C for live (z, θ, ΔR) or (X', Y', mV)",
            "jsBulge": "Inward Bulge",
            "jsDent": "Outward Dent",
            "jsSurfFmt": "[Zoom {scale}×] Inner Surface Probe: z = {z} mm, θ = {th}°, Non-axi ΔR = {dr} μm ({kind})",
            "jsScanInFmt": "[Zoom {scale}×] Entry Scan Point: X'={x} mm, Y'={y} mm | Meas={vm} mV, Sim={vs} mV (Diff {diff}%)",
            "jsScanOutFmt": "[Zoom {scale}×] Outside Clear Aperture: X'={x} mm, Y'={y} mm (Background Region)",
        }

    return {
        "bannerTitle": "LHCb Upgrade PicoCal 圆锥光导内表面不均匀性逆向重构与镜面反射光学模拟报告",
        "authorTag": "作者: 李衡讷 (Hengne Li) | Hengne.Li@cern.ch",
        "bannerSub": (
            f"STEP 基准几何: 入射圆口 Ø17.50 mm (R=8.75 mm) → 出射圆口 Ø8.00 mm (R=4.00 mm), 轴向长 L=100.0 mm | "
            f"圆内拟合 RMSE = {metrics['innerRmsePct']:.2f}% (单点统计噪声极限 ≈ {metrics['noiseFloorPct']:.2f}%), "
            f"全图相关系数 r = {metrics['totalCorr']:.4f}, 非对称凹凸纹理相关系数 r_bump = {metrics['nonAxiCorr']:.4f}"
        ),
        "paTitle": "A. 反演重构的光导内表面几何形变展开图 ΔR_non-axi(θ, z) [单位: μm]",
        "paSub": "蓝色 (ΔR &lt; 0) 表示内壁向光导中心凸起/鼓包，红色 (ΔR &gt; 0) 表示内壁向外凹陷；右侧附轴对称贴膜厚度与母线修正 ΔR_axi(z)",
        "paAxisX": "光导轴向深度 z (mm) [z=0 入射大口 → z=100 PMT 出射小口]",
        "paAxisY": "内壁周向方位角 θ (deg)",
        "paAxiTitle": "轴对称截面修正 ΔR_axi(z)",
        "pbTitle": "B. 不完美光导三维几何曲面 (形变 8× 视觉放大) 与镜面反射三维光线追踪轨迹",
        "pbSub": "青绿色光线：经内壁完美镜面反射后进入 z=100 mm PMT 窗口；橙红色光线：撞击内凸陡坡后超临界折返逃逸",
        "ring0": "入射漫散射扫描窗 (z=0, Ø17.5mm CAD / Ø16.9mm 贴膜通光孔)",
        "ring100": "出射孔 → PMT 读出窗 (z=100mm, Ø8.0mm CAD)",
        "ring110": "STEP PMT 安装沉孔 (z=110mm, Ø16.36mm)",
        "pbLeg1": "有效收集光线（经多次完美镜面反射穿过 PMT 窗口）",
        "pbLeg2": "损失光线（受锥角收缩或局部内凸鼓包调制，极角超90°折返输入端）",
        "pcTitle": "C. 入射口二维 XY 扫描光产额实测数据与重构变形内表面模拟结果对比 (100 × 100 全分辨率网格)",
        "pcSub": "在圆口内每一点 (X, Y) 放置同强度漫散射光源，经重构的不完美内表面纯镜面反射后到达 PMT 的二维读出对比（悬停可查看精确坐标与读数）",
        "c1Title": "C1. 实测净光产额 V_meas (mV)",
        "c2Title": "C2. 变形内表面模拟 V_sim (mV)",
        "c3Title": "C3. 实测非轴对称凹凸起伏 (%)",
        "c4Title": "C4. 模拟非轴对称凹凸起伏 (%)",
        "pcNoteTitle": "物理机制与逆向反演结论摘要（对应上方内表面各形状区域）：",
        "pcNotes": [
            f"1. 整体通光孔径与径向滚降：实测有效通光半径 R_eff ≈ 8.45 mm（较 STEP 标称 8.75 mm 内缩约 0.30 mm，与内壁贴附反射膜厚度吻合），激光漫散射光斑 σ_beam ≈ {metrics['beamSigmaMm']:.2f} mm。",
            "2. 中心偏左上低谷区 (X'≈-0.8mm, Y'≈+1.8mm, 幅度约 -5.5%)：由光导中后段 (z ≈ 35~75 mm) 的非对称凸起调制临界角光线的末次转向角所致。",
            "3. 上半圈及右下月牙形高响应带 (+3.5%~+5.5%) 与下方低响应缺口 (-6.0%)：主要由前中段 (z ≈ 8~32 mm) 几百微米级反射膜贴合鼓包/凹陷的迎光陡坡与背光缓坡效应引起。",
            f"4. 拟合残差评估：模拟与实测全圆内均方根误差 RMSE = {metrics['innerRmseMv']:.3f} mV ({metrics['innerRmsePct']:.2f}%)，逼近 n_events=1000 泊松统计涨落本底 ({metrics['noiseFloorPct']:.2f}%)。",
        ],
        "pdTitle": "D. 径向均匀性曲线、截面切片与本机 Geant4 11.1.3 光学光子独立验证",
        "d1Title": "D1. 方位角平均径向光响应 V(r) 对比：完美 STEP 圆锥 vs 实测 vs 变形曲面模拟",
        "d1AxisX": "距中心半径 r (mm)",
        "d1AxisY": "净信号 (mV)",
        "d1LegIdeal": "完美 STEP 圆锥基准",
        "d1LegMeas": "实测数据 (scan_summary.csv)",
        "d1LegSim": "重构不完美曲面镜面反射模拟",
        "d2Title": "D2. 横截面光产额起伏切片 (Y'=+1.6mm) 与 Geant4 11.1.3 校验",
        "d2AxisX": "截面局部坐标 X' (mm)",
        "d2LegG4": "Geant4 11.1.3 (G4TessellatedSolid + G4OpticalPhysics) 验证点",
        "d3Title": "D3. 面形迭代拟合收敛曲线",
        "d3Noise": f"统计噪声极限 ({metrics['noiseFloorPct']:.2f}%)",
        "d3AxisX": "反演迭代步 (RMSE %)",
        "btnZoomIn": "＋ 放大视图",
        "btnZoomOut": "－ 缩小视图",
        "btnReset": "⟲ 1:1 复位画布",
        "btnContourOn": "✓ 等高线: 开启",
        "btnContourOff": "○ 等高线: 隐藏",
        "btnAnnotOn": "✓ 形状标注: 开启",
        "btnAnnotOff": "○ 形状标注: 隐藏",
        "btnRaysOn": "✓ 3D反射光路: 开启",
        "btnRaysOff": "○ 3D反射光路: 隐藏",
        "hudDefault": "交互提示：支持鼠标滚轮以光标为中心无级缩放 (0.6× ~ 24×)、按住左键拖拽平移、移动鼠标至图面实时读取 (z, θ, ΔR) 或 (X, Y, V)",
        "jsBulge": "内凸鼓包",
        "jsDent": "外扩凹陷",
        "jsSurfFmt": "[缩放 {scale}×] 内表面探针: z = {z} mm, θ = {th}°, 非对称形变 ΔR = {dr} μm ({kind})",
        "jsScanInFmt": "[缩放 {scale}×] 入射口扫描点: X'={x} mm, Y'={y} mm | 实测={vm} mV, 模拟={vs} mV (偏差 {diff}%)",
        "jsScanOutFmt": "[缩放 {scale}×] 入射口外侧端面: X'={x} mm, Y'={y} mm (通光孔外本底区)",
    }


def buildInteractiveSvg(
    outputSvgPath: Path,
    payload: dict[str, Any],
    lang: str = "zh",
) -> None:
    """
    Assemble the multi-panel zoomable interactive SVG report in the requested language
    ("zh" or "en") and write it to disk.
    """
    logger = logging.getLogger(__name__)

    thetaRad1d: np.ndarray = payload["thetaRad1d"]
    thetaDeg1d = np.degrees(thetaRad1d)
    zMm1d: np.ndarray = payload["zMm1d"]
    nonAxiDeltaRMm: np.ndarray = payload["nonAxiDeltaRMm"]
    axiProfileZ: np.ndarray = payload["axiProfileZ"]
    axiProfileMm: np.ndarray = payload["axiProfileMm"]

    scanXLocalMm: np.ndarray = payload["scanXLocalMm"]
    scanYLocalMm: np.ndarray = payload["scanYLocalMm"]
    measNetMv: np.ndarray = payload["measNetMv"]
    simNetMv: np.ndarray = payload["simNetMv"]
    measNonAxiPct: np.ndarray = payload["measNonAxiPct"]
    simNonAxiPct: np.ndarray = payload["simNonAxiPct"]
    insideMask2d: np.ndarray = payload["insideMask2d"]

    radialR: np.ndarray = payload["radialR"]
    radialIdealMv: np.ndarray = payload["radialIdealMv"]
    radialMeasMv: np.ndarray = payload["radialMeasMv"]
    radialSimMv: np.ndarray = payload["radialSimMv"]

    sliceXMm: np.ndarray = payload["sliceXMm"]
    sliceMeasMv: np.ndarray = payload["sliceMeasMv"]
    sliceSimMv: np.ndarray = payload["sliceSimMv"]
    g4PointsXMm: np.ndarray = payload["g4PointsXMm"]
    g4PointsMv: np.ndarray = payload["g4PointsMv"]
    g4PointsErrMv: np.ndarray = payload["g4PointsErrMv"]

    iterHistoryRmsePct: list[float] = payload["iterHistoryRmsePct"]
    trajectories: list[Any] = payload["trajectories"]
    metrics: dict[str, float] = payload["metrics"]

    txt = _getLocalizedStrings(lang, metrics)

    deltaRUm = nonAxiDeltaRMm * 1000.0
    maxAbsUm = max(80.0, float(np.percentile(np.abs(deltaRUm), 99.5)))
    features = detectDominantSurfaceShapes(thetaDeg1d, zMm1d, deltaRUm, maxFeatures=6, lang=lang)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}" '
        f'width="100%" height="100%" id="lgRootSvg" style="background:#0b1020; user-select:none; font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',\'PingFang SC\',\'Microsoft YaHei\',sans-serif;">'
    )

    # Embedded vector styles
    parts.append(
        """<defs>
  <linearGradient id="bgGrad" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#0b1120"/>
    <stop offset="100%" stop-color="#111827"/>
  </linearGradient>
  <style>
    .panel-card { fill: #131c31; stroke: #283655; stroke-width: 1.5; rx: 12; }
    .panel-title { fill: #f1f5f9; font-size: 18px; font-weight: 700; }
    .panel-sub { fill: #94a3b8; font-size: 12.5px; }
    .axis-label { fill: #cbd5e1; font-size: 12px; }
    .tick-text { fill: #94a3b8; font-size: 11px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .hud-btn { fill: #1e293b; stroke: #475569; stroke-width: 1.2; cursor: pointer; transition: fill 0.15s; }
    .hud-btn:hover { fill: #334155; stroke: #38bdf8; }
    .hud-btn-text { fill: #e2e8f0; font-size: 12.5px; font-weight: 600; pointer-events: none; }
    .badge-pill { fill: #0f172a; stroke: #38bdf8; stroke-width: 1.2; }
    .probe-box { fill: rgba(15, 23, 42, 0.94); stroke: #38bdf8; stroke-width: 1.3; }
  </style>
</defs>"""
    )

    parts.append(f'<rect x="0" y="0" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" fill="url(#bgGrad)"/>')

    # ==================== Zoomable & Pannable Main Viewport Group ====================
    parts.append('<g id="zoomViewport" transform="translate(0,0) scale(1)">')

    # Main Title Banner
    parts.append('<rect x="24" y="72" width="2192" height="66" class="panel-card"/>')
    parts.append(
        f'<text x="46" y="101" fill="#f8fafc" font-size="20" font-weight="700">'
        f'{html.escape(txt["bannerTitle"])}'
        f'</text>'
    )
    parts.append(
        f'<text x="2194" y="100" text-anchor="end" fill="#38bdf8" font-size="12.5" font-weight="600">'
        f'{html.escape(txt["authorTag"])}'
        f'</text>'
    )
    parts.append(
        f'<text x="46" y="124" class="panel-sub">'
        f'{html.escape(txt["bannerSub"])}'
        f'</text>'
    )

    # ==================== Panel A: Unwrapped Inner-Surface Deformation Map (Top-Left) ====================
    paX, paY, paW, paH = 24, 152, 1180, 620
    parts.append(f'<rect x="{paX}" y="{paY}" width="{paW}" height="{paH}" class="panel-card"/>')
    parts.append(
        f'<text x="{paX + 22}" y="{paY + 32}" class="panel-title">'
        f'{html.escape(txt["paTitle"])}'
        f'</text>'
    )
    parts.append(
        f'<text x="{paX + 22}" y="{paY + 52}" class="panel-sub">'
        f'{txt["paSub"]}'
        f'</text>'
    )

    mapX0, mapY0, mapW, mapH = paX + 76, paY + 74, 790, 476
    nThPlot = 90
    nZPlot = 100
    thIdx = np.linspace(0, len(thetaDeg1d) - 1, nThPlot).astype(int)
    zIdx = np.linspace(0, len(zMm1d) - 1, nZPlot).astype(int)
    subDeltaUm = deltaRUm[np.ix_(thIdx, zIdx)]
    subThDeg = thetaDeg1d[thIdx]
    subZMm = zMm1d[zIdx]

    cellW = mapW / nZPlot
    cellH = mapH / nThPlot

    parts.append(
        f'<g id="surfaceHeatmapGroup" data-panel="surface" data-x0="{mapX0}" data-y0="{mapY0}" '
        f'data-w="{mapW}" data-h="{mapH}">'
    )
    for i in range(nThPlot):
        pyCell = mapY0 + mapH - (i + 1) * cellH
        for j in range(nZPlot):
            pxCell = mapX0 + j * cellW
            valUm = float(subDeltaUm[i, j])
            col = colormapDivergingHex(valUm / maxAbsUm)
            parts.append(
                f'<rect x="{pxCell:.2f}" y="{pyCell:.2f}" width="{cellW + 0.45:.2f}" height="{cellH + 0.45:.2f}" fill="{col}"/>'
            )
    parts.append("</g>")

    # Marching Squares Contour Layer
    def surfZToPx(zVal: float) -> float:
        return mapX0 + (zVal / 100.0) * mapW

    def surfThToPx(thDeg: float) -> float:
        return mapY0 + mapH - ((thDeg + 180.0) / 360.0) * mapH

    parts.append('<g id="layerContours">')
    contourStep = 30.0
    levels = [lv for lv in np.arange(-300.0, 301.0, contourStep) if abs(lv) >= 25.0]
    for lv in levels:
        pathStr = extractMarchingSquaresPaths(subDeltaUm, subZMm, subThDeg, float(lv), surfZToPx, surfThToPx)
        if not pathStr:
            continue
        dashAttr = ' stroke-dasharray="4,3"' if lv > 0 else ""
        strokeCol = "#e0f2fe" if lv < 0 else "#fee2e2"
        parts.append(
            f'<path d="{pathStr}" fill="none" stroke="{strokeCol}" stroke-width="0.95" stroke-opacity="0.72"{dashAttr}/>'
        )
    parts.append("</g>")

    # Axes & Ticks for Panel A
    parts.append(f'<rect x="{mapX0}" y="{mapY0}" width="{mapW}" height="{mapH}" fill="none" stroke="#94a3b8" stroke-width="1.2"/>')
    for zTick in range(0, 101, 10):
        tx = surfZToPx(float(zTick))
        parts.append(f'<line x1="{tx:.1f}" y1="{mapY0 + mapH}" x2="{tx:.1f}" y2="{mapY0 + mapH + 6}" stroke="#94a3b8"/>')
        parts.append(f'<text x="{tx:.1f}" y="{mapY0 + mapH + 20}" text-anchor="middle" class="tick-text">{zTick}</text>')
    for thTick in [-180, -120, -60, 0, 60, 120, 180]:
        ty = surfThToPx(float(thTick))
        parts.append(f'<line x1="{mapX0 - 6}" y1="{ty:.1f}" x2="{mapX0}" y2="{ty:.1f}" stroke="#94a3b8"/>')
        parts.append(f'<text x="{mapX0 - 10}" y="{ty + 4:.1f}" text-anchor="end" class="tick-text">{thTick}°</text>')
    parts.append(
        f'<text x="{mapX0 + mapW * 0.5:.1f}" y="{mapY0 + mapH + 40}" text-anchor="middle" class="axis-label">'
        f'{html.escape(txt["paAxisX"])}'
        f'</text>'
    )
    parts.append(
        f'<text x="{paX + 22}" y="{mapY0 + mapH * 0.5:.1f}" text-anchor="middle" class="axis-label" '
        f'transform="rotate(-90 {paX + 22} {mapY0 + mapH * 0.5:.1f})">{html.escape(txt["paAxisY"])}</text>'
    )

    # Colorbar for Panel A
    cbX, cbY, cbW, cbH = mapX0 + mapW + 18, mapY0, 18, mapH
    for k in range(60):
        frac = k / 59.0
        valNorm = 1.0 - 2.0 * frac
        parts.append(
            f'<rect x="{cbX}" y="{cbY + frac * cbH:.1f}" width="{cbW}" height="{cbH / 60.0 + 0.8:.1f}" '
            f'fill="{colormapDivergingHex(valNorm)}"/>'
        )
    parts.append(f'<rect x="{cbX}" y="{cbY}" width="{cbW}" height="{cbH}" fill="none" stroke="#94a3b8"/>')
    for frac, vLabel in [(0.0, f"+{maxAbsUm:.0f}"), (0.25, f"+{0.5 * maxAbsUm:.0f}"), (0.5, "0"), (0.75, f"-{0.5 * maxAbsUm:.0f}"), (1.0, f"-{maxAbsUm:.0f}")]:
        ly = cbY + frac * cbH
        parts.append(f'<text x="{cbX + cbW + 6}" y="{ly + 4:.1f}" class="tick-text">{vLabel} μm</text>')

    # Automatic Dominant Feature Callout Badges
    parts.append('<g id="layerAnnotations">')
    tagW = 148 if lang.lower().startswith("en") else 128
    for feat in features:
        fx = surfZToPx(feat.zMm)
        fy = surfThToPx(feat.thetaDeg)
        ringCol = "#38bdf8" if feat.amplitudeUm < 0 else "#fb7185"
        parts.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="9" fill="none" stroke="{ringCol}" stroke-width="2.2"/>')
        parts.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="2.5" fill="#ffffff"/>')
        tagX = fx + 12 if fx < mapX0 + mapW - (tagW + 18) else fx - tagW - 12
        tagY = max(mapY0 + 6, min(mapY0 + mapH - 34, fy - 16))
        parts.append(
            f'<rect x="{tagX:.1f}" y="{tagY:.1f}" width="{tagW}" height="28" rx="5" '
            f'fill="rgba(15,23,42,0.86)" stroke="{ringCol}" stroke-width="1.1"/>'
        )
        parts.append(
            f'<text x="{tagX + 6:.1f}" y="{tagY + 12:.1f}" fill="#f8fafc" font-size="10.2" font-weight="700">'
            f'#{feat.featureId} {html.escape(feat.kindLabel)} ({feat.amplitudeUm:+.0f}μm)</text>'
        )
        parts.append(
            f'<text x="{tagX + 6:.1f}" y="{tagY + 23:.1f}" fill="#cbd5e1" font-size="9.5">'
            f'z={feat.zMm:.1f}mm, θ={feat.thetaDeg:+.0f}°</text>'
        )
    parts.append("</g>")

    # Panel A Right Inset: Axisymmetric Profile Offset ΔR_axi(z)
    axBoxX, axBoxY, axBoxW, axBoxH = cbX + 82, mapY0, 198, mapH
    parts.append(f'<rect x="{axBoxX}" y="{axBoxY}" width="{axBoxW}" height="{axBoxH}" fill="#0f172a" stroke="#475569"/>')
    parts.append(
        f'<text x="{axBoxX + axBoxW * 0.5:.1f}" y="{axBoxY - 10}" text-anchor="middle" fill="#e2e8f0" font-size="12" font-weight="600">'
        f'{html.escape(txt["paAxiTitle"])}'
        f'</text>'
    )
    axiUm = axiProfileMm * 1000.0
    axiMin = min(-550.0, float(np.min(axiUm)) - 40.0)
    axiMax = max(100.0, float(np.max(axiUm)) + 40.0)

    def axiXToPx(umVal: float) -> float:
        return axBoxX + 14 + ((umVal - axiMin) / max(1e-6, axiMax - axiMin)) * (axBoxW - 28)

    def axiZToPx(zVal: float) -> float:
        return axBoxY + axBoxH - (zVal / 100.0) * axBoxH

    zeroX = axiXToPx(0.0)
    parts.append(f'<line x1="{zeroX:.1f}" y1="{axBoxY}" x2="{zeroX:.1f}" y2="{axBoxY + axBoxH}" stroke="#64748b" stroke-dasharray="3,3"/>')
    ptsAxi = " ".join(f"{axiXToPx(float(axiUm[k])):.1f},{axiZToPx(float(axiProfileZ[k])):.1f}" for k in range(len(axiProfileZ)))
    parts.append(f'<polyline points="{ptsAxi}" fill="none" stroke="#38bdf8" stroke-width="2.4"/>')
    for umTick in [-400, -200, 0]:
        if axiMin <= umTick <= axiMax:
            tx = axiXToPx(float(umTick))
            parts.append(f'<text x="{tx:.1f}" y="{axBoxY + axBoxH + 18}" text-anchor="middle" class="tick-text">{umTick}</text>')
    parts.append(
        f'<text x="{axBoxX + axBoxW * 0.5:.1f}" y="{axBoxY + axBoxH + 38}" text-anchor="middle" class="axis-label">ΔR_axi (μm)</text>'
    )

    # ==================== Panel B: 3D Deformed Surface & Multi-Bounce Ray Trajectories (Top-Right) ====================
    pbX, pbY, pbW, pbH = 1220, 152, 996, 620
    parts.append(f'<rect x="{pbX}" y="{pbY}" width="{pbW}" height="{pbH}" class="panel-card"/>')
    parts.append(
        f'<text x="{pbX + 22}" y="{pbY + 32}" class="panel-title">'
        f'{html.escape(txt["pbTitle"])}'
        f'</text>'
    )
    parts.append(
        f'<text x="{pbX + 22}" y="{pbY + 52}" class="panel-sub">'
        f'{html.escape(txt["pbSub"])}'
        f'</text>'
    )

    center3d = (pbX + 495.0, pbY + 350.0)
    scale3d = 11.2
    exagFactor = 8.0

    # Build semi-transparent shaded 3D quads sorted back-to-front
    nTh3d, nZ3d = 48, 34
    th3dIdx = np.linspace(0, len(thetaRad1d) - 1, nTh3d, endpoint=False).astype(int)
    z3dIdx = np.linspace(0, len(zMm1d) - 1, nZ3d).astype(int)

    quads: list[tuple[float, str, str]] = []
    for ii in range(nTh3d):
        i0 = th3dIdx[ii]
        i1 = th3dIdx[(ii + 1) % nTh3d]
        th0, th1 = float(thetaRad1d[i0]), float(thetaRad1d[i1])
        if ii == nTh3d - 1:
            th1 = th0 + (2.0 * math.pi / nTh3d)
        for jj in range(nZ3d - 1):
            j0, j1 = z3dIdx[jj], z3dIdx[jj + 1]
            z0, z1 = float(zMm1d[j0]), float(zMm1d[j1])

            rBase0 = 8.75 - 0.0475 * z0
            rBase1 = 8.75 - 0.0475 * z1
            d00 = float(nonAxiDeltaRMm[i0, j0])
            d10 = float(nonAxiDeltaRMm[i1, j0])
            d11 = float(nonAxiDeltaRMm[i1, j1])
            d01 = float(nonAxiDeltaRMm[i0, j1])

            r00 = rBase0 + exagFactor * d00
            r10 = rBase0 + exagFactor * d10
            r11 = rBase1 + exagFactor * d11
            r01 = rBase1 + exagFactor * d01

            pts3 = [
                projectPoint3d(r00 * math.cos(th0), r00 * math.sin(th0), z0, center3d, scale3d),
                projectPoint3d(r10 * math.cos(th1), r10 * math.sin(th1), z0, center3d, scale3d),
                projectPoint3d(r11 * math.cos(th1), r11 * math.sin(th1), z1, center3d, scale3d),
                projectPoint3d(r01 * math.cos(th0), r01 * math.sin(th0), z1, center3d, scale3d),
            ]
            avgDepth = 0.25 * sum(p[2] for p in pts3)
            meanUm = 250.0 * (d00 + d10 + d11 + d01)
            fillHex = colormapDivergingHex(meanUm / maxAbsUm)
            opacity = "0.24" if avgDepth > 0.0 else "0.52"
            polyPts = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts3)
            quads.append((avgDepth, polyPts, f'fill="{fillHex}" fill-opacity="{opacity}" stroke="#475569" stroke-width="0.35" stroke-opacity="0.45"'))

    quads.sort(key=lambda item: item[0])
    parts.append('<g id="mesh3dBack">')
    for depthVal, polyStr, styleStr in quads:
        if depthVal <= 0.0:
            parts.append(f'<polygon points="{polyStr}" {styleStr}/>')
    parts.append("</g>")

    # 3D Ray Trajectory Layer sandwiched between back and front wall hemispheres
    parts.append('<g id="layerRays3d">')
    for traj in trajectories:
        scrPts = [projectPoint3d(px, py, pz, center3d, scale3d) for (px, py, pz) in traj.pointsMm]
        pathCoords = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in scrPts)
        if traj.reachedPmt:
            parts.append(
                f'<polyline points="{pathCoords}" fill="none" stroke="#2dd4bf" stroke-width="1.65" stroke-opacity="0.88"/>'
            )
        else:
            parts.append(
                f'<polyline points="{pathCoords}" fill="none" stroke="#fb7185" stroke-width="1.35" stroke-opacity="0.72" stroke-dasharray="4,2.5"/>'
            )
    parts.append("</g>")

    parts.append('<g id="mesh3dFront">')
    for depthVal, polyStr, styleStr in quads:
        if depthVal > 0.0:
            parts.append(f'<polygon points="{polyStr}" {styleStr}/>')
    parts.append("</g>")

    # Reference Aperture & PMT Counterbore Rings
    for zRing, rRing, colRing, lblRing in [
        (0.0, 8.45, "#38bdf8", txt["ring0"]),
        (100.0, 3.72, "#34d399", txt["ring100"]),
        (110.0, 8.18, "#a78bfa", txt["ring110"]),
    ]:
        ringPts = []
        for k in range(73):
            ang = 2.0 * math.pi * k / 72.0
            sx, sy, _ = projectPoint3d(rRing * math.cos(ang), rRing * math.sin(ang), zRing, center3d, scale3d)
            ringPts.append(f"{sx:.1f},{sy:.1f}")
        parts.append(f'<polyline points="{" ".join(ringPts)}" fill="none" stroke="{colRing}" stroke-width="2.1"/>')
        lx, ly, _ = projectPoint3d(0.0, rRing + 1.4, zRing, center3d, scale3d)
        parts.append(f'<text x="{lx:.1f}" y="{ly - 6:.1f}" text-anchor="middle" fill="{colRing}" font-size="11.5" font-weight="600">{html.escape(lblRing)}</text>')

    # Legend Box (Panel B Bottom-Left)
    legBoxW = 540 if lang.lower().startswith("en") else 450
    parts.append(f'<rect x="{pbX + 24}" y="{pbY + pbH - 76}" width="{legBoxW}" height="56" rx="8" fill="#0f172a" stroke="#334155"/>')
    parts.append(f'<line x1="{pbX + 40}" y1="{pbY + pbH - 55}" x2="{pbX + 76}" y2="{pbY + pbH - 55}" stroke="#2dd4bf" stroke-width="2.5"/>')
    parts.append(f'<text x="{pbX + 86}" y="{pbY + pbH - 51}" fill="#e2e8f0" font-size="11.2">{html.escape(txt["pbLeg1"])}</text>')
    parts.append(f'<line x1="{pbX + 40}" y1="{pbY + pbH - 33}" x2="{pbX + 76}" y2="{pbY + pbH - 33}" stroke="#fb7185" stroke-width="2.2" stroke-dasharray="4,2.5"/>')
    parts.append(f'<text x="{pbX + 86}" y="{pbY + pbH - 29}" fill="#e2e8f0" font-size="11.2">{html.escape(txt["pbLeg2"])}</text>')

    # ==================== Panel C: 100x100 Full-Resolution 2D Scan Comparison (Bottom-Left) ====================
    pcX, pcY, pcW, pcH = 24, 790, 1340, 666
    parts.append(f'<rect x="{pcX}" y="{pcY}" width="{pcW}" height="{pcH}" class="panel-card"/>')
    parts.append(
        f'<text x="{pcX + 22}" y="{pcY + 32}" class="panel-title">'
        f'{html.escape(txt["pcTitle"])}'
        f'</text>'
    )
    parts.append(
        f'<text x="{pcX + 22}" y="{pcY + 52}" class="panel-sub">'
        f'{html.escape(txt["pcSub"])}'
        f'</text>'
    )

    def renderScanSubmap(
        subIdx: int,
        titleStr: str,
        gridData: np.ndarray,
        vmin: float,
        vmax: float,
        diverging: bool,
        unitStr: str,
        panelTag: str,
    ) -> None:
        boxSize = 266.0
        gap = 58.0
        sx0 = pcX + 46.0 + subIdx * (boxSize + gap)
        sy0 = pcY + 92.0
        nyG, nxG = gridData.shape
        cw = boxSize / nxG
        ch = boxSize / nyG

        parts.append(f'<text x="{sx0 + boxSize * 0.5:.1f}" y="{sy0 - 12:.1f}" text-anchor="middle" fill="#f1f5f9" font-size="12.5" font-weight="600">{html.escape(titleStr)}</text>')
        parts.append(
            f'<g data-panel="{panelTag}" data-x0="{sx0}" data-y0="{sy0}" data-w="{boxSize}" data-h="{boxSize}">'
        )
        for iy in range(nyG):
            ry = sy0 + boxSize - (iy + 1) * ch
            for ix in range(nxG):
                if not bool(insideMask2d[iy, ix]):
                    continue
                val = float(gridData[iy, ix])
                if not math.isfinite(val):
                    continue
                if diverging:
                    col = colormapDivergingHex(val / max(1e-6, vmax))
                else:
                    col = colormapTurboHex((val - vmin) / max(1e-6, vmax - vmin))
                rx = sx0 + ix * cw
                parts.append(
                    f'<rect x="{rx:.2f}" y="{ry:.2f}" width="{cw + 0.35:.2f}" height="{ch + 0.35:.2f}" fill="{col}"/>'
                )
        parts.append("</g>")
        parts.append(f'<rect x="{sx0:.1f}" y="{sy0:.1f}" width="{boxSize}" height="{boxSize}" fill="none" stroke="#64748b" stroke-width="1.1"/>')

        # Effective clear aperture reference dashed circle
        xMin, xMax = float(scanXLocalMm[0]), float(scanXLocalMm[-1])
        yMin, yMax = float(scanYLocalMm[0]), float(scanYLocalMm[-1])
        cxSvg = sx0 + ((0.0 - xMin) / (xMax - xMin)) * boxSize
        cySvg = sy0 + boxSize - ((0.0 - yMin) / (yMax - yMin)) * boxSize
        rSvg = (8.45 / (xMax - xMin)) * boxSize
        parts.append(f'<circle cx="{cxSvg:.1f}" cy="{cySvg:.1f}" r="{rSvg:.1f}" fill="none" stroke="#e2e8f0" stroke-width="0.9" stroke-dasharray="3,3" stroke-opacity="0.7"/>')

        # Bottom horizontal colorbar
        barY = sy0 + boxSize + 36.0
        for b in range(44):
            bf = b / 43.0
            bCol = colormapDivergingHex(-1.0 + 2.0 * bf) if diverging else colormapTurboHex(bf)
            parts.append(
                f'<rect x="{sx0 + bf * boxSize:.1f}" y="{barY:.1f}" width="{boxSize / 44.0 + 0.6:.1f}" height="12" fill="{bCol}"/>'
            )
        parts.append(f'<rect x="{sx0:.1f}" y="{barY:.1f}" width="{boxSize}" height="12" fill="none" stroke="#64748b"/>')
        parts.append(f'<text x="{sx0:.1f}" y="{barY + 26:.1f}" class="tick-text">{vmin:+.1f}{unitStr}</text>' if diverging else f'<text x="{sx0:.1f}" y="{barY + 26:.1f}" class="tick-text">{vmin:.1f}{unitStr}</text>')
        midVal = 0.5 * (vmin + vmax)
        parts.append(f'<text x="{sx0 + boxSize * 0.5:.1f}" y="{barY + 26:.1f}" text-anchor="middle" class="tick-text">{midVal:+.1f}{unitStr}</text>' if diverging else f'<text x="{sx0 + boxSize * 0.5:.1f}" y="{barY + 26:.1f}" text-anchor="middle" class="tick-text">{midVal:.1f}{unitStr}</text>')
        parts.append(f'<text x="{sx0 + boxSize:.1f}" y="{barY + 26:.1f}" text-anchor="end" class="tick-text">{vmax:+.1f}{unitStr}</text>' if diverging else f'<text x="{sx0 + boxSize:.1f}" y="{barY + 26:.1f}" text-anchor="end" class="tick-text">{vmax:.1f}{unitStr}</text>')

        for xyTick in [-8, -4, 0, 4, 8]:
            tx = sx0 + ((xyTick - xMin) / (xMax - xMin)) * boxSize
            ty = sy0 + boxSize - ((xyTick - yMin) / (yMax - yMin)) * boxSize
            if sx0 <= tx <= sx0 + boxSize:
                parts.append(f'<text x="{tx:.1f}" y="{sy0 + boxSize + 16:.1f}" text-anchor="middle" class="tick-text">{xyTick}</text>')
            if sy0 <= ty <= sy0 + boxSize:
                parts.append(f'<text x="{sx0 - 6:.1f}" y="{ty + 4:.1f}" text-anchor="end" class="tick-text">{xyTick}</text>')

    renderScanSubmap(0, txt["c1Title"], measNetMv, 20.5, 27.4, False, " mV", "c1_meas")
    renderScanSubmap(1, txt["c2Title"], simNetMv, 20.5, 27.4, False, " mV", "c2_sim")
    renderScanSubmap(2, txt["c3Title"], measNonAxiPct, -7.5, 7.5, True, "%", "c3_bump_m")
    renderScanSubmap(3, txt["c4Title"], simNonAxiPct, -7.5, 7.5, True, "%", "c4_bump_s")

    # Physical Mechanism Summary Box (Panel C Lower Half)
    noteY = pcY + 462
    parts.append(f'<rect x="{pcX + 36}" y="{noteY}" width="{pcW - 72}" height="180" rx="8" fill="#0f172a" stroke="#334155"/>')
    parts.append(
        f'<text x="{pcX + 56}" y="{noteY + 28}" fill="#38bdf8" font-size="14" font-weight="700">'
        f'{html.escape(txt["pcNoteTitle"])}'
        f'</text>'
    )
    noteFont = "11.8" if lang.lower().startswith("en") else "12.5"
    for lIdx, lineText in enumerate(txt["pcNotes"]):
        parts.append(
            f'<text x="{pcX + 56}" y="{noteY + 56 + lIdx * 30}" fill="#cbd5e1" font-size="{noteFont}">{html.escape(lineText)}</text>'
        )

    # ==================== Panel D: Quantitative Profiles & Geant4 11.1.3 Cross-Validation (Bottom-Right) ====================
    pdX, pdY, pdW, pdH = 1380, 790, 836, 666
    parts.append(f'<rect x="{pdX}" y="{pdY}" width="{pdW}" height="{pdH}" class="panel-card"/>')
    parts.append(
        f'<text x="{pdX + 22}" y="{pdY + 32}" class="panel-title">'
        f'{html.escape(txt["pdTitle"])}'
        f'</text>'
    )

    # Subplot D1: Azimuthal-Mean Radial Profile (Upper)
    d1X, d1Y, d1W, d1H = pdX + 68, pdY + 62, 736, 232
    parts.append(f'<rect x="{d1X}" y="{d1Y}" width="{d1W}" height="{d1H}" fill="#0f172a" stroke="#475569"/>')
    parts.append(
        f'<text x="{d1X + 12}" y="{d1Y + 20}" fill="#e2e8f0" font-size="12.2" font-weight="600">'
        f'{html.escape(txt["d1Title"])}'
        f'</text>'
    )

    def d1Rx(rVal: float) -> float:
        return d1X + (rVal / 9.4) * d1W

    def d1Vy(mvVal: float) -> float:
        return d1Y + d1H - ((mvVal - 0.0) / 30.0) * d1H

    ptsIdeal = " ".join(f"{d1Rx(float(radialR[i])):.1f},{d1Vy(float(radialIdealMv[i])):.1f}" for i in range(len(radialR)))
    ptsSimR = " ".join(f"{d1Rx(float(radialR[i])):.1f},{d1Vy(float(radialSimMv[i])):.1f}" for i in range(len(radialR)))
    parts.append(f'<polyline points="{ptsIdeal}" fill="none" stroke="#94a3b8" stroke-width="1.8" stroke-dasharray="5,4"/>')
    parts.append(f'<polyline points="{ptsSimR}" fill="none" stroke="#38bdf8" stroke-width="2.6"/>')
    for i in range(len(radialR)):
        cxDot, cyDot = d1Rx(float(radialR[i])), d1Vy(float(radialMeasMv[i]))
        parts.append(f'<circle cx="{cxDot:.1f}" cy="{cyDot:.1f}" r="2.8" fill="#f97316"/>')

    cadX = d1Rx(8.75)
    parts.append(f'<line x1="{cadX:.1f}" y1="{d1Y}" x2="{cadX:.1f}" y2="{d1Y + d1H}" stroke="#ef4444" stroke-dasharray="3,3"/>')
    parts.append(f'<text x="{cadX - 6:.1f}" y="{d1Y + 42}" text-anchor="end" fill="#fca5a5" font-size="10.5">STEP R_in=8.75mm</text>')

    for rTick in range(0, 10, 1):
        tx = d1Rx(float(rTick))
        parts.append(f'<text x="{tx:.1f}" y="{d1Y + d1H + 16}" text-anchor="middle" class="tick-text">{rTick}</text>')
    for vTick in [0, 10, 20, 28]:
        ty = d1Vy(float(vTick))
        parts.append(f'<text x="{d1X - 8}" y="{ty + 4:.1f}" text-anchor="end" class="tick-text">{vTick}</text>')
    parts.append(f'<text x="{d1X + d1W * 0.5:.1f}" y="{d1Y + d1H + 32}" text-anchor="middle" class="axis-label">{html.escape(txt["d1AxisX"])}</text>')
    parts.append(f'<text x="{pdX + 22}" y="{d1Y + d1H * 0.5:.1f}" text-anchor="middle" class="axis-label" transform="rotate(-90 {pdX + 22} {d1Y + d1H * 0.5:.1f})">{html.escape(txt["d1AxisY"])}</text>')

    # Legend D1
    parts.append(f'<line x1="{d1X + 20}" y1="{d1Y + d1H - 46}" x2="{d1X + 48}" y2="{d1Y + d1H - 46}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="5,4"/>')
    parts.append(f'<text x="{d1X + 55}" y="{d1Y + d1H - 42}" fill="#cbd5e1" font-size="11">{html.escape(txt["d1LegIdeal"])}</text>')
    parts.append(f'<circle cx="{d1X + 34}" cy="{d1Y + d1H - 28}" r="3.5" fill="#f97316"/>')
    parts.append(f'<text x="{d1X + 55}" y="{d1Y + d1H - 24}" fill="#cbd5e1" font-size="11">{html.escape(txt["d1LegMeas"])}</text>')
    parts.append(f'<line x1="{d1X + 20}" y1="{d1Y + d1H - 10}" x2="{d1X + 48}" y2="{d1Y + d1H - 10}" stroke="#38bdf8" stroke-width="2.6"/>')
    parts.append(f'<text x="{d1X + 55}" y="{d1Y + d1H - 6}" fill="#cbd5e1" font-size="11">{html.escape(txt["d1LegSim"])}</text>')

    # Subplot D2: Cross-Section Slice + Geant4 11.1.3 Independent Validation (Bottom-Left of Panel D)
    d2X, d2Y, d2W, d2H = pdX + 68, pdY + 356, 480, 260
    parts.append(f'<rect x="{d2X}" y="{d2Y}" width="{d2W}" height="{d2H}" fill="#0f172a" stroke="#475569"/>')
    parts.append(
        f'<text x="{d2X + 10}" y="{d2Y + 20}" fill="#e2e8f0" font-size="12" font-weight="600">'
        f'{html.escape(txt["d2Title"])}'
        f'</text>'
    )

    def d2Xx(xVal: float) -> float:
        return d2X + ((xVal + 8.5) / 17.0) * d2W

    def d2Vy(mvVal: float) -> float:
        return d2Y + d2H - ((mvVal - 20.5) / 7.5) * d2H

    validSlice = (np.abs(sliceXMm) < 8.1) & np.isfinite(sliceMeasMv) & np.isfinite(sliceSimMv)
    sX = sliceXMm[validSlice]
    sM = sliceMeasMv[validSlice]
    sS = sliceSimMv[validSlice]
    ptsSlMeas = " ".join(f"{d2Xx(float(sX[k])):.1f},{max(d2Y + 4, min(d2Y + d2H - 4, d2Vy(float(sM[k])))):.1f}" for k in range(len(sX)))
    ptsSlSim = " ".join(f"{d2Xx(float(sX[k])):.1f},{max(d2Y + 4, min(d2Y + d2H - 4, d2Vy(float(sS[k])))):.1f}" for k in range(len(sX)))
    parts.append(f'<polyline points="{ptsSlMeas}" fill="none" stroke="#f97316" stroke-width="1.6" stroke-opacity="0.85"/>')
    parts.append(f'<polyline points="{ptsSlSim}" fill="none" stroke="#38bdf8" stroke-width="2.5"/>')

    # Geant4 validation markers and binomial error bars
    for k in range(len(g4PointsXMm)):
        gxPx = d2Xx(float(g4PointsXMm[k]))
        gyPx = max(d2Y + 8, min(d2Y + d2H - 8, d2Vy(float(g4PointsMv[k]))))
        errPx = (float(g4PointsErrMv[k]) / 7.5) * d2H
        parts.append(f'<line x1="{gxPx:.1f}" y1="{gyPx - errPx:.1f}" x2="{gxPx:.1f}" y2="{gyPx + errPx:.1f}" stroke="#4ade80" stroke-width="1.8"/>')
        parts.append(f'<rect x="{gxPx - 3.5:.1f}" y="{gyPx - 3.5:.1f}" width="7" height="7" fill="#22c55e" stroke="#052e16"/>')

    for xTk in [-8, -4, 0, 4, 8]:
        tx = d2Xx(float(xTk))
        parts.append(f'<text x="{tx:.1f}" y="{d2Y + d2H + 16}" text-anchor="middle" class="tick-text">{xTk}</text>')
    for vTk in [21, 23, 25, 27]:
        ty = d2Vy(float(vTk))
        parts.append(f'<text x="{d2X - 6}" y="{ty + 4:.1f}" text-anchor="end" class="tick-text">{vTk}</text>')
    parts.append(f'<text x="{d2X + d2W * 0.5:.1f}" y="{d2Y + d2H + 34}" text-anchor="middle" class="axis-label">{html.escape(txt["d2AxisX"])}</text>')
    if len(g4PointsXMm) > 0:
        parts.append(f'<rect x="{d2X + 14}" y="{d2Y + d2H - 24}" width="8" height="8" fill="#22c55e"/>')
        parts.append(f'<text x="{d2X + 28}" y="{d2Y + d2H - 16}" fill="#bbf7d0" font-size="10.8">{html.escape(txt["d2LegG4"])}</text>')

    # Subplot D3: Multi-Stage Nonlinear Optimization Convergence (Bottom-Right of Panel D)
    d3X, d3Y, d3W, d3H = d2X + d2W + 30, pdY + 356, 226, 260
    parts.append(f'<rect x="{d3X}" y="{d3Y}" width="{d3W}" height="{d3H}" fill="#0f172a" stroke="#475569"/>')
    parts.append(
        f'<text x="{d3X + d3W * 0.5:.1f}" y="{d3Y + 20}" text-anchor="middle" fill="#e2e8f0" font-size="12" font-weight="600">'
        f'{html.escape(txt["d3Title"])}'
        f'</text>'
    )
    maxErrPlot = max(3.5, max(iterHistoryRmsePct) * 1.1)
    nIt = max(1, len(iterHistoryRmsePct) - 1)

    def d3Ix(itVal: int) -> float:
        return d3X + 28 + (itVal / nIt) * (d3W - 44)

    def d3Ey(errPct: float) -> float:
        return d3Y + d3H - 22 - (errPct / maxErrPlot) * (d3H - 58)

    noiseY = d3Ey(metrics["noiseFloorPct"])
    parts.append(f'<line x1="{d3X + 20}" y1="{noiseY:.1f}" x2="{d3X + d3W - 10}" y2="{noiseY:.1f}" stroke="#fbbf24" stroke-dasharray="3,3"/>')
    parts.append(f'<text x="{d3X + 26}" y="{noiseY - 5:.1f}" fill="#fde68a" font-size="9.5">{html.escape(txt["d3Noise"])}</text>')

    convPts = " ".join(f"{d3Ix(k):.1f},{d3Ey(val):.1f}" for k, val in enumerate(iterHistoryRmsePct))
    parts.append(f'<polyline points="{convPts}" fill="none" stroke="#a78bfa" stroke-width="2.4"/>')
    for k, val in enumerate(iterHistoryRmsePct):
        parts.append(f'<circle cx="{d3Ix(k):.1f}" cy="{d3Ey(val):.1f}" r="3.2" fill="#c4b5fd"/>')
    parts.append(f'<text x="{d3X + d3W * 0.5:.1f}" y="{d3Y + d3H + 34}" text-anchor="middle" class="axis-label">{html.escape(txt["d3AxisX"])}</text>')

    parts.append("</g>")  # End zoomViewport

    # ==================== Fixed Top Interactive HUD Control Bar ====================
    parts.append('<g id="fixedHudBar">')
    parts.append(f'<rect x="0" y="0" width="{CANVAS_WIDTH}" height="56" fill="#0f172a" stroke="#1e293b" stroke-width="1.5"/>')

    buttons = [
        ("btnZoomIn", 24, 110, txt["btnZoomIn"]),
        ("btnZoomOut", 144, 110, txt["btnZoomOut"]),
        ("btnReset", 264, 120, txt["btnReset"]),
        ("btnToggleContour", 396, 150, txt["btnContourOn"]),
        ("btnToggleAnnot", 556, 156, txt["btnAnnotOn"]),
        ("btnToggleRays", 722, 160, txt["btnRaysOn"]),
    ]
    for btnId, bx, bw, bText in buttons:
        parts.append(f'<g id="{btnId}">')
        parts.append(f'<rect x="{bx}" y="11" width="{bw}" height="34" rx="7" class="hud-btn"/>')
        parts.append(f'<text id="{btnId}Text" x="{bx + bw * 0.5:.1f}" y="33" text-anchor="middle" class="hud-btn-text">{html.escape(bText)}</text>')
        parts.append("</g>")

    parts.append('<rect x="900" y="11" width="1316" height="34" rx="7" class="badge-pill"/>')
    parts.append(
        f'<text id="hudReadoutText" x="918" y="33" fill="#7dd3fc" font-size="12.5" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">'
        f'{html.escape(txt["hudDefault"])}'
        f'</text>'
    )
    parts.append("</g>")

    # ==================== Embedded Pure-Vector Zoom / Pan / Probe JavaScript Engine ====================
    surfPreview = np.round(subDeltaUm, 1).tolist()
    measPreview = np.where(insideMask2d, np.round(measNetMv, 2), -1.0).tolist()
    simPreview = np.where(insideMask2d, np.round(simNetMv, 2), -1.0).tolist()

    isEnJs = "true" if lang.lower().startswith("en") else "false"

    scriptContent = f"""<![CDATA[
(function() {{
  const svg = document.getElementById('lgRootSvg');
  const vp = document.getElementById('zoomViewport');
  const readout = document.getElementById('hudReadoutText');
  const layerContours = document.getElementById('layerContours');
  const layerAnnot = document.getElementById('layerAnnotations');
  const layerRays = document.getElementById('layerRays3d');
  const isEn = {isEnJs};

  const surfData = {surfPreview};
  const measData = {measPreview};
  const simData = {simPreview};
  const scanXMin = {float(scanXLocalMm[0]):.3f}, scanXMax = {float(scanXLocalMm[-1]):.3f};
  const scanYMin = {float(scanYLocalMm[0]):.3f}, scanYMax = {float(scanYLocalMm[-1]):.3f};

  let scale = 1.0, tx = 0.0, ty = 0.0;
  let dragging = false, startX = 0.0, startY = 0.0;
  let showContours = true, showAnnot = true, showRays = true;

  function applyTransform() {{
    vp.setAttribute('transform', 'translate(' + tx.toFixed(2) + ',' + ty.toFixed(2) + ') scale(' + scale.toFixed(4) + ')');
  }}

  function clientToSvg(evt) {{
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX;
    pt.y = evt.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }}

  function zoomAt(factor, cx, cy) {{
    const nextScale = Math.max(0.55, Math.min(24.0, scale * factor));
    const ratio = nextScale / scale;
    tx = cx - (cx - tx) * ratio;
    ty = cy - (cy - ty) * ratio;
    scale = nextScale;
    applyTransform();
  }}

  svg.addEventListener('wheel', function(evt) {{
    evt.preventDefault();
    const p = clientToSvg(evt);
    const factor = evt.deltaY < 0 ? 1.16 : (1.0 / 1.16);
    zoomAt(factor, p.x, p.y);
  }}, {{ passive: false }});

  svg.addEventListener('mousedown', function(evt) {{
    if (evt.clientY < 58) return;
    dragging = true;
    const p = clientToSvg(evt);
    startX = p.x - tx;
    startY = p.y - ty;
  }});

  window.addEventListener('mousemove', function(evt) {{
    const p = clientToSvg(evt);
    if (dragging) {{
      tx = p.x - startX;
      ty = p.y - startY;
      applyTransform();
      return;
    }}
    const wx = (p.x - tx) / scale;
    const wy = (p.y - ty) / scale;

    // Check if cursor is inside Panel A unwrapped surface map
    const sX0 = {mapX0}, sY0 = {mapY0}, sW = {mapW}, sH = {mapH};
    if (wx >= sX0 && wx <= sX0 + sW && wy >= sY0 && wy <= sY0 + sH) {{
      const uZ = (wx - sX0) / sW;
      const uTh = 1.0 - (wy - sY0) / sH;
      const zMm = uZ * 100.0;
      const thDeg = -180.0 + uTh * 360.0;
      const j = Math.min({nZPlot - 1}, Math.max(0, Math.floor(uZ * {nZPlot})));
      const i = Math.min({nThPlot - 1}, Math.max(0, Math.floor(uTh * {nThPlot})));
      const dUm = surfData[i][j];
      if (isEn) {{
        const kind = dUm < 0 ? 'Inward Bulge' : 'Outward Dent';
        readout.textContent = '[Zoom ' + scale.toFixed(2) + '×] Inner Surface Probe: z = ' + zMm.toFixed(1) + ' mm, θ = ' + thDeg.toFixed(1) + '°, Non-axi ΔR = ' + (dUm >= 0 ? '+' : '') + dUm.toFixed(1) + ' μm (' + kind + ')';
      }} else {{
        const kind = dUm < 0 ? '内凸鼓包' : '外扩凹陷';
        readout.textContent = '[缩放 ' + scale.toFixed(2) + '×] 内表面探针: z = ' + zMm.toFixed(1) + ' mm, θ = ' + thDeg.toFixed(1) + '°, 非对称形变 ΔR = ' + (dUm >= 0 ? '+' : '') + dUm.toFixed(1) + ' μm (' + kind + ')';
      }}
      return;
    }}

    // Check if cursor is inside any of the four Panel C 2D scan submaps
    for (let k = 0; k < 4; k++) {{
      const bx = {pcX + 46.0} + k * (266.0 + 58.0);
      const by = {pcY + 92.0};
      if (wx >= bx && wx <= bx + 266.0 && wy >= by && wy <= by + 266.0) {{
        const ux = (wx - bx) / 266.0;
        const uy = 1.0 - (wy - by) / 266.0;
        const xMm = scanXMin + ux * (scanXMax - scanXMin);
        const yMm = scanYMin + uy * (scanYMax - scanYMin);
        const ix = Math.min(99, Math.max(0, Math.floor(ux * 100)));
        const iy = Math.min(99, Math.max(0, Math.floor(uy * 100)));
        const vm = measData[iy][ix];
        const vs = simData[iy][ix];
        if (vm > 0) {{
          const diffPct = ((vs - vm) / vm * 100.0).toFixed(2);
          if (isEn) {{
            readout.textContent = '[Zoom ' + scale.toFixed(2) + '×] Entry Scan Point: X\\'=' + xMm.toFixed(2) + ' mm, Y\\'=' + yMm.toFixed(2) + ' mm | Meas=' + vm.toFixed(2) + ' mV, Sim=' + vs.toFixed(2) + ' mV (Diff ' + diffPct + '%)';
          }} else {{
            readout.textContent = '[缩放 ' + scale.toFixed(2) + '×] 入射口扫描点: X\\'=' + xMm.toFixed(2) + ' mm, Y\\'=' + yMm.toFixed(2) + ' mm | 实测=' + vm.toFixed(2) + ' mV, 模拟=' + vs.toFixed(2) + ' mV (偏差 ' + diffPct + '%)';
          }}
        }} else {{
          if (isEn) {{
            readout.textContent = '[Zoom ' + scale.toFixed(2) + '×] Outside Clear Aperture: X\\'=' + xMm.toFixed(2) + ' mm, Y\\'=' + yMm.toFixed(2) + ' mm (Background Region)';
          }} else {{
            readout.textContent = '[缩放 ' + scale.toFixed(2) + '×] 入射口外侧端面: X\\'=' + xMm.toFixed(2) + ' mm, Y\\'=' + yMm.toFixed(2) + ' mm (通光孔外本底区)';
          }}
        }}
        return;
      }}
    }}
  }});

  window.addEventListener('mouseup', function() {{ dragging = false; }});
  svg.addEventListener('dblclick', function() {{ scale = 1.0; tx = 0.0; ty = 0.0; applyTransform(); }});

  document.getElementById('btnZoomIn').addEventListener('click', function() {{ zoomAt(1.3, {CANVAS_WIDTH * 0.5}, {CANVAS_HEIGHT * 0.5}); }});
  document.getElementById('btnZoomOut').addEventListener('click', function() {{ zoomAt(1.0 / 1.3, {CANVAS_WIDTH * 0.5}, {CANVAS_HEIGHT * 0.5}); }});
  document.getElementById('btnReset').addEventListener('click', function() {{ scale = 1.0; tx = 0.0; ty = 0.0; applyTransform(); }});

  document.getElementById('btnToggleContour').addEventListener('click', function() {{
    showContours = !showContours;
    layerContours.setAttribute('display', showContours ? 'inline' : 'none');
    document.getElementById('btnToggleContourText').textContent = showContours
      ? (isEn ? '✓ Contours: ON' : '✓ 等高线: 开启')
      : (isEn ? '○ Contours: OFF' : '○ 等高线: 隐藏');
  }});

  document.getElementById('btnToggleAnnot').addEventListener('click', function() {{
    showAnnot = !showAnnot;
    layerAnnot.setAttribute('display', showAnnot ? 'inline' : 'none');
    document.getElementById('btnToggleAnnotText').textContent = showAnnot
      ? (isEn ? '✓ Callouts: ON' : '✓ 形状标注: 开启')
      : (isEn ? '○ Callouts: OFF' : '○ 形状标注: 隐藏');
  }});

  document.getElementById('btnToggleRays').addEventListener('click', function() {{
    showRays = !showRays;
    layerRays.setAttribute('display', showRays ? 'inline' : 'none');
    document.getElementById('btnToggleRaysText').textContent = showRays
      ? (isEn ? '✓ 3D Rays: ON' : '✓ 3D反射光路: 开启')
      : (isEn ? '○ 3D Rays: OFF' : '○ 3D反射光路: 隐藏');
  }});
}})();
]]>"""
    parts.append(f"<script>{scriptContent}</script>")
    parts.append("</svg>")

    outputSvgPath.write_text("\n".join(parts), encoding="utf-8")
    logger.info(
        "Generated interactive zoomable SVG (%s): %s (%.2f MB)",
        lang,
        outputSvgPath,
        outputSvgPath.stat().st_size / (1024.0 * 1024.0),
    )
