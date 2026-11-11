"""
Animated vector diagram builder for photon specular propagation and PMT readout.

Generates standalone, resolution-independent SMIL-animated SVG diagrams
comparing photon transport inside an ideal STEP conical light guide against
a reconstructed imperfect surface containing micro-scale inward bulges.

Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""

import logging
import math
from pathlib import Path
from typing import Any

__author__ = "Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>"

SVG_WIDTH: int = 1600
SVG_HEIGHT: int = 980


def traceMeridionalRay2d(
    ySourceMm: float,
    angleRad: float,
    hasBulge: bool,
) -> tuple[list[tuple[float, float]], bool, int]:
    """
    Trace an exact 2D multi-bounce specular ray inside the meridional plane
    (z in [0, 100] mm, R_in = 8.44 mm -> R_out = 4.00 mm) to produce
    physically self-consistent SVG trajectory polylines.
    """
    zLen = 100.0
    rIn = 8.44
    rOut = 4.00
    baseSlope = (rOut - rIn) / zLen

    def wallProfileUpper(zVal: float) -> tuple[float, float]:
        rBase = rIn + baseSlope * zVal
        if not hasBulge:
            return rBase, baseSlope
        # NOTE: Construct an inward bulge near z=34 mm (moderately exaggerated for visual clarity)
        dz1 = (zVal - 34.0) / 8.5
        bulge1 = -0.92 * math.exp(-0.5 * dz1 * dz1)
        dBulge1 = -0.92 * (-dz1 / 8.5) * math.exp(-0.5 * dz1 * dz1)

        dz2 = (zVal - 68.0) / 10.0
        bulge2 = 0.42 * math.exp(-0.5 * dz2 * dz2)
        dBulge2 = 0.42 * (-dz2 / 10.0) * math.exp(-0.5 * dz2 * dz2)
        return rBase + bulge1 + bulge2, baseSlope + dBulge1 + dBulge2

    def wallProfileLower(zVal: float) -> tuple[float, float]:
        rBase = -(rIn + baseSlope * zVal)
        if not hasBulge:
            return rBase, -baseSlope
        dz1 = (zVal - 52.0) / 9.5
        bulge1 = 0.65 * math.exp(-0.5 * dz1 * dz1)
        dBulge1 = 0.65 * (-dz1 / 9.5) * math.exp(-0.5 * dz1 * dz1)
        return rBase + bulge1, -baseSlope + dBulge1

    currZ = 0.0
    currY = ySourceMm
    vz = math.cos(angleRad)
    vy = math.sin(angleRad)
    pts: list[tuple[float, float]] = [(currZ, currY)]
    bounces = 0

    for _ in range(16):
        bestT = 1e9
        hitType = "none"

        if vz > 1e-6:
            tExit = (zLen - currZ) / vz
            yExit = currY + tExit * vy
            if abs(yExit) <= rOut + 0.05:
                bestT = tExit
                hitType = "pmt"
        elif vz < -1e-6 and bounces > 0:
            tBack = -currZ / vz
            if tBack > 1e-3:
                bestT = tBack
                hitType = "escape"

        # Fine forward stepping combined with Newton refinement for curved wall intersection
        dtScan = 0.8
        for stepIdx in range(1, 180):
            tTry = stepIdx * dtScan
            if tTry >= bestT:
                break
            zTry = currZ + tTry * vz
            if zTry <= 0.0 or zTry >= zLen:
                continue
            yTry = currY + tTry * vy
            rUp, _ = wallProfileUpper(zTry)
            rLo, _ = wallProfileLower(zTry)
            if yTry >= rUp:
                tRefine = tTry - 0.4
                for _ in range(6):
                    zRef = max(0.0, min(zLen, currZ + tRefine * vz))
                    rVal, drDz = wallProfileUpper(zRef)
                    fVal = (currY + tRefine * vy) - rVal
                    dfDt = vy - drDz * vz
                    if abs(dfDt) > 1e-6:
                        tRefine -= fVal / dfDt
                if 1e-3 < tRefine < bestT:
                    bestT = tRefine
                    hitType = "upper"
                break
            if yTry <= rLo:
                tRefine = tTry - 0.4
                for _ in range(6):
                    zRef = max(0.0, min(zLen, currZ + tRefine * vz))
                    rVal, drDz = wallProfileLower(zRef)
                    fVal = (currY + tRefine * vy) - rVal
                    dfDt = vy - drDz * vz
                    if abs(dfDt) > 1e-6:
                        tRefine -= fVal / dfDt
                if 1e-3 < tRefine < bestT:
                    bestT = tRefine
                    hitType = "lower"
                break

        if hitType == "none" or bestT > 1e8:
            break

        currZ += bestT * vz
        currY += bestT * vy
        pts.append((currZ, currY))

        if hitType == "pmt":
            return pts, True, bounces
        if hitType == "escape":
            return pts, False, bounces

        bounces += 1
        if hitType == "upper":
            _, slopeW = wallProfileUpper(currZ)
            normLen = math.hypot(slopeW, 1.0)
            nz, ny = slopeW / normLen, -1.0 / normLen
        else:
            _, slopeW = wallProfileLower(currZ)
            normLen = math.hypot(slopeW, 1.0)
            nz, ny = -slopeW / normLen, 1.0 / normLen

        dotVal = vz * nz + vy * ny
        vz = vz - 2.0 * dotVal * nz
        vy = vy - 2.0 * dotVal * ny

    return pts, False, bounces


def getLocalizedLabels(lang: str, totalCorr: float, rmsePct: float) -> dict[str, Any]:
    """
    Return localized UI copy dictionary for Simplified Chinese ('zh') or English ('en').
    """
    if lang == "en":
        return {
            "fontStack": "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif",
            "hdrTitle": "LHCb PicoCal Conical Light-Guide Specular Photon Transport &amp; PMT Readout",
            "hdrTitleSize": "21.5",
            "hdrSub": (
                f"Geometry: L = 100.0 mm | Ø17.5 mm → Ø8.0 mm | Full-Map r = {totalCorr:.4f}, "
                f"Inner RMSE = {rmsePct:.2f}% | Author: Hengne Li (Hengne.Li@cern.ch)"
            ),
            "legX": 1050,
            "legGuided": "Guided Photon → PMT",
            "legEscape": "Turnaround Escaped Photon",
            "legMirror": "Reflective Cone Wall (R≈98.5%)",
            "legStep": "Nominal STEP Cone Profile",
            "legCol2X": 236,
            "legLine2X": 244,
            "inWin": "Entrance Ø17.5 mm",
            "srcLabel": "Diffuse Laser Spot",
            "pmtTitle": "PMT Photomultiplier",
            "pmtSub": "Ø8.0 mm Exit → Photocathode",
            "oscTitle": "PMT Anode Pulse",
            "bulgeBoxW": 462,
            "bulgeTextXOff": 273.0,
            "bulgeFontSize": "11.3",
            "bulgeText": "Inward Bulge (ΔR ≈ -187 μm): Upstream Steeper Slope Causes Turnaround Escape",
            "chATitle": "A. Ideal STEP Conical Surface (ΔR = 0 μm)",
            "chASub": "Diffuse rays undergo regular specular reflections (+2α = 5.44° per bounce); all sub-critical photons reach the PMT photocathode.",
            "chAMv": "Net Response: 27.4 mV (Smooth Symmetric Profile)",
            "chBTitle": "B. Imperfect Machined/Film Inner Surface (Micro-Bulges)",
            "chBSub": "Rays hitting the upstream steeper slope of an inward bulge suffer early turnaround escape (pink dashed), creating dark scan pits.",
            "chBMv": "Net Response: 24.8 mV (-9.5% Local Dip &amp; Ripple)",
            "z0": "z = 0 mm (Laser Entrance)",
            "z34": "z ≈ 34 mm (Primary Bulge Zone)",
            "z68": "z ≈ 68 mm (Mid-Rear Refocusing Zone)",
            "z100": "z = 100 mm (PMT Exit Coupling)",
        }
    return {
        "fontStack": "-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif",
        "hdrTitle": "LHCb PicoCal 锥形光导光子镜面反射传输与 PMT 读出动态物理演示",
        "hdrTitleSize": "23.5",
        "hdrSub": (
            f"长 100.0 mm | Ø17.5 → Ø8.0 mm | 全图相关 r = {totalCorr:.4f}，圆内 RMSE = {rmsePct:.2f}% "
            f"| Hengne Li (李衡讷) · SCNU / CERN LHCb"
        ),
        "legX": 1090,
        "legGuided": "成功到达 PMT 的传导光子",
        "legEscape": "受锥度/鼓包偏折返逃逸光子",
        "legMirror": "高反射膜锥壁镜面 (R ≈ 98.5%)",
        "legStep": "STEP 标称基准锥面线",
        "legCol2X": 212,
        "legLine2X": 252,
        "inWin": "入射圆窗 Ø17.5 mm",
        "srcLabel": "激光漫散射点源",
        "pmtTitle": "PMT 光电倍增管",
        "pmtSub": "Ø8.0 mm 出射窗 → 光阴极",
        "oscTitle": "PMT 阳极负脉冲读出波形",
        "bulgeBoxW": 318,
        "bulgeTextXOff": 201.0,
        "bulgeFontSize": "12.2",
        "bulgeText": "内凸鼓包 (ΔR ≈ -187 μm)：迎光陡坡致临界光线折返",
        "chATitle": "A. 理想 STEP 圆锥内表面（完美加工面形，ΔR = 0 μm）",
        "chASub": "漫散射光线经圆锥内壁平滑高反膜规律镜面反射（单次反射极角增加 2α = 5.44°），临界角以内光子全部稳定进入右侧 PMT 光阴极",
        "chAMv": "净读出响应：27.4 mV（平滑对称曲线）",
        "chBTitle": "B. 实际加工/贴膜不完美内表面（含微米级局部凸起与凹陷）",
        "chBSub": "光线撞击鼓包「迎光陡坡」时局部锥角骤增，原本可到达 PMT 的临界角光子提前调头从大口逃逸（粉红虚线），造成扫描图对应 (X, Y) 处出现暗坑",
        "chBMv": "净读出响应：24.8 mV（局部下降 -9.5%，形成凹凸纹理）",
        "z0": "z = 0 mm (激光入射面)",
        "z34": "z ≈ 34 mm (主鼓包敏感区)",
        "z68": "z ≈ 68 mm (中后段反射聚焦区)",
        "z100": "z = 100 mm (PMT 耦合出口)",
    }


def buildAnimatedPropagationSvg(
    outputPath: Path,
    metrics: dict[str, Any],
    lang: str = "zh",
) -> None:
    """
    Build an animated vector SVG illustrating multi-bounce specular photon transport
    and PMT anode pulse readout inside ideal vs imperfect conical light guides.
    """
    logger = logging.getLogger(__name__)
    outputPath.parent.mkdir(parents=True, exist_ok=True)

    rmsePct = float(metrics.get("innerRmsePct", 1.56))
    totalCorr = float(metrics.get("totalCorr", 0.975))
    labels = getLocalizedLabels(lang=lang, totalCorr=totalCorr, rmsePct=rmsePct)

    def mapCoord(zMm: float, yMm: float, centerY: float) -> tuple[float, float]:
        return 170.0 + zMm * 8.4, centerY - yMm * 14.5

    def renderChannelGroup(
        channelId: str,
        titleText: str,
        subtitleText: str,
        centerY: float,
        hasBulge: bool,
        pmtMvText: str,
        pulseScale: float,
    ) -> str:
        zSamples = [i * 1.0 for i in range(101)]
        upperPts: list[str] = []
        lowerPts: list[str] = []
        rIn, rOut, zLen = 8.44, 4.00, 100.0
        baseSlope = (rOut - rIn) / zLen

        for zv in zSamples:
            rBase = rIn + baseSlope * zv
            if hasBulge:
                dz1 = (zv - 34.0) / 8.5
                dz2 = (zv - 68.0) / 10.0
                rUp = rBase - 0.92 * math.exp(-0.5 * dz1 * dz1) + 0.42 * math.exp(-0.5 * dz2 * dz2)
                dzLo = (zv - 52.0) / 9.5
                rLo = -(rBase - 0.65 * math.exp(-0.5 * dzLo * dzLo))
            else:
                rUp, rLo = rBase, -rBase
            ux, uy = mapCoord(zv, rUp, centerY)
            lx, ly = mapCoord(zv, rLo, centerY)
            upperPts.append(f"{ux:.1f},{uy:.1f}")
            lowerPts.append(f"{lx:.1f},{ly:.1f}")

        polyPoints = " ".join(upperPts + list(reversed(lowerPts)))
        upPathD = "M " + " L ".join(upperPts)
        loPathD = "M " + " L ".join(lowerPts)

        # NOTE: Meridional critical angle is arcsin(4.00 / 8.44) ~ 28.3 deg; sample near-critical rays
        rayAnglesDeg = [-30.8, -27.4, -18.5, -7.5, 4.5, 16.5, 26.9, 28.0]
        rayPathsSvg: list[str] = []
        for rIdx, deg in enumerate(rayAnglesDeg):
            ptsMm, reachedPmt, _ = traceMeridionalRay2d(
                ySourceMm=1.8, angleRad=math.radians(deg), hasBulge=hasBulge
            )
            svgCoords = [mapCoord(zp, yp, centerY) for zp, yp in ptsMm]
            if reachedPmt:
                lastX, lastY = svgCoords[-1]
                svgCoords.append((lastX + 34.0, lastY * 0.98 + centerY * 0.02))
            pathD = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in svgCoords)
            strokeCol = "#22d3ee" if reachedPmt else "#fb7185"
            opacityBase = "0.52" if reachedPmt else "0.45"
            dashStyle = "none" if reachedPmt else "7,5"
            durSec = 2.6 + (rIdx % 4) * 0.38
            beginSec = rIdx * 0.28
            photonColor = "#67e8f9" if reachedPmt else "#fda4af"

            rayPathsSvg.append(
                f'<path id="{channelId}-ray-{rIdx}" d="{pathD}" fill="none" stroke="{strokeCol}" '
                f'stroke-width="1.9" stroke-opacity="{opacityBase}" stroke-dasharray="{dashStyle}" />'
            )
            rayPathsSvg.append(
                f'<circle r="4.2" fill="{photonColor}" filter="url(#photonGlow)">'
                f'<animateMotion dur="{durSec:.2f}s" begin="{beginSec:.2f}s" repeatCount="indefinite" path="{pathD}" />'
                f"</circle>"
            )

        inTopX, inTopY = mapCoord(0.0, 8.44, centerY)
        inBotX, inBotY = mapCoord(0.0, -8.44, centerY)
        outTopX, outTopY = mapCoord(100.0, 4.00, centerY)
        _, outBotY = mapCoord(100.0, -4.00, centerY)
        srcX, srcY = mapCoord(0.0, 1.8, centerY)

        pmtX = outTopX + 4.0
        pmtTop = centerY - 96.0
        pmtHeight = 192.0
        oscX = 1265.0
        oscY = centerY + 32.0
        peakDrop = 92.0 * pulseScale
        oscWaveD = (
            f"M {oscX:.1f},{oscY - 50:.1f} L {oscX + 48:.1f},{oscY - 50:.1f} "
            f"Q {oscX + 66:.1f},{oscY - 50:.1f} {oscX + 78:.1f},{oscY - 50 + peakDrop:.1f} "
            f"Q {oscX + 92:.1f},{oscY - 50 + peakDrop * 1.06:.1f} {oscX + 124:.1f},{oscY - 50 + peakDrop * 0.32:.1f} "
            f"Q {oscX + 162:.1f},{oscY - 50:.1f} {oscX + 255:.1f},{oscY - 50:.1f}"
        )
        waveColor = "#34d399" if not hasBulge else "#fbbf24"

        bulgeCallout = ""
        if hasBulge:
            bx, by = mapCoord(34.0, rIn + baseSlope * 34.0 - 0.92, centerY)
            idealBx, idealBy = mapCoord(0.0, 8.44, centerY)
            idealEx, idealEy = mapCoord(100.0, 4.00, centerY)
            boxW = int(labels["bulgeBoxW"])
            txtXOff = float(labels["bulgeTextXOff"])
            bulgeCallout = f"""
            <line x1="{idealBx:.1f}" y1="{idealBy:.1f}" x2="{idealEx:.1f}" y2="{idealEy:.1f}"
                  stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="5,5" stroke-opacity="0.65" />
            <circle cx="{bx:.1f}" cy="{by:.1f}" r="6.5" fill="none" stroke="#f43f5e" stroke-width="2.0">
              <animate attributeName="r" values="5.5;10.5;5.5" dur="1.8s" repeatCount="indefinite" />
              <animate attributeName="stroke-opacity" values="1.0;0.3;1.0" dur="1.8s" repeatCount="indefinite" />
            </circle>
            <line x1="{bx + 6:.1f}" y1="{by + 4:.1f}" x2="{bx + 42:.1f}" y2="{by + 24:.1f}" stroke="#fb7185" stroke-width="1.6" />
            <rect x="{bx + 42:.1f}" y="{by + 12:.1f}" width="{boxW}" height="26" rx="6" fill="#1e1b4b" fill-opacity="0.92" stroke="#fb7185" stroke-width="1.2" />
            <text x="{bx + txtXOff:.1f}" y="{by + 29:.1f}" fill="#fecdd3" font-size="{labels['bulgeFontSize']}" font-weight="600" text-anchor="middle">{labels['bulgeText']}</text>
            """

        return f"""
        <g id="{channelId}">
          <rect x="40" y="{centerY - 178:.1f}" width="1520" height="352" rx="16" fill="#0f172a" stroke="#1e293b" stroke-width="1.8" />
          <text x="68" y="{centerY - 144:.1f}" fill="#f8fafc" font-size="19" font-weight="700">{titleText}</text>
          <text x="68" y="{centerY - 120:.1f}" fill="#94a3b8" font-size="13.2">{subtitleText}</text>
          <polygon points="{polyPoints}" fill="url(#coneInteriorGrad)" />
          <path d="{upPathD}" fill="none" stroke="#e2e8f0" stroke-width="3.4" filter="url(#mirrorSheen)" />
          <path d="{loPathD}" fill="none" stroke="#e2e8f0" stroke-width="3.4" filter="url(#mirrorSheen)" />
          <line x1="{inTopX:.1f}" y1="{inTopY:.1f}" x2="{inBotX:.1f}" y2="{inBotY:.1f}" stroke="#38bdf8" stroke-width="2.5" stroke-dasharray="4,3" />
          <text x="{inTopX - 12:.1f}" y="{inBotY + 22:.1f}" fill="#7dd3fc" font-size="12.2" text-anchor="middle">{labels['inWin']}</text>
          <line x1="62" y1="{srcY:.1f}" x2="{srcX - 4:.1f}" y2="{srcY:.1f}" stroke="#38bdf8" stroke-width="3.2" stroke-dasharray="10,4">
            <animate attributeName="stroke-dashoffset" from="28" to="0" dur="0.7s" repeatCount="indefinite" />
          </line>
          <circle cx="{srcX:.1f}" cy="{srcY:.1f}" r="6.5" fill="#fde047" filter="url(#photonGlow)">
            <animate attributeName="r" values="5.2;8.5;5.2" dur="1.2s" repeatCount="indefinite" />
          </circle>
          <text x="108" y="{srcY - 12:.1f}" fill="#fde047" font-size="12.2" font-weight="600" text-anchor="middle">{labels['srcLabel']}</text>
          {"".join(rayPathsSvg)}
          {bulgeCallout}
          <rect x="{pmtX:.1f}" y="{pmtTop:.1f}" width="188" height="{pmtHeight:.1f}" rx="12" fill="#1e293b" stroke="#475569" stroke-width="2" />
          <rect x="{pmtX + 2:.1f}" y="{outTopY - 12:.1f}" width="14" height="{outBotY - outTopY + 24:.1f}" rx="4" fill="#0ea5e9">
            <animate attributeName="fill-opacity" values="0.45;0.98;0.45" dur="1.3s" repeatCount="indefinite" />
          </rect>
          <path d="M {pmtX + 32:.1f},{centerY - 38:.1f} Q {pmtX + 48:.1f},{centerY - 18:.1f} {pmtX + 64:.1f},{centerY + 26:.1f}
                   M {pmtX + 68:.1f},{centerY - 42:.1f} Q {pmtX + 86:.1f},{centerY - 18:.1f} {pmtX + 102:.1f},{centerY + 28:.1f}
                   M {pmtX + 108:.1f},{centerY - 38:.1f} Q {pmtX + 124:.1f},{centerY - 14:.1f} {pmtX + 140:.1f},{centerY + 26:.1f}"
                fill="none" stroke="#94a3b8" stroke-width="2.4" stroke-linecap="round" />
          <line x1="{pmtX + 20:.1f}" y1="{centerY:.1f}" x2="{pmtX + 152:.1f}" y2="{centerY:.1f}" stroke="#38bdf8" stroke-width="1.6" stroke-dasharray="4,4">
            <animate attributeName="stroke-dashoffset" from="16" to="0" dur="0.55s" repeatCount="indefinite" />
          </line>
          <rect x="{pmtX + 154:.1f}" y="{centerY - 34:.1f}" width="12" height="68" rx="3" fill="#f59e0b" />
          <text x="{pmtX + 95:.1f}" y="{pmtTop + 25:.1f}" fill="#e2e8f0" font-size="13.2" font-weight="700" text-anchor="middle">{labels['pmtTitle']}</text>
          <text x="{pmtX + 95:.1f}" y="{pmtTop + pmtHeight - 14:.1f}" fill="#38bdf8" font-size="11.6" text-anchor="middle">{labels['pmtSub']}</text>
          <rect x="{oscX - 16:.1f}" y="{centerY - 102:.1f}" width="290" height="204" rx="12" fill="#020617" stroke="#334155" stroke-width="1.6" />
          <text x="{oscX:.1f}" y="{centerY - 76:.1f}" fill="#cbd5e1" font-size="13" font-weight="700">{labels['oscTitle']}</text>
          <line x1="{oscX:.1f}" y1="{oscY - 50:.1f}" x2="{oscX + 255:.1f}" y2="{oscY - 50:.1f}" stroke="#1e293b" stroke-width="1.2" />
          <line x1="{oscX:.1f}" y1="{oscY:.1f}" x2="{oscX + 255:.1f}" y2="{oscY:.1f}" stroke="#1e293b" stroke-width="1.2" />
          <line x1="{oscX:.1f}" y1="{oscY + 50:.1f}" x2="{oscX + 255:.1f}" y2="{oscY + 50:.1f}" stroke="#1e293b" stroke-width="1.2" />
          <path d="{oscWaveD}" fill="none" stroke="{waveColor}" stroke-width="3.0" filter="url(#photonGlow)">
            <animate attributeName="stroke-opacity" values="0.72;1.0;0.72" dur="1.3s" repeatCount="indefinite" />
          </path>
          <text x="{oscX + 128:.1f}" y="{centerY + 84:.1f}" fill="{waveColor}" font-size="12.2" font-weight="700" text-anchor="middle">{pmtMvText}</text>
        </g>
        """

    upperGroup = renderChannelGroup(
        channelId="idealChannel",
        titleText=labels["chATitle"],
        subtitleText=labels["chASub"],
        centerY=305.0,
        hasBulge=False,
        pmtMvText=labels["chAMv"],
        pulseScale=1.0,
    )
    lowerGroup = renderChannelGroup(
        channelId="deformedChannel",
        titleText=labels["chBTitle"],
        subtitleText=labels["chBSub"],
        centerY=705.0,
        hasBulge=True,
        pmtMvText=labels["chBMv"],
        pulseScale=0.74,
    )

    c2 = int(labels["legCol2X"])
    l2 = int(labels["legLine2X"])
    svgContent = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" width="100%" height="100%" style="background:#020617;font-family:{labels['fontStack']};">
  <defs>
    <linearGradient id="coneInteriorGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0f172a" stop-opacity="0.95" />
      <stop offset="55%" stop-color="#1e293b" stop-opacity="0.70" />
      <stop offset="100%" stop-color="#0c4a6e" stop-opacity="0.55" />
    </linearGradient>
    <filter id="photonGlow" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="2.8" result="blur" />
      <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
    </filter>
    <filter id="mirrorSheen" x="-10%" y="-20%" width="120%" height="140%">
      <feDropShadow dx="0" dy="0" stdDeviation="2.2" flood-color="#38bdf8" flood-opacity="0.45" />
    </filter>
  </defs>
  <rect x="40" y="24" width="1520" height="84" rx="14" fill="#0f172a" stroke="#1e293b" stroke-width="1.6" />
  <text x="68" y="57" fill="#f8fafc" font-size="{labels['hdrTitleSize']}" font-weight="800">{labels['hdrTitle']}</text>
  <text x="68" y="87" fill="#94a3b8" font-size="13.2">{labels['hdrSub']}</text>
  <g transform="translate({labels['legX']}, 44)">
    <circle cx="12" cy="10" r="5" fill="#67e8f9" />
    <text x="24" y="15" fill="#e2e8f0" font-size="12.5">{labels['legGuided']}</text>
    <circle cx="{c2}" cy="10" r="5" fill="#fda4af" />
    <text x="{c2 + 12}" y="15" fill="#fda4af" font-size="12.5">{labels['legEscape']}</text>
    <line x1="8" y1="36" x2="36" y2="36" stroke="#e2e8f0" stroke-width="3.2" />
    <text x="44" y="40" fill="#cbd5e1" font-size="12.5">{labels['legMirror']}</text>
    <line x1="{l2}" y1="36" x2="{l2 + 28}" y2="36" stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="5,4" />
    <text x="{l2 + 36}" y="40" fill="#94a3b8" font-size="12.5">{labels['legStep']}</text>
  </g>
  {upperGroup}
  {lowerGroup}
  <g transform="translate(170, 912)">
    <line x1="0" y1="0" x2="840" y2="0" stroke="#475569" stroke-width="1.5" />
    <line x1="0" y1="-6" x2="0" y2="6" stroke="#94a3b8" stroke-width="1.8" />
    <text x="0" y="22" fill="#94a3b8" font-size="12.5" text-anchor="middle">{labels['z0']}</text>
    <line x1="285.6" y1="-6" x2="285.6" y2="6" stroke="#fb7185" stroke-width="1.8" />
    <text x="285.6" y="22" fill="#fda4af" font-size="12.5" text-anchor="middle">{labels['z34']}</text>
    <line x1="571.2" y1="-6" x2="571.2" y2="6" stroke="#94a3b8" stroke-width="1.8" />
    <text x="571.2" y="22" fill="#94a3b8" font-size="12.5" text-anchor="middle">{labels['z68']}</text>
    <line x1="840" y1="-6" x2="840" y2="6" stroke="#38bdf8" stroke-width="1.8" />
    <text x="840" y="22" fill="#7dd3fc" font-size="12.5" text-anchor="middle">{labels['z100']}</text>
  </g>
</svg>
"""
    outputPath.write_text(svgContent, encoding="utf-8")
    logger.info("Generated animated photon propagation SVG (%s): %s (%.1f KB)", lang, outputPath, outputPath.stat().st_size / 1024.0)
