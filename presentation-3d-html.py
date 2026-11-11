"""
Interactive 3D light-guide photon propagation and PMT readout simulator generator.

Builds a zero-CDN-dependency offline HTML5 3D optical workbench embedding the
reconstructed freeform inner surface mesh and 100x100 scan matrices. Supports
360-degree camera orbit, wheel zoom, real-time client-side 3D multi-bounce
specular ray tracing, comet wavepackets, PMT photocathode flash, oscilloscope
waveform, and interactive/automated 2D laser scan control in Chinese and English.

Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__author__ = "Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>"


def getSimulatorUiTexts(lang: str) -> dict[str, str]:
    """
    Return localized static and dynamic UI text dictionary for 'zh' or 'en'.
    """
    if lang == "en":
        return {
            "htmlLang": "en",
            "fontStack": '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
            "pageTitle": "LHCb PicoCal 3D Conical Light-Guide Photon Transport & PMT Readout Simulator",
            "hdrTitle": "LHCb PicoCal 3D Light-Guide Specular Photon Transport & PMT Readout Bench",
            "hdrSub": "Left-drag 360° orbit | Wheel zoom | Click 2D scan pad to move laser spot · Author: Hengne Li · Hengne.Li@cern.ch",
            "pillRmse": "Inner Fit RMSE",
            "pillCorr": "Full-Map Pearson r",
            "pillBump": "Non-Axi Texture r",
            "hudModeLbl": "Surface Mode: ",
            "hudLaserLbl": "Laser Local Pos: ",
            "hudEffLbl": "Collection / Bounces: ",
            "legGuided": "Guided → PMT Photocathode",
            "legEscape": "Turnaround Escaped Ray",
            "legBulge": "Inward Bulge (ΔR < 0)",
            "legDent": "Outward Dent (ΔR > 0)",
            "legReset": "Double-click 3D view to reset camera | Hengne Li · Hengne.Li@cern.ch",
            "c1Title": "1. Entrance 2D Laser Scan Pad (Ø17.5 mm)",
            "btnSim": "Sim Map",
            "btnMeas": "Meas Map",
            "btnAuto": "Auto Scan",
            "roXLbl": "Local X': ",
            "roYLbl": "Local Y': ",
            "roRLbl": "Radius r: ",
            "roMeasLbl": "Measured Net: ",
            "roSimLbl": "Simulated Net: ",
            "scanHint": "Click or drag inside aperture to move diffuse laser spot",
            "c2Title": "2. Dynamic PMT Anode Pulse Oscilloscope",
            "pmtOnline": "Photocathode Live",
            "c3Title": "3. Surface Geometry & Photon Stream Controls",
            "btnDef": "Reconstructed Surface",
            "btnIdeal": "Ideal STEP Cone",
            "exagTitle": "3D Surface Deformation Exaggeration",
            "exagInit": "8× (True ±187 μm)",
            "speedTitle": "Photon Wavepacket Flight Speed",
            "speedInit": "1.0× Standard Speed",
            "countTitle": "Concurrent Tracked Photon Packets",
            "countInit": "42 Packets",
            "surfDeformed": "Reconstructed Imperfect Inner Surface",
            "surfIdeal": "Ideal STEP Conical Baseline",
            "effPrefix": "Guided ",
            "effMid": "% | Avg Bounces ",
            "effSuffix": "",
            "exagSuffix": "× (True ±187 μm)",
            "speedSuffix": "× Flight Speed",
            "countSuffix": " Packets",
            "oscPrefix": "PMT Anode Negative Pulse: -",
            "lblIn3d": "Entrance z=0 (Ø17.5mm)",
            "lblOut3d": "PMT Readout End (z=100mm)",
        }
    return {
        "htmlLang": "zh-CN",
        "fontStack": '-apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif',
        "pageTitle": "LHCb PicoCal 锥形光导三维光子传播与 PMT 接收实时交互模拟台",
        "hdrTitle": "LHCb PicoCal 光导三维光子镜面反射与 PMT 读出交互式模拟实验台",
        "hdrSub": "左键拖拽 360° 旋转 | 滚轮无级缩放 | 点击右侧扫描图移动激光点 · Hengne Li (李衡讷) · Hengne.Li@cern.ch",
        "pillRmse": "圆内拟合 RMSE",
        "pillCorr": "全图相关系数",
        "pillBump": "凹凸纹理相关",
        "hudModeLbl": "当前表面模式：",
        "hudLaserLbl": "激光入射局部坐标：",
        "hudEffLbl": "实时收集效率 / 平均反射跳数：",
        "legGuided": "传导至 PMT 光阴极光子",
        "legEscape": "锥壁/鼓包偏折返逃逸光子",
        "legBulge": "内壁向内鼓包 (ΔR < 0)",
        "legDent": "内壁向外凹陷 (ΔR > 0)",
        "legReset": "双击三维视图复位相机视角 | Hengne Li · Hengne.Li@cern.ch",
        "c1Title": "1. 入射口二维激光扫描交互板 (Ø17.5 mm)",
        "btnSim": "模拟图",
        "btnMeas": "实测图",
        "btnAuto": "自动光栅扫描",
        "roXLbl": "局部坐标 X': ",
        "roYLbl": "局部坐标 Y': ",
        "roRLbl": "极径 r: ",
        "roMeasLbl": "实测净响应: ",
        "roSimLbl": "模拟净响应: ",
        "scanHint": "点击或拖动圆内任意点切换激光漫散射源位置",
        "c2Title": "2. PMT 光电倍增管阳极动态脉冲示波器",
        "pmtOnline": "光电阴极在线",
        "c3Title": "3. 几何表面与光子脉冲流控制",
        "btnDef": "重构不完美表面",
        "btnIdeal": "STEP 理想圆锥",
        "exagTitle": "内壁微米形变三维视觉放大倍率",
        "exagInit": "8× (真实形变 ±187 μm)",
        "speedTitle": "光子脉冲飞行速度",
        "speedInit": "1.0× 标准教学速度",
        "countTitle": "并发动态追踪光子波包数",
        "countInit": "42 束",
        "surfDeformed": "反演重构不完美内表面",
        "surfIdeal": "STEP 理想完美圆锥基准",
        "effPrefix": "动态光子到达率 ",
        "effMid": "% | 平均反射 ",
        "effSuffix": " 次",
        "exagSuffix": "× (真实形变 ±187 μm)",
        "speedSuffix": "× 脉冲飞行倍率",
        "countSuffix": " 束",
        "oscPrefix": "PMT 阳极负脉冲幅度: -",
        "lblIn3d": "入射口 z=0 (Ø17.5mm)",
        "lblOut3d": "PMT 光电倍增管读出端 (z=100mm)",
    }


SIMULATOR_JS_TEMPLATE: str = """(function() {
  "use strict";
  const DATA = __JSON_PAYLOAD__;
  const UI = DATA.ui;

  document.getElementById("hdrRmse").textContent = DATA.metrics.rmsePct.toFixed(2) + "% (" + DATA.metrics.rmseMv.toFixed(2) + " mV)";
  document.getElementById("hdrCorr").textContent = "r = " + DATA.metrics.totalCorr.toFixed(4);
  document.getElementById("hdrBumpCorr").textContent = "r = " + DATA.metrics.nonAxiCorr.toFixed(4);

  const state = {
    useDeformed: true, showSimMap: true, autoScan: false, autoScanPhase: 0.0,
    laserX: -1.20, laserY: 1.60, exagFactor: 8.0, photonSpeed: 1.0, targetPhotonCount: 42,
    yaw: -0.62, pitch: 0.30, zoom: 1.05, pmtFlashEnergy: 0.0, currentMv: 25.5,
    collectedRatio: 0.21, avgBounces: 2.8
  };

  // Periodic bilinear interpolation of reconstructed inner surface radius and partial derivatives
  function evalSurface(thetaRad, zMm, visualExag) {
    const zClamped = Math.max(0.0, Math.min(DATA.coneLenMm, zMm));
    if (!state.useDeformed) {
      const slope = (DATA.rOutIdealMm - DATA.rInIdealMm) / DATA.coneLenMm;
      return { r: DATA.rInIdealMm + slope * zClamped, drDth: 0.0, drDz: slope, deltaUm: 0.0 };
    }
    let normTh = (thetaRad + Math.PI) / (2.0 * Math.PI);
    normTh = normTh - Math.floor(normTh);
    const fTh = normTh * DATA.nTheta;
    const i0 = Math.floor(fTh) % DATA.nTheta, i1 = (i0 + 1) % DATA.nTheta, wTh = fTh - Math.floor(fTh);
    const fZ = (zClamped / DATA.coneLenMm) * (DATA.nZ - 1);
    const j0 = Math.min(DATA.nZ - 2, Math.max(0, Math.floor(fZ))), j1 = j0 + 1, wZ = fZ - j0;

    const r00 = DATA.radiusGridMm[i0][j0], r10 = DATA.radiusGridMm[i1][j0];
    const r01 = DATA.radiusGridMm[i0][j1], r11 = DATA.radiusGridMm[i1][j1];
    const rReal = (1 - wTh) * ((1 - wZ) * r00 + wZ * r01) + wTh * ((1 - wZ) * r10 + wZ * r11);

    const d00 = DATA.deltaUmGrid[i0][j0], d10 = DATA.deltaUmGrid[i1][j0];
    const d01 = DATA.deltaUmGrid[i0][j1], d11 = DATA.deltaUmGrid[i1][j1];
    const deltaUm = (1 - wTh) * ((1 - wZ) * d00 + wZ * d01) + wTh * ((1 - wZ) * d10 + wZ * d11);

    const extraMm = (visualExag - 1.0) * (deltaUm * 0.001);
    const rTotal = Math.max(1.8, rReal + extraMm);
    const dThStep = (2.0 * Math.PI) / DATA.nTheta, dZStep = DATA.coneLenMm / (DATA.nZ - 1);
    const drRealDth = (((1 - wZ) * (r10 - r00) + wZ * (r11 - r01))) / dThStep;
    const drRealDz = (((1 - wTh) * (r01 - r00) + wTh * (r11 - r10))) / dZStep;
    const ddUmDth = (((1 - wZ) * (d10 - d00) + wZ * (d11 - d01))) / dThStep;
    const ddUmDz = (((1 - wTh) * (d01 - d00) + wTh * (d11 - d10))) / dZStep;
    return {
      r: rTotal,
      drDth: drRealDth + (visualExag - 1.0) * 0.001 * ddUmDth,
      drDz: drRealDz + (visualExag - 1.0) * 0.001 * ddUmDz,
      deltaUm: deltaUm
    };
  }

  // Exact client-side 3D multi-bounce specular ray tracer
  function traceSinglePhoton3d(srcX, srcY) {
    const phi = 2.0 * Math.PI * Math.random();
    const cosTheta = Math.pow(1.0 - 0.55 * Math.random(), 0.5);
    const sinTheta = Math.sqrt(Math.max(0.0, 1.0 - cosTheta * cosTheta));
    let x = srcX, y = srcY, z = 0.0;
    let dx = sinTheta * Math.cos(phi), dy = sinTheta * Math.sin(phi), dz = cosTheta;

    const physExag = state.useDeformed ? Math.min(3.5, 1.0 + 0.18 * state.exagFactor) : 1.0;
    const visExag = state.useDeformed ? state.exagFactor : 1.0;
    const slopeBase = (DATA.rOutIdealMm - DATA.rInIdealMm) / DATA.coneLenMm;
    const waypoints = [{ x: x, y: y, z: z }];
    let bounces = 0, reachedPmt = false;

    for (let b = 0; b < 16; b++) {
      const tzOut = dz > 1e-5 ? (DATA.coneLenMm - z) / dz : 1e9;
      const tzIn = (dz < -1e-5 && b > 0) ? -z / dz : 1e9;
      let tEst = 1e9;
      if (b === 0) {
        const s0 = evalSurface(Math.atan2(y, x), z, physExag);
        const qa = dx * dx + dy * dy - slopeBase * slopeBase * dz * dz;
        const qb = 2.0 * (x * dx + y * dy - s0.r * slopeBase * dz);
        const qc = x * x + y * y - s0.r * s0.r;
        const disc = Math.max(0.0, qb * qb - 4.0 * qa * qc);
        tEst = (-qb + Math.sqrt(disc)) / (2.0 * (Math.abs(qa) > 1e-6 ? qa : 1.0));
      } else {
        const rCurr = Math.hypot(x, y) + 1e-9;
        const qa = dx * dx + dy * dy - slopeBase * slopeBase * dz * dz;
        const qb = 2.0 * (x * dx + y * dy - rCurr * slopeBase * dz);
        tEst = -qb / (Math.abs(qa) > 1e-6 ? qa : 1.0);
      }

      let tVal = Math.max(0.06, Math.min(140.0, tEst));
      for (let it = 0; it < 4; it++) {
        const xt = x + tVal * dx, yt = y + tVal * dy;
        const zt = Math.max(0.0, Math.min(DATA.coneLenMm, z + tVal * dz));
        const rt = Math.hypot(xt, yt) + 1e-9;
        const surf = evalSurface(Math.atan2(yt, xt), zt, physExag);
        const fVal = rt - surf.r;
        const dfDt = (xt * dx + yt * dy) / rt - surf.drDth * ((xt * dy - yt * dx) / (rt * rt)) - surf.drDz * dz;
        if (Math.abs(dfDt) > 1e-5) tVal = Math.max(0.005, Math.min(150.0, tVal - fVal / dfDt));
      }

      const zWall = z + tVal * dz, xWall = x + tVal * dx, yWall = y + tVal * dy;
      const checkSurf = evalSurface(Math.atan2(yWall, xWall), Math.max(0.0, Math.min(DATA.coneLenMm, zWall)), physExag);
      const validWall = (tVal > 0.01) && (zWall > 0.0) && (zWall < DATA.coneLenMm) && (Math.abs(Math.hypot(xWall, yWall) - checkSurf.r) < 0.12);
      const tWall = validWall ? tVal : 1e9;

      if (tzOut <= tWall && tzOut <= tzIn && tzOut < 1e8) {
        const xOut = x + tzOut * dx, yOut = y + tzOut * dy;
        const endSurf = evalSurface(Math.atan2(yOut, xOut), DATA.coneLenMm, physExag);
        if (xOut * xOut + yOut * yOut <= (endSurf.r + 0.15) * (endSurf.r + 0.15)) {
          waypoints.push({ x: xOut, y: yOut, z: DATA.coneLenMm });
          waypoints.push({ x: xOut + 4.5 * dx, y: yOut + 4.5 * dy, z: 104.5 });
          reachedPmt = true;
        }
        break;
      }
      if (tzIn <= tWall && tzIn < 1e8) {
        waypoints.push({ x: x + tzIn * dx, y: y + tzIn * dy, z: 0.0 });
        waypoints.push({ x: x + (tzIn + 9.0) * dx, y: y + (tzIn + 9.0) * dy, z: -9.0 * Math.abs(dz) });
        break;
      }
      if (tWall >= 1e8) break;

      x = xWall; y = yWall; z = zWall; bounces++;
      const thHit = Math.atan2(y, x);
      const visSurf = evalSurface(thHit, z, visExag);
      waypoints.push({ x: visSurf.r * Math.cos(thHit), y: visSurf.r * Math.sin(thHit), z: z });

      const hitSurf = evalSurface(thHit, z, physExag);
      const rHit = Math.hypot(x, y) + 1e-9;
      const gx = x / rHit - hitSurf.drDth * (-y / (rHit * rHit));
      const gy = y / rHit - hitSurf.drDth * (x / (rHit * rHit));
      const gz = -hitSurf.drDz;
      const gn = Math.hypot(gx, gy, gz) + 1e-12;
      const nx = -gx / gn, ny = -gy / gn, nz = -gz / gn;
      const dot = dx * nx + dy * ny + dz * nz;
      dx -= 2.0 * dot * nx; dy -= 2.0 * dot * ny; dz -= 2.0 * dot * nz;
    }

    let totalLen = 0.0;
    const segLens = [];
    for (let i = 0; i < waypoints.length - 1; i++) {
      const d = Math.hypot(waypoints[i + 1].x - waypoints[i].x, waypoints[i + 1].y - waypoints[i].y, waypoints[i + 1].z - waypoints[i].z);
      segLens.push(d); totalLen += d;
    }
    return { waypoints: waypoints, segLens: segLens, totalLen: Math.max(1.0, totalLen), dist: Math.random() * totalLen, reachedPmt: reachedPmt, bounces: bounces };
  }

  const photons = [];
  function rebuildPhotonPool() {
    photons.length = 0;
    let hitCnt = 0, bounceSum = 0;
    for (let i = 0; i < state.targetPhotonCount; i++) {
      const p = traceSinglePhoton3d(state.laserX, state.laserY);
      photons.push(p);
      if (p.reachedPmt) hitCnt++;
      bounceSum += p.bounces;
    }
    state.collectedRatio = hitCnt / Math.max(1, state.targetPhotonCount);
    state.avgBounces = bounceSum / Math.max(1, state.targetPhotonCount);
    updateReadouts();
  }

  const scanCanvas = document.getElementById("scanCanvas");
  const scanCtx = scanCanvas.getContext("2d");

  function heatColorRgb(norm) {
    const t = Math.max(0.0, Math.min(1.0, norm));
    if (t < 0.33) { const u = t / 0.33; return [Math.round(15 + 15 * u), Math.round(23 + 65 * u), Math.round(42 + 145 * u)]; }
    if (t < 0.66) { const u = (t - 0.33) / 0.33; return [Math.round(30 - 10 * u), Math.round(88 + 95 * u), Math.round(187 - 45 * u)]; }
    const u = (t - 0.66) / 0.34;
    return [Math.round(20 + 233 * u), Math.round(183 + 41 * u), Math.round(142 - 71 * u)];
  }

  function lookupScanValues(localX, localY) {
    const ix = Math.max(0, Math.min(99, Math.round(((localX + DATA.centerXMm) - DATA.xMinMm) / DATA.gridStepMm)));
    const iy = Math.max(0, Math.min(99, Math.round(((localY + DATA.centerYMm) - DATA.yMinMm) / DATA.gridStepMm)));
    return { meas: DATA.measMv100[iy][ix], sim: DATA.simMv100[iy][ix] };
  }

  function drawScanPad() {
    const img = scanCtx.createImageData(200, 200);
    const grid = state.showSimMap ? DATA.simMv100 : DATA.measMv100;
    for (let py = 0; py < 200; py++) {
      const iy = 99 - Math.floor(py / 2);
      for (let px = 0; px < 200; px++) {
        const rgb = heatColorRgb(grid[iy][Math.floor(px / 2)] / 29.5);
        const k = (py * 200 + px) * 4;
        img.data[k] = rgb[0]; img.data[k + 1] = rgb[1]; img.data[k + 2] = rgb[2]; img.data[k + 3] = 255;
      }
    }
    scanCtx.putImageData(img, 0, 0);
    const cxPx = ((DATA.centerXMm - DATA.xMinMm) / 19.8) * 200;
    const cyPx = (1.0 - (DATA.centerYMm - DATA.yMinMm) / 19.8) * 200;
    scanCtx.strokeStyle = "rgba(248, 250, 252, 0.72)"; scanCtx.lineWidth = 1.3; scanCtx.setLineDash([4, 3]);
    scanCtx.beginPath(); scanCtx.arc(cxPx, cyPx, (8.44 / 19.8) * 200, 0, Math.PI * 2); scanCtx.stroke(); scanCtx.setLineDash([]);
    const lxPx = (((state.laserX + DATA.centerXMm) - DATA.xMinMm) / 19.8) * 200;
    const lyPx = (1.0 - ((state.laserY + DATA.centerYMm) - DATA.yMinMm) / 19.8) * 200;
    scanCtx.strokeStyle = "#fde047"; scanCtx.lineWidth = 2.0;
    scanCtx.beginPath(); scanCtx.arc(lxPx, lyPx, 5.5, 0, Math.PI * 2); scanCtx.stroke();
  }

  function updateReadouts() {
    const r = Math.hypot(state.laserX, state.laserY);
    const vals = lookupScanValues(state.laserX, state.laserY);
    state.currentMv = state.useDeformed ? vals.sim : 26.8 * Math.exp(-0.0022 * r * r);
    const sx = (state.laserX >= 0 ? "+" : "") + state.laserX.toFixed(2);
    const sy = (state.laserY >= 0 ? "+" : "") + state.laserY.toFixed(2);
    document.getElementById("roX").textContent = sx + " mm";
    document.getElementById("roY").textContent = sy + " mm";
    document.getElementById("roR").textContent = r.toFixed(2) + " mm";
    document.getElementById("roMeas").textContent = vals.meas.toFixed(2) + " mV";
    document.getElementById("roSim").textContent = vals.sim.toFixed(2) + " mV";
    document.getElementById("hudLaserXY").textContent = "X' = " + sx + " mm, Y' = " + sy + " mm";
    document.getElementById("hudEffStats").textContent = UI.effPrefix + (state.collectedRatio * 100).toFixed(0) + UI.effMid + state.avgBounces.toFixed(1) + UI.effSuffix;
    drawScanPad();
  }

  function handleScanPointer(ev) {
    const rect = scanCanvas.getBoundingClientRect();
    let lx = (DATA.xMinMm + ((ev.clientX - rect.left) / rect.width) * 19.8) - DATA.centerXMm;
    let ly = (DATA.yMinMm + (1.0 - (ev.clientY - rect.top) / rect.height) * 19.8) - DATA.centerYMm;
    const rad = Math.hypot(lx, ly);
    if (rad > 7.95) { lx *= 7.95 / rad; ly *= 7.95 / rad; }
    state.laserX = lx; state.laserY = ly;
    rebuildPhotonPool();
  }

  let scanDrag = false;
  scanCanvas.addEventListener("mousedown", function(e) { state.autoScan = false; document.getElementById("btnAutoScan").classList.remove("active"); scanDrag = true; handleScanPointer(e); });
  window.addEventListener("mousemove", function(e) { if (scanDrag) handleScanPointer(e); });
  window.addEventListener("mouseup", function() { scanDrag = false; });

  document.getElementById("btnMapSim").addEventListener("click", function() { state.showSimMap = true; this.classList.add("active"); document.getElementById("btnMapMeas").classList.remove("active"); drawScanPad(); });
  document.getElementById("btnMapMeas").addEventListener("click", function() { state.showSimMap = false; this.classList.add("active"); document.getElementById("btnMapSim").classList.remove("active"); drawScanPad(); });
  document.getElementById("btnAutoScan").addEventListener("click", function() { state.autoScan = !state.autoScan; this.classList.toggle("active", state.autoScan); });
  document.getElementById("btnDeformed").addEventListener("click", function() { state.useDeformed = true; this.classList.add("active"); document.getElementById("btnIdeal").classList.remove("active"); document.getElementById("hudSurfMode").textContent = UI.surfDeformed; rebuildPhotonPool(); });
  document.getElementById("btnIdeal").addEventListener("click", function() { state.useDeformed = false; this.classList.add("active"); document.getElementById("btnDeformed").classList.remove("active"); document.getElementById("hudSurfMode").textContent = UI.surfIdeal; rebuildPhotonPool(); });
  document.getElementById("rngExag").addEventListener("input", function() { state.exagFactor = parseFloat(this.value); document.getElementById("lblExag").textContent = state.exagFactor.toFixed(0) + UI.exagSuffix; rebuildPhotonPool(); });
  document.getElementById("rngSpeed").addEventListener("input", function() { state.photonSpeed = parseFloat(this.value); document.getElementById("lblSpeed").textContent = state.photonSpeed.toFixed(1) + UI.speedSuffix; });
  document.getElementById("rngCount").addEventListener("input", function() { state.targetPhotonCount = parseInt(this.value, 10); document.getElementById("lblCount").textContent = state.targetPhotonCount + UI.countSuffix; rebuildPhotonPool(); });

  const vpCanvas = document.getElementById("viewport3d");
  const vpCtx = vpCanvas.getContext("2d");
  let camDrag = false, lastMx = 0, lastMy = 0;
  vpCanvas.addEventListener("mousedown", function(e) { camDrag = true; lastMx = e.clientX; lastMy = e.clientY; });
  window.addEventListener("mousemove", function(e) {
    if (!camDrag) return;
    state.yaw += (e.clientX - lastMx) * 0.0075;
    state.pitch = Math.max(-1.25, Math.min(1.25, state.pitch + (e.clientY - lastMy) * 0.0075));
    lastMx = e.clientX; lastMy = e.clientY;
  });
  window.addEventListener("mouseup", function() { camDrag = false; });
  vpCanvas.addEventListener("wheel", function(e) { e.preventDefault(); state.zoom = Math.max(0.55, Math.min(2.6, state.zoom * (e.deltaY < 0 ? 1.09 : 0.92))); }, { passive: false });
  vpCanvas.addEventListener("dblclick", function() { state.yaw = -0.62; state.pitch = 0.30; state.zoom = 1.05; });

  function project3d(xMm, yMm, zMm, width, height) {
    const px0 = zMm - 52.0, py0 = yMm * 2.25, pz0 = xMm * 2.25;
    const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
    const x1 = px0 * cy + pz0 * sy, z1 = -px0 * sy + pz0 * cy;
    const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
    const y2 = py0 * cp - z1 * sp, z2 = py0 * sp + z1 * cp;
    const depth = Math.max(30.0, 215.0 - z2);
    const fov = Math.min(width, height) * 1.48 * state.zoom;
    return { sx: width * 0.50 + (x1 * fov) / depth, sy: height * 0.52 - (y2 * fov) / depth, depth: depth };
  }

  function deltaColorRgba(deltaUm, alpha) {
    if (!state.useDeformed) return "rgba(148, 163, 184, " + alpha + ")";
    const t = Math.max(-1.0, Math.min(1.0, deltaUm / 165.0));
    if (t < 0) { const u = -t; return "rgba(" + Math.round(148 * (1 - u) + 37 * u) + "," + Math.round(163 * (1 - u) + 115 * u) + "," + Math.round(184 * (1 - u) + 245 * u) + "," + alpha + ")"; }
    return "rgba(" + Math.round(148 * (1 - t) + 244 * t) + "," + Math.round(163 * (1 - t) + 63 * t) + "," + Math.round(184 * (1 - t) + 94 * t) + "," + alpha + ")";
  }

  const oscCanvas = document.getElementById("oscCanvas");
  const oscCtx = oscCanvas.getContext("2d");
  function drawOscilloscope(timeSec) {
    const w = oscCanvas.width, h = oscCanvas.height;
    oscCtx.clearRect(0, 0, w, h);
    oscCtx.strokeStyle = "#1e293b"; oscCtx.lineWidth = 1;
    for (let g = 1; g <= 3; g++) { const gy = (h / 4) * g; oscCtx.beginPath(); oscCtx.moveTo(0, gy); oscCtx.lineTo(w, gy); oscCtx.stroke(); }
    const ampRatio = Math.max(0.15, Math.min(1.12, state.currentMv / 28.5));
    oscCtx.strokeStyle = state.useDeformed ? "#34d399" : "#38bdf8"; oscCtx.lineWidth = 2.4; oscCtx.beginPath();
    for (let px = 0; px < w; px++) {
      const tNorm = px / w;
      let pulse = 0.0;
      if (tNorm > 0.18) { const dt = (tNorm - 0.18) * 11.5; pulse = ampRatio * (dt * dt) * Math.exp(-dt * 1.45) * (1.0 + 0.04 * Math.sin(timeSec * 14.0)); }
      const py = 20.0 + pulse * (h - 36.0) * 1.32;
      if (px === 0) oscCtx.moveTo(px, py); else oscCtx.lineTo(px, py);
    }
    oscCtx.stroke();
    oscCtx.fillStyle = "#e2e8f0"; oscCtx.font = "600 12px -apple-system, sans-serif";
    oscCtx.fillText(UI.oscPrefix + state.currentMv.toFixed(2) + " mV", 12, h - 10);
  }

  function samplePointAlongPhoton(p, distVal) {
    let rem = Math.max(0.0, Math.min(p.totalLen, distVal));
    for (let i = 0; i < p.segLens.length; i++) {
      if (rem <= p.segLens[i] || i === p.segLens.length - 1) {
        const u = p.segLens[i] > 1e-6 ? rem / p.segLens[i] : 0.0;
        const a = p.waypoints[i], b = p.waypoints[i + 1];
        return { x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u, z: a.z + (b.z - a.z) * u };
      }
      rem -= p.segLens[i];
    }
    return p.waypoints[0];
  }

  let lastTs = performance.now();
  function renderLoop(nowTs) {
    const dt = Math.min(0.05, (nowTs - lastTs) * 0.001);
    lastTs = nowTs;
    if (state.autoScan) {
      state.autoScanPhase += dt * 0.85;
      const rSweep = 6.8 * Math.abs(Math.sin(state.autoScanPhase * 0.42));
      state.laserX = rSweep * Math.cos(state.autoScanPhase * 1.9);
      state.laserY = rSweep * Math.sin(state.autoScanPhase * 1.9);
      if (Math.floor(nowTs / 90) !== Math.floor((nowTs - dt * 1000) / 90)) rebuildPhotonPool();
    }

    const dpr = window.devicePixelRatio || 1;
    const rect = vpCanvas.getBoundingClientRect();
    if (vpCanvas.width !== Math.round(rect.width * dpr) || vpCanvas.height !== Math.round(rect.height * dpr)) {
      vpCanvas.width = Math.round(rect.width * dpr); vpCanvas.height = Math.round(rect.height * dpr);
    }
    vpCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = rect.width, H = rect.height;
    vpCtx.clearRect(0, 0, W, H);

    const visExag = state.useDeformed ? state.exagFactor : 1.0;
    const nThMesh = 36, nZMesh = 24, quads = [];
    for (let i = 0; i < nThMesh; i++) {
      const th0 = -Math.PI + (2 * Math.PI * i) / nThMesh, th1 = -Math.PI + (2 * Math.PI * (i + 1)) / nThMesh;
      for (let j = 0; j < nZMesh; j++) {
        const z0 = (DATA.coneLenMm * j) / nZMesh, z1 = (DATA.coneLenMm * (j + 1)) / nZMesh;
        const s00 = evalSurface(th0, z0, visExag), s10 = evalSurface(th1, z0, visExag), s11 = evalSurface(th1, z1, visExag), s01 = evalSurface(th0, z1, visExag);
        const p00 = project3d(s00.r * Math.cos(th0), s00.r * Math.sin(th0), z0, W, H);
        const p10 = project3d(s10.r * Math.cos(th1), s10.r * Math.sin(th1), z0, W, H);
        const p11 = project3d(s11.r * Math.cos(th1), s11.r * Math.sin(th1), z1, W, H);
        const p01 = project3d(s01.r * Math.cos(th0), s01.r * Math.sin(th0), z1, W, H);
        quads.push({ p00: p00, p10: p10, p11: p11, p01: p01, depth: 0.25 * (p00.depth + p10.depth + p11.depth + p01.depth), delta: 0.25 * (s00.deltaUm + s10.deltaUm + s11.deltaUm + s01.deltaUm) });
      }
    }
    quads.sort(function(a, b) { return b.depth - a.depth; });

    const halfIdx = Math.floor(quads.length * 0.55);
    for (let k = 0; k < halfIdx; k++) {
      const q = quads[k];
      vpCtx.fillStyle = deltaColorRgba(q.delta, 0.20); vpCtx.strokeStyle = deltaColorRgba(q.delta, 0.34); vpCtx.lineWidth = 0.65;
      vpCtx.beginPath(); vpCtx.moveTo(q.p00.sx, q.p00.sy); vpCtx.lineTo(q.p10.sx, q.p10.sy); vpCtx.lineTo(q.p11.sx, q.p11.sy); vpCtx.lineTo(q.p01.sx, q.p01.sy); vpCtx.closePath(); vpCtx.fill(); vpCtx.stroke();
    }

    state.pmtFlashEnergy = Math.max(0.0, state.pmtFlashEnergy - dt * 3.8);
    for (const ringZ of [100.2, 105.0, 116.0]) {
      vpCtx.strokeStyle = ringZ < 106.0 ? "rgba(56, 189, 248, " + (0.45 + 0.45 * state.pmtFlashEnergy).toFixed(2) + ")" : "rgba(100, 116, 139, 0.55)";
      vpCtx.lineWidth = ringZ < 106.0 ? 2.4 : 1.5;
      vpCtx.beginPath();
      for (let m = 0; m <= 48; m++) {
        const ang = (2 * Math.PI * m) / 48;
        const pr = project3d(8.18 * Math.cos(ang), 8.18 * Math.sin(ang), ringZ, W, H);
        if (m === 0) vpCtx.moveTo(pr.sx, pr.sy); else vpCtx.lineTo(pr.sx, pr.sy);
      }
      vpCtx.stroke();
    }
    const pmtCenter = project3d(0, 0, 104.5, W, H), pmtEdge = project3d(0, 5.2, 104.5, W, H);
    const glowR = Math.max(10.0, Math.hypot(pmtEdge.sx - pmtCenter.sx, pmtEdge.sy - pmtCenter.sy));
    const gradPmt = vpCtx.createRadialGradient(pmtCenter.sx, pmtCenter.sy, 1, pmtCenter.sx, pmtCenter.sy, glowR * 1.4);
    gradPmt.addColorStop(0, "rgba(56, 189, 248, " + (0.35 + 0.55 * state.pmtFlashEnergy).toFixed(2) + ")");
    gradPmt.addColorStop(1, "rgba(14, 165, 233, 0.0)");
    vpCtx.fillStyle = gradPmt; vpCtx.beginPath(); vpCtx.arc(pmtCenter.sx, pmtCenter.sy, glowR * 1.4, 0, Math.PI * 2); vpCtx.fill();

    const laserExt = project3d(state.laserX, state.laserY, -16.0, W, H), laserSpot = project3d(state.laserX, state.laserY, 0.0, W, H);
    vpCtx.strokeStyle = "#38bdf8"; vpCtx.lineWidth = 3.0; vpCtx.beginPath(); vpCtx.moveTo(laserExt.sx, laserExt.sy); vpCtx.lineTo(laserSpot.sx, laserSpot.sy); vpCtx.stroke();
    vpCtx.fillStyle = "#fde047"; vpCtx.beginPath(); vpCtx.arc(laserSpot.sx, laserSpot.sy, 5.2, 0, Math.PI * 2); vpCtx.fill();

    const speedMmPerSec = 68.0 * state.photonSpeed;
    for (let idx = 0; idx < photons.length; idx++) {
      const ph = photons[idx];
      vpCtx.strokeStyle = ph.reachedPmt ? "rgba(34, 211, 238, 0.14)" : "rgba(251, 113, 133, 0.13)"; vpCtx.lineWidth = 1.0;
      vpCtx.beginPath();
      for (let w = 0; w < ph.waypoints.length; w++) {
        const pr = project3d(ph.waypoints[w].x, ph.waypoints[w].y, ph.waypoints[w].z, W, H);
        if (w === 0) vpCtx.moveTo(pr.sx, pr.sy); else vpCtx.lineTo(pr.sx, pr.sy);
      }
      vpCtx.stroke();

      ph.dist += speedMmPerSec * dt;
      if (ph.dist >= ph.totalLen) {
        if (ph.reachedPmt) state.pmtFlashEnergy = Math.min(1.0, state.pmtFlashEnergy + 0.22);
        const fresh = traceSinglePhoton3d(state.laserX, state.laserY);
        fresh.dist = 0.0; photons[idx] = fresh; continue;
      }

      vpCtx.strokeStyle = ph.reachedPmt ? "rgba(103, 232, 249, 0.88)" : "rgba(253, 164, 175, 0.82)"; vpCtx.lineWidth = 2.3;
      vpCtx.beginPath();
      for (let s = 0; s <= 6; s++) {
        const pt3 = samplePointAlongPhoton(ph, Math.max(0.0, ph.dist - 13.0 * (1.0 - s / 6)));
        const pr = project3d(pt3.x, pt3.y, pt3.z, W, H);
        if (s === 0) vpCtx.moveTo(pr.sx, pr.sy); else vpCtx.lineTo(pr.sx, pr.sy);
      }
      vpCtx.stroke();
      const head3 = samplePointAlongPhoton(ph, ph.dist), headPr = project3d(head3.x, head3.y, head3.z, W, H);
      vpCtx.fillStyle = ph.reachedPmt ? "#ecfeff" : "#ffe4e6";
      vpCtx.beginPath(); vpCtx.arc(headPr.sx, headPr.sy, 2.9, 0, Math.PI * 2); vpCtx.fill();
    }

    for (let k = halfIdx; k < quads.length; k++) {
      const q = quads[k];
      vpCtx.fillStyle = deltaColorRgba(q.delta, 0.09); vpCtx.strokeStyle = deltaColorRgba(q.delta, 0.24); vpCtx.lineWidth = 0.6;
      vpCtx.beginPath(); vpCtx.moveTo(q.p00.sx, q.p00.sy); vpCtx.lineTo(q.p10.sx, q.p10.sy); vpCtx.lineTo(q.p11.sx, q.p11.sy); vpCtx.lineTo(q.p01.sx, q.p01.sy); vpCtx.closePath(); vpCtx.fill(); vpCtx.stroke();
    }

    const lblIn = project3d(0, 10.4, 0, W, H), lblOut = project3d(0, 9.8, 106, W, H);
    vpCtx.fillStyle = "#7dd3fc"; vpCtx.font = "600 12.5px -apple-system, sans-serif";
    vpCtx.fillText(UI.lblIn3d, lblIn.sx - 55, lblIn.sy - 6);
    vpCtx.fillStyle = "#6ee7b7";
    vpCtx.fillText(UI.lblOut3d, lblOut.sx - 75, lblOut.sy - 8);

    drawOscilloscope(nowTs * 0.001);
    document.getElementById("pmtFlashBadge").style.opacity = (0.55 + 0.45 * state.pmtFlashEnergy).toFixed(2);
    requestAnimationFrame(renderLoop);
  }

  rebuildPhotonPool();
  requestAnimationFrame(renderLoop);
})();"""


def buildInteractive3dSimulatorHtml(
    outputPath: Path,
    surfaceDf: pd.DataFrame,
    scanDf: pd.DataFrame,
    metrics: dict[str, Any],
    lang: str = "zh",
) -> None:
    """
    Build the offline interactive 3D photon transport and PMT readout HTML workbench.
    """
    logger = logging.getLogger(__name__)
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    ui = getSimulatorUiTexts(lang)

    pivR = surfaceDf.pivot(index="theta_deg", columns="z_mm", values="radius_mm")
    pivDelta = surfaceDf.pivot(index="theta_deg", columns="z_mm", values="delta_r_non_axi_um")
    rSub = np.round(pivR.to_numpy(dtype=np.float32)[::3, ::4], 4).tolist()
    dSub = np.round(pivDelta.to_numpy(dtype=np.float32)[::3, ::4], 1).tolist()

    measGrid = np.zeros((100, 100), dtype=np.float32)
    simGrid = np.zeros((100, 100), dtype=np.float32)
    for row in scanDf.itertuples(index=False):
        ix, iy = int(row.idx_x), int(row.idx_y)
        measGrid[iy, ix] = float(row.probe_V) * 1000.0
        simGrid[iy, ix] = float(row.simulated_probe_V) * 1000.0

    bgMv = float(np.percentile(measGrid, 8.0))
    measNet = np.round(np.maximum(0.0, measGrid - bgMv), 2).tolist()
    simNet = np.round(np.maximum(0.0, simGrid - bgMv), 2).tolist()

    embeddedData: dict[str, Any] = {
        "nTheta": len(rSub),
        "nZ": len(rSub[0]),
        "coneLenMm": 100.0,
        "rInIdealMm": 8.44,
        "rOutIdealMm": 4.00,
        "radiusGridMm": rSub,
        "deltaUmGrid": dSub,
        "measMv100": measNet,
        "simMv100": simNet,
        "centerXMm": 2.3665,
        "centerYMm": -4.3788,
        "xMinMm": -7.5,
        "yMinMm": -14.5,
        "gridStepMm": 0.2,
        "metrics": {
            "rmsePct": round(float(metrics.get("innerRmsePct", 1.56)), 2),
            "rmseMv": round(float(metrics.get("innerRmseMv", 0.40)), 3),
            "totalCorr": round(float(metrics.get("totalCorr", 0.975)), 4),
            "nonAxiCorr": round(float(metrics.get("nonAxiCorr", 0.879)), 4),
            "peakBulgeUm": round(float(metrics.get("peakBulgeUm", -187.0)), 0),
        },
        "ui": ui,
    }
    jsBody = SIMULATOR_JS_TEMPLATE.replace("__JSON_PAYLOAD__", json.dumps(embeddedData, separators=(",", ":")))

    htmlContent = f"""<!DOCTYPE html>
<html lang="{ui['htmlLang']}">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{ui['pageTitle']}</title>
<style>
  :root {{ color-scheme: dark; --bg-main: #020617; --bg-card: #0f172a; --bg-sub: #1e293b; --border-col: #334155; --txt-pri: #f8fafc; --txt-sec: #94a3b8; --acc-cyan: #22d3ee; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg-main); color: var(--txt-pri); font-family: {ui['fontStack']}; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
  header {{ padding: 12px 22px; background: var(--bg-card); border-bottom: 1px solid var(--border-col); display: flex; justify-content: space-between; align-items: center; gap: 16px; }}
  .header-title h1 {{ font-size: 17.5px; font-weight: 800; letter-spacing: 0.2px; }}
  .header-title p {{ font-size: 12px; color: var(--txt-sec); margin-top: 3px; }}
  .badge-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
  .metric-pill {{ background: #020617; border: 1px solid var(--border-col); border-radius: 8px; padding: 5px 11px; font-size: 12px; color: var(--txt-sec); }}
  .metric-pill strong {{ color: var(--acc-cyan); font-size: 13.5px; margin-left: 4px; }}
  .workspace {{ flex: 1; display: grid; grid-template-columns: 1fr 410px; gap: 12px; padding: 12px; min-height: 0; }}
  .stage-panel {{ position: relative; background: radial-gradient(circle at 45% 48%, #0f172a 0%, #020617 85%); border: 1px solid var(--border-col); border-radius: 14px; overflow: hidden; display: flex; flex-direction: column; }}
  #viewport3d {{ width: 100%; height: 100%; cursor: grab; display: block; }}
  #viewport3d:active {{ cursor: grabbing; }}
  .stage-hud {{ position: absolute; top: 14px; left: 16px; background: rgba(15, 23, 42, 0.84); backdrop-filter: blur(8px); border: 1px solid rgba(51, 65, 85, 0.85); border-radius: 10px; padding: 10px 14px; pointer-events: none; font-size: 12.5px; line-height: 1.65; }}
  .legend-bar {{ position: absolute; bottom: 14px; left: 16px; right: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px; background: rgba(15, 23, 42, 0.86); border: 1px solid rgba(51, 65, 85, 0.85); border-radius: 10px; padding: 8px 14px; font-size: 11.8px; color: var(--txt-sec); }}
  .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }}
  .side-column {{ display: flex; flex-direction: column; gap: 10px; overflow-y: auto; padding-right: 2px; }}
  .card {{ background: var(--bg-card); border: 1px solid var(--border-col); border-radius: 12px; padding: 12px 14px; }}
  .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; gap: 6px; }}
  .card-title {{ font-size: 13.2px; font-weight: 700; color: var(--txt-pri); }}
  .btn-group {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  button {{ background: var(--bg-sub); color: var(--txt-pri); border: 1px solid var(--border-col); border-radius: 7px; padding: 5px 10px; font-size: 11.8px; font-weight: 600; cursor: pointer; transition: all 0.15s ease; }}
  button:hover {{ border-color: var(--acc-cyan); }}
  button.active {{ background: #0e7490; border-color: var(--acc-cyan); color: #ffffff; }}
  .scan-wrap {{ display: flex; gap: 12px; align-items: center; }}
  #scanCanvas {{ width: 196px; height: 196px; border-radius: 8px; border: 1px solid var(--border-col); cursor: crosshair; flex-shrink: 0; }}
  .scan-readout {{ font-size: 12px; color: var(--txt-sec); line-height: 1.75; flex: 1; }}
  .scan-readout b {{ color: var(--txt-pri); }}
  #oscCanvas {{ width: 100%; height: 118px; border-radius: 8px; background: #020617; border: 1px solid #1e293b; display: block; }}
  .ctrl-row {{ display: flex; flex-direction: column; gap: 4px; margin-top: 8px; font-size: 12px; color: var(--txt-sec); }}
  .ctrl-label {{ display: flex; justify-content: space-between; }}
  input[type="range"] {{ width: 100%; accent-color: var(--acc-cyan); cursor: pointer; }}
</style>
</head>
<body>
<header>
  <div class="header-title">
    <h1>{ui['hdrTitle']}</h1>
    <p>{ui['hdrSub']}</p>
  </div>
  <div class="badge-row">
    <div class="metric-pill">{ui['pillRmse']} <strong id="hdrRmse">--</strong></div>
    <div class="metric-pill">{ui['pillCorr']} <strong id="hdrCorr">--</strong></div>
    <div class="metric-pill">{ui['pillBump']} <strong id="hdrBumpCorr">--</strong></div>
  </div>
</header>
<div class="workspace">
  <div class="stage-panel">
    <canvas id="viewport3d"></canvas>
    <div class="stage-hud">
      <div>{ui['hudModeLbl']}<strong id="hudSurfMode" style="color:#22d3ee;">{ui['surfDeformed']}</strong></div>
      <div>{ui['hudLaserLbl']}<strong id="hudLaserXY" style="color:#fde047;">X' = -1.20 mm, Y' = +1.60 mm</strong></div>
      <div>{ui['hudEffLbl']}<strong id="hudEffStats" style="color:#34d399;">--</strong></div>
    </div>
    <div class="legend-bar">
      <div>
        <span><i class="dot" style="background:#67e8f9;"></i>{ui['legGuided']}</span>
        <span style="margin-left:12px;"><i class="dot" style="background:#fb7185;"></i>{ui['legEscape']}</span>
        <span style="margin-left:12px;"><i class="dot" style="background:#3b82f6;"></i>{ui['legBulge']}</span>
        <span style="margin-left:12px;"><i class="dot" style="background:#f43f5e;"></i>{ui['legDent']}</span>
      </div>
      <div>{ui['legReset']}</div>
    </div>
  </div>
  <div class="side-column">
    <div class="card">
      <div class="card-header">
        <span class="card-title">{ui['c1Title']}</span>
        <div class="btn-group">
          <button id="btnMapSim" class="active">{ui['btnSim']}</button>
          <button id="btnMapMeas">{ui['btnMeas']}</button>
          <button id="btnAutoScan">{ui['btnAuto']}</button>
        </div>
      </div>
      <div class="scan-wrap">
        <canvas id="scanCanvas" width="200" height="200"></canvas>
        <div class="scan-readout">
          <div>{ui['roXLbl']}<b id="roX">-1.20 mm</b></div>
          <div>{ui['roYLbl']}<b id="roY">+1.60 mm</b></div>
          <div>{ui['roRLbl']}<b id="roR">2.00 mm</b></div>
          <div>{ui['roMeasLbl']}<b id="roMeas" style="color:#38bdf8;">-- mV</b></div>
          <div>{ui['roSimLbl']}<b id="roSim" style="color:#34d399;">-- mV</b></div>
          <div style="margin-top:4px;font-size:11.5px;color:#64748b;">{ui['scanHint']}</div>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-header">
        <span class="card-title">{ui['c2Title']}</span>
        <span id="pmtFlashBadge" style="font-size:12px;color:#38bdf8;font-weight:700;">{ui['pmtOnline']}</span>
      </div>
      <canvas id="oscCanvas" width="376" height="118"></canvas>
    </div>
    <div class="card">
      <div class="card-header">
        <span class="card-title">{ui['c3Title']}</span>
        <div class="btn-group">
          <button id="btnDeformed" class="active">{ui['btnDef']}</button>
          <button id="btnIdeal">{ui['btnIdeal']}</button>
        </div>
      </div>
      <div class="ctrl-row">
        <div class="ctrl-label"><span>{ui['exagTitle']}</span><strong id="lblExag" style="color:#22d3ee;">{ui['exagInit']}</strong></div>
        <input id="rngExag" type="range" min="1" max="20" step="1" value="8" />
      </div>
      <div class="ctrl-row">
        <div class="ctrl-label"><span>{ui['speedTitle']}</span><strong id="lblSpeed" style="color:#22d3ee;">{ui['speedInit']}</strong></div>
        <input id="rngSpeed" type="range" min="0.2" max="2.8" step="0.1" value="1.0" />
      </div>
      <div class="ctrl-row">
        <div class="ctrl-label"><span>{ui['countTitle']}</span><strong id="lblCount" style="color:#22d3ee;">{ui['countInit']}</strong></div>
        <input id="rngCount" type="range" min="14" max="90" step="4" value="42" />
      </div>
    </div>
  </div>
</div>
<script>
{jsBody}
</script>
</body>
</html>
"""
    outputPath.write_text(htmlContent, encoding="utf-8")
    logger.info("Generated interactive 3D simulator HTML (%s): %s (%.1f KB)", lang, outputPath, outputPath.stat().st_size / 1024.0)
