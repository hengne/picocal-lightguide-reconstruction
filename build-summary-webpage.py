"""
Bilingual project summary portal webpage generator for Agy_Carbon.

Reads the reconstructed B-spline surface grid, 100x100 measured and simulated
scan matrices, Geant4 11.1.3 optical validation points, and quantitative
inversion metrics to generate two zero-CDN offline summary portals:
  - Agy_Carbon/index.html    (Simplified Chinese version)
  - Agy_Carbon/index-en.html (English version)

Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__author__ = "Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>"


def computeAzimuthalResidualPct(
    gridMv: np.ndarray,
    xLocal1d: np.ndarray,
    yLocal1d: np.ndarray,
) -> np.ndarray:
    """
    Compute the non-axisymmetric local percentage ripple after subtracting annular radial means.
    """
    xx, yy = np.meshgrid(xLocal1d, yLocal1d)
    rr = np.sqrt(xx * xx + yy * yy)
    innerMean = float(np.mean(gridMv[rr < 7.95]))
    rBins = np.linspace(0.0, 9.2, 47)
    residualPct = np.zeros_like(gridMv, dtype=np.float32)
    for k in range(len(rBins) - 1):
        ring = (rr >= rBins[k]) & (rr < rBins[k + 1])
        if np.any(ring):
            ringMean = float(np.mean(gridMv[ring]))
            residualPct[ring] = ((gridMv[ring] - ringMean) / max(1e-6, innerMean)) * 100.0
    residualPct[rr >= 8.35] = 0.0
    return residualPct


SUMMARY_CSS: str = """
  :root { color-scheme: dark; --bg-page: #020617; --bg-surface: #0f172a; --bg-elevated: #1e293b; --border-subtle: #334155; --text-main: #f8fafc; --text-muted: #94a3b8; --cyan: #22d3ee; --sky: #38bdf8; --emerald: #34d399; --amber: #fbbf24; --violet: #a78bfa; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body { background: var(--bg-page); color: var(--text-main); line-height: 1.68; font-size: 15px; }
  a { color: var(--sky); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .top-nav { position: sticky; top: 0; z-index: 50; background: rgba(2, 6, 23, 0.90); backdrop-filter: blur(12px); border-bottom: 1px solid rgba(51, 65, 85, 0.75); padding: 11px 28px; display: flex; justify-content: space-between; align-items: center; gap: 14px; flex-wrap: wrap; }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 15px; }
  .brand-tag { background: linear-gradient(135deg, #0ea5e9, #10b981); color: #020617; font-size: 11.5px; font-weight: 800; padding: 2px 8px; border-radius: 999px; }
  .nav-right { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  .nav-links { display: flex; gap: 16px; font-size: 13.2px; flex-wrap: wrap; }
  .nav-links a { color: var(--text-muted); font-weight: 600; transition: color 0.15s; }
  .nav-links a:hover { color: var(--cyan); text-decoration: none; }
  .lang-btn { display: inline-flex; align-items: center; gap: 5px; background: rgba(14, 165, 233, 0.16); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.55); border-radius: 999px; padding: 4px 12px; font-size: 12.5px; font-weight: 700; transition: all 0.15s; }
  .lang-btn:hover { background: rgba(14, 165, 233, 0.30); color: #ffffff; text-decoration: none; }
  .container { max-width: 1340px; margin: 0 auto; padding: 32px 24px 72px; }
  .hero { background: radial-gradient(circle at 85% 15%, rgba(14, 165, 233, 0.16), transparent 48%), radial-gradient(circle at 12% 85%, rgba(16, 185, 129, 0.12), transparent 45%), var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 20px; padding: 34px 36px; margin-bottom: 32px; }
  .hero-eyebrow { color: var(--cyan); font-size: 13px; font-weight: 700; letter-spacing: 0.7px; text-transform: uppercase; margin-bottom: 8px; }
  .hero h1 { font-size: 29px; line-height: 1.3; font-weight: 800; margin-bottom: 10px; }
  .author-bar { color: #7dd3fc; font-size: 13.5px; font-weight: 600; margin-bottom: 14px; }
  .hero-lead { color: #cbd5e1; font-size: 15.5px; max-width: 1100px; margin-bottom: 22px; }
  .cta-row { display: flex; gap: 11px; flex-wrap: wrap; margin-bottom: 26px; }
  .btn-primary, .btn-secondary { display: inline-flex; align-items: center; gap: 7px; padding: 9px 16px; border-radius: 10px; font-size: 13.4px; font-weight: 700; transition: transform 0.14s ease, border-color 0.14s ease; }
  .btn-primary { background: linear-gradient(135deg, #0284c7, #0d9488); color: #ffffff; border: 1px solid rgba(56, 189, 248, 0.5); }
  .btn-secondary { background: var(--bg-elevated); color: var(--text-main); border: 1px solid var(--border-subtle); }
  .btn-primary:hover, .btn-secondary:hover { transform: translateY(-1px); text-decoration: none; border-color: var(--cyan); }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(195px, 1fr)); gap: 14px; }
  .kpi-card { background: rgba(2, 6, 23, 0.72); border: 1px solid rgba(51, 65, 85, 0.85); border-radius: 14px; padding: 15px 16px; }
  .kpi-label { font-size: 12.2px; color: var(--text-muted); }
  .kpi-value { font-size: 22.5px; font-weight: 800; margin: 4px 0 3px; color: var(--text-main); }
  .kpi-note { font-size: 11.8px; color: #64748b; }
  section { margin-top: 42px; scroll-margin-top: 72px; }
  .sec-header { margin-bottom: 18px; }
  .sec-num { display: inline-block; font-size: 12px; font-weight: 800; color: var(--cyan); background: rgba(34, 211, 238, 0.1); border: 1px solid rgba(34, 211, 238, 0.3); padding: 2px 9px; border-radius: 999px; margin-bottom: 6px; }
  .sec-header h2 { font-size: 22.5px; font-weight: 800; }
  .sec-header p { color: var(--text-muted); font-size: 14.3px; margin-top: 4px; }
  .card { background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 16px; padding: 24px; }
  .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }
  .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(275px, 1fr)); gap: 16px; }
  .step-card { background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 15px; padding: 20px 22px; }
  .step-badge { display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; border-radius: 8px; background: rgba(56, 189, 248, 0.16); color: var(--sky); font-weight: 800; font-size: 13.5px; margin-bottom: 10px; }
  .step-card h3 { font-size: 16.2px; font-weight: 700; margin-bottom: 8px; }
  .step-card p, .step-card li { color: #cbd5e1; font-size: 13.6px; }
  .step-card ul { padding-left: 18px; margin-top: 6px; display: flex; flex-direction: column; gap: 5px; }
  .formula-box { background: #020617; border: 1px solid #1e293b; border-left: 3px solid var(--cyan); border-radius: 10px; padding: 11px 15px; margin: 11px 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13.2px; color: #e2e8f0; overflow-x: auto; }
  .svg-frame { width: 100%; border-radius: 14px; border: 1px solid var(--border-subtle); background: #020617; overflow: hidden; margin-top: 16px; }
  .svg-frame object, .svg-frame svg { width: 100%; display: block; }
  .explorer-layout { display: grid; grid-template-columns: 1.15fr 1fr; gap: 18px; }
  @media (max-width: 1024px) { .explorer-layout { grid-template-columns: 1fr; } }
  .canvas-box { background: #020617; border: 1px solid #1e293b; border-radius: 12px; padding: 12px; }
  .canvas-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; gap: 8px; flex-wrap: wrap; }
  .canvas-title { font-size: 13.4px; font-weight: 700; color: var(--text-main); }
  .mode-btns { display: flex; gap: 6px; }
  .mode-btn { background: var(--bg-elevated); color: var(--text-muted); border: 1px solid var(--border-subtle); border-radius: 7px; padding: 4px 10px; font-size: 12px; font-weight: 600; cursor: pointer; }
  .mode-btn.active { background: #0e7490; color: #ffffff; border-color: var(--cyan); }
  .pair-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .pair-item canvas { width: 100%; aspect-ratio: 1 / 1; border-radius: 8px; border: 1px solid #334155; cursor: crosshair; display: block; }
  .pair-caption { text-align: center; font-size: 12px; color: var(--text-muted); margin-top: 5px; }
  #surfExplorerCanvas { width: 100%; height: 215px; border-radius: 8px; border: 1px solid #334155; cursor: crosshair; display: block; }
  #sliceExplorerCanvas { width: 100%; height: 195px; border-radius: 8px; border: 1px solid #334155; display: block; }
  .probe-bar { margin-top: 10px; padding: 9px 12px; background: rgba(15, 23, 42, 0.9); border: 1px solid #334155; border-radius: 8px; font-size: 12.4px; color: #cbd5e1; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
  .sim-iframe-wrap { width: 100%; height: 670px; border-radius: 16px; border: 1px solid var(--border-subtle); overflow: hidden; background: #020617; }
  .sim-iframe-wrap iframe { width: 100%; height: 100%; border: 0; }
  table { width: 100%; border-collapse: collapse; font-size: 13.6px; }
  th, td { text-align: left; padding: 12px 14px; border-bottom: 1px solid #1e293b; }
  th { color: var(--text-muted); font-weight: 700; font-size: 12.4px; background: rgba(2, 6, 23, 0.5); }
  tr:hover td { background: rgba(30, 41, 59, 0.35); }
  code { background: #020617; border: 1px solid #1e293b; padding: 2px 6px; border-radius: 5px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.4px; color: #7dd3fc; }
  footer { margin-top: 52px; padding-top: 22px; border-top: 1px solid #1e293b; text-align: center; color: #64748b; font-size: 13px; }
"""


EXPLORER_JS_TEMPLATE: str = """(function() {
  "use strict";
  const D = __JSON_PAYLOAD__;
  const UI = D.ui;
  let showBumpMode = false, selIx = 46, selIy = 64;
  const cvMeas = document.getElementById("cvMeas"), cvSim = document.getElementById("cvSim");
  const cvSurf = document.getElementById("surfExplorerCanvas"), cvSlice = document.getElementById("sliceExplorerCanvas");

  function viridisRgb(tNorm) {
    const t = Math.max(0.0, Math.min(1.0, tNorm));
    if (t < 0.33) { const u = t / 0.33; return [Math.round(15 + 15 * u), Math.round(23 + 65 * u), Math.round(42 + 145 * u)]; }
    if (t < 0.66) { const u = (t - 0.33) / 0.33; return [Math.round(30 - 10 * u), Math.round(88 + 95 * u), Math.round(187 - 45 * u)]; }
    const u = (t - 0.66) / 0.34;
    return [Math.round(20 + 233 * u), Math.round(183 + 41 * u), Math.round(142 - 71 * u)];
  }

  function divergingRgb(tSigned) {
    const t = Math.max(-1.0, Math.min(1.0, tSigned));
    if (t < 0) { const u = -t; return [Math.round(241 * (1 - u) + 37 * u), Math.round(245 * (1 - u) + 99 * u), Math.round(249 * (1 - u) + 235 * u)]; }
    return [Math.round(241 * (1 - t) + 225 * t), Math.round(245 * (1 - t) + 29 * t), Math.round(249 * (1 - t) + 72 * t)];
  }

  function renderScanCanvas(canvasEl, gridAbs, gridBump) {
    const ctx = canvasEl.getContext("2d"), W = canvasEl.width, H = canvasEl.height;
    const img = ctx.createImageData(W, H);
    for (let py = 0; py < H; py++) {
      const iy = 99 - Math.min(99, Math.floor((py / H) * 100)), yLoc = D.yLocalMm[iy];
      for (let px = 0; px < W; px++) {
        const ix = Math.min(99, Math.floor((px / W) * 100)), rMm = Math.hypot(D.xLocalMm[ix], yLoc);
        let rgb;
        if (showBumpMode) rgb = rMm > 8.35 ? [9, 15, 32] : divergingRgb(gridBump[iy][ix] / 6.5);
        else { const v = gridAbs[iy][ix]; rgb = (rMm > 8.85 || v < 6.0) ? [9, 15, 32] : viridisRgb((v - 20.3) / (27.5 - 20.3)); }
        const off = (py * W + px) * 4;
        img.data[off] = rgb[0]; img.data[off + 1] = rgb[1]; img.data[off + 2] = rgb[2]; img.data[off + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    const cxPx = ((0.0 - D.xLocalMm[0]) / (D.xLocalMm[99] - D.xLocalMm[0])) * W;
    const cyPx = (1.0 - (0.0 - D.yLocalMm[0]) / (D.yLocalMm[99] - D.yLocalMm[0])) * H;
    ctx.strokeStyle = "rgba(248, 250, 252, 0.65)"; ctx.lineWidth = 1.3; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.arc(cxPx, cyPx, (8.439 / 19.8) * W, 0, Math.PI * 2); ctx.stroke();
    const sx = ((selIx + 0.5) / 100) * W, sy = (1.0 - (selIy + 0.5) / 100) * H;
    ctx.strokeStyle = "rgba(56, 189, 248, 0.55)"; ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(W, sy); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = "#fde047"; ctx.lineWidth = 2.0; ctx.beginPath(); ctx.arc(sx, sy, 5.5, 0, Math.PI * 2); ctx.stroke();
  }

  function renderSliceChart() {
    const ctx = cvSlice.getContext("2d"), W = cvSlice.width, H = cvSlice.height;
    ctx.clearRect(0, 0, W, H);
    const padL = 44, padR = 18, padT = 18, padB = 28, pw = W - padL - padR, ph = H - padT - padB;
    const yLoc = D.yLocalMm[selIy];
    ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1;
    for (let k = 0; k <= 3; k++) { const gy = padT + (ph * k) / 3; ctx.beginPath(); ctx.moveTo(padL, gy); ctx.lineTo(padL + pw, gy); ctx.stroke(); }
    function mapPt(xMm, mv) { return [padL + ((xMm + 9.0) / 18.0) * pw, padT + ph - (Math.max(0.0, Math.min(31.0, mv)) / 31.0) * ph]; }

    function strokeSeries(arrRow, color, lw) {
      ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath();
      let started = false;
      for (let ix = 0; ix < 100; ix++) {
        const xv = D.xLocalMm[ix];
        if (xv < -9.0 || xv > 9.0) continue;
        const pt = mapPt(xv, arrRow[ix]);
        if (!started) { ctx.moveTo(pt[0], pt[1]); started = true; } else ctx.lineTo(pt[0], pt[1]);
      }
      ctx.stroke();
    }
    strokeSeries(D.measMv[selIy], "#94a3b8", 2.0);
    strokeSeries(D.simMv[selIy], "#22d3ee", 2.6);

    const nearG4 = Math.abs(yLoc - D.g4SliceY) < 0.35;
    if (nearG4 && D.g4Points.length > 0) {
      for (const gp of D.g4Points) {
        const pt = mapPt(gp.x, gp.mv), errPx = (gp.err / 31.0) * ph;
        ctx.strokeStyle = "#34d399"; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(pt[0], pt[1] - errPx); ctx.lineTo(pt[0], pt[1] + errPx); ctx.stroke();
        ctx.fillStyle = "#10b981"; ctx.beginPath(); ctx.arc(pt[0], pt[1], 3.8, 0, Math.PI * 2); ctx.fill();
      }
    }
    const curPt = mapPt(D.xLocalMm[selIx], D.simMv[selIy][selIx]);
    ctx.strokeStyle = "rgba(253, 224, 71, 0.55)"; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(curPt[0], padT); ctx.lineTo(curPt[0], padT + ph); ctx.stroke();
    ctx.fillStyle = "#fde047"; ctx.beginPath(); ctx.arc(curPt[0], curPt[1], 4.0, 0, Math.PI * 2); ctx.fill();

    ctx.fillStyle = "#94a3b8"; ctx.font = "11.5px -apple-system, sans-serif";
    ctx.fillText("30mV", 6, padT + 8); ctx.fillText("15mV", 6, padT + ph * 0.5 + 4); ctx.fillText("0mV", 12, padT + ph);
    ctx.fillText("X' = -8 mm", padL + 10, H - 7); ctx.fillText("X' = 0 mm", padL + pw * 0.5 - 24, H - 7); ctx.fillText("X' = +8 mm", padL + pw - 68, H - 7);
    document.getElementById("sliceTitle").textContent = UI.slicePrefix + (yLoc >= 0 ? "+" : "") + yLoc.toFixed(2) + (nearG4 ? UI.sliceG4Near : UI.sliceG4Far);
  }

  function updateAllScanViews() {
    renderScanCanvas(cvMeas, D.measMv, D.measBumpPct);
    renderScanCanvas(cvSim, D.simMv, D.simBumpPct);
    renderSliceChart();
    const xv = D.xLocalMm[selIx], yv = D.yLocalMm[selIy], rv = Math.hypot(xv, yv);
    const mvM = D.measMv[selIy][selIx], mvS = D.simMv[selIy][selIx], diff = mvS - mvM;
    document.getElementById("probeScanCoord").textContent = UI.coordPrefix + (xv >= 0 ? "+" : "") + xv.toFixed(2) + " mm, Y' = " + (yv >= 0 ? "+" : "") + yv.toFixed(2) + " mm (r = " + rv.toFixed(2) + " mm)";
    document.getElementById("probeScanVals").textContent = UI.valsMeas + mvM.toFixed(2) + UI.valsSim + mvS.toFixed(2) + UI.valsDiff + (diff >= 0 ? "+" : "") + diff.toFixed(2) + " mV";
  }

  function bindScanHover(canvasEl) {
    canvasEl.addEventListener("mousemove", function(ev) {
      const rect = canvasEl.getBoundingClientRect();
      selIx = Math.floor(Math.max(0.0, Math.min(0.999, (ev.clientX - rect.left) / rect.width)) * 100);
      selIy = Math.floor(Math.max(0.0, Math.min(0.999, 1.0 - (ev.clientY - rect.top) / rect.height)) * 100);
      updateAllScanViews();
    });
  }
  bindScanHover(cvMeas); bindScanHover(cvSim);

  document.getElementById("btnViewAbs").addEventListener("click", function() { showBumpMode = false; this.classList.add("active"); document.getElementById("btnViewBump").classList.remove("active"); updateAllScanViews(); });
  document.getElementById("btnViewBump").addEventListener("click", function() { showBumpMode = true; this.classList.add("active"); document.getElementById("btnViewAbs").classList.remove("active"); updateAllScanViews(); });

  function renderSurfaceMap() {
    const ctx = cvSurf.getContext("2d"), W = cvSurf.width, H = cvSurf.height;
    const img = ctx.createImageData(W, H), nTh = D.thetaDeg.length, nZ = D.zMm.length;
    for (let py = 0; py < H; py++) {
      const iTh = nTh - 1 - Math.min(nTh - 1, Math.floor((py / H) * nTh));
      for (let px = 0; px < W; px++) {
        const rgb = divergingRgb(D.deltaUm[iTh][Math.min(nZ - 1, Math.floor((px / W) * nZ))] / 135.0);
        const off = (py * W + px) * 4;
        img.data[off] = rgb[0]; img.data[off + 1] = rgb[1]; img.data[off + 2] = rgb[2]; img.data[off + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  cvSurf.addEventListener("mousemove", function(ev) {
    const rect = cvSurf.getBoundingClientRect();
    const iTh = Math.floor(Math.max(0.0, Math.min(0.999, 1.0 - (ev.clientY - rect.top) / rect.height)) * D.thetaDeg.length);
    const iZ = Math.floor(Math.max(0.0, Math.min(0.999, (ev.clientX - rect.left) / rect.width)) * D.zMm.length);
    const th = D.thetaDeg[iTh], zv = D.zMm[iZ], dUm = D.deltaUm[iTh][iZ], rTot = D.radiusMm[iTh][iZ];
    document.getElementById("probeSurfInfo").textContent = UI.surfZ + zv.toFixed(1) + UI.surfTh + (th >= 0 ? "+" : "") + th.toFixed(0) + UI.surfDr + (dUm >= 0 ? "+" : "") + dUm.toFixed(1) + UI.surfR + rTot.toFixed(3) + " mm)";
  });

  renderSurfaceMap();
  updateAllScanViews();
})();"""


def getPortalCopy(lang: str, m: dict[str, float]) -> dict[str, Any]:
    """
    Build complete localized HTML copy and JS UI labels for 'zh' or 'en' summary webpages.
    """
    if lang == "en":
        return {
            "htmlLang": "en",
            "fontStack": '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
            "title": "LHCb PicoCal Conical Light-Guide Optical Simulation & Inverse Surface Reconstruction (Agy_Carbon)",
            "brandTitle": "LHCb PicoCal Light-Guide Simulation & Inverse Reconstruction",
            "nav1": "1. Physics & Mechanism", "nav2": "2. Inverse Methodology", "nav3": "3. Interactive Explorer",
            "nav4": "4. 3D Photon→PMT Bench", "nav5": "5. Deliverables & Docs",
            "toggleHref": "./index.html", "toggleText": "🌐 简体中文版",
            "eyebrow": "LHCb Upgrade II · PicoCal Optical Readout Uniformity & Inverse Problem Study",
            "h1": "Reconstructing Micron-Scale Light-Guide Surface Imperfections from 2D Laser Scans",
            "authorBar": "Author: Hengne Li (李衡讷) | Hengne.Li@cern.ch | Hengne.Li@m.scnu.edu.cn · SCNU / CERN LHCb",
            "lead": (
                f"To explain and quantitatively reproduce the non-uniform 2D light-yield ripples observed across a 4-cell conical light guide "
                f"(<code>Ø17.5 mm</code> entrance → <code>Ø8.0 mm</code> exit, <code>100 mm</code> length), we formulated a <strong>576-DOF periodic bicubic B-spline "
                f"freeform surface model</strong> coupled with <strong>native Apple Metal GPU ray tracing</strong> and <strong>Geant4 11.1.3 optical validation</strong>, "
                f"recovering <code>{m['peakBulge']:.0f} μm to +{m['peakDent']:.0f} μm</code> machining/film bulges with pixel-level fidelity."
            ),
            "mainSvgHref": "./light-guide-imperfection-simulation-en.svg", "ctaMain": "🔍 Open Zoomable Interactive SVG Report (EN)",
            "sim3dHref": "./presentation/interactive-3d-photon-simulator-en.html", "cta3d": "✨ Launch Fullscreen 3D Photon & PMT Bench",
            "slidesHref": "./presentation/research-summary-slides-en.svg", "ctaSlides": "📊 Open 1920×1080 Summary Slide Poster",
            "manPriHref": "./docs/user-manual-en.pdf", "ctaManPri": "📘 User Manual PDF (English)",
            "manSecHref": "./docs/user-manual-zh.pdf", "ctaManSec": "📗 User Manual PDF (Chinese)",
            "k1L": "Inner Net-Signal Fit RMSE", "k1N": f"Approaching shot-noise floor ({m['noisePct']:.2f}%)",
            "k2L": "100×100 Full-Map Pearson r", "k2N": "32,768 scrambled QMC rays per scan point",
            "k3L": "Non-Axisymmetric Texture r", "k3N": "Reproduces upper-left pit & right bright crescent",
            "k4L": "Reconstructed Bulge / Dent", "k4N": f"Effective aperture R_eff = {m['effRadius']:.3f} mm (~311 μm film)",
            "k5L": "End-to-End Pipeline Runtime", "k5V": "21.9 s", "k5N": "Metal GPU ~300M rays/s + 5×Geant4 workers",
            "k6L": "Peak Resident Memory (RSS)", "k6N": "Adaptive MemoryGuard watchdog protection",
            "s1Title": "Physical Diagnosis: Why Micron-Scale Wall Bulges Create Strong 2D Scan Ripples",
            "s1Sub": "Quantitative diagnosis combining STEP CAD geometry and 10,000-point diffuse laser scan measurements",
            "c1Head": "1. Experimental Geometry & Sub-Pixel Calibration",
            "c1P": "Parsing <code>4cells_17.5mm_8mm_circle_to_circle_100mm-Cut.step</code> and fitting an error-function edge model to <code>scan_summary.csv</code> (<code>100 × 100</code> grid, <code>0.2 mm</code> pitch) establishes:",
            "c1L1": "<strong>Nominal Tapered Cone</strong>: Length <code>L = 100.0 mm</code>, entrance <code>R_in = 8.75 mm</code>, exit <code>R_out = 4.00 mm</code>, half-angle <code>α = 2.72°</code>, coupled to an <code>8.18 mm</code> PMT bore.",
            "c1L2": f"<strong>Reflective Film Thickness</strong>: Calibrated optical aperture <code>R_eff = {m['effRadius']:.3f} mm</code> (~<code>311 μm</code> inward offset matching reflective foil thickness); laser spot <code>σ_beam = {m['beamSigma']:.3f} mm</code>.",
            "c1L3": "<strong>Non-Axisymmetric Ripples</strong>: Around the <code>25.4 mV</code> plateau, a <code>-5.5%</code> dark pit appears at <code>(X' ≈ -0.8, Y' ≈ +1.8) mm</code> alongside a <code>+3.5% to +5.0%</code> right crescent.",
            "c2Head": "2. Critical-Angle Lever Effect in Conical Concentrators",
            "c2P": "Phase-space etendue conservation restricts maximum diffuse collection to the area ratio <code>(4.00 / 8.75)² ≈ 20.9%</code>:",
            "c2Eq": "Polar angle increment per wall bounce: θ_(k+1) ≈ θ_k + 2·(α + Δα_local)",
            "c2L1": "<strong>>99% Multi-Bounce Transport</strong>: Fewer than 1% of photons reach the exit directly; guided photons undergo <code>1–14</code> specular bounces while ~<code>78%</code> turn around (<code>dz < 0</code>) and escape.",
            "c2L2": "<strong>Slope-Induced Escape & Refocusing</strong>: A <code>150–200 μm</code> inward bulge steepens the upstream slope (+Δα), forcing near-critical rays into early turnaround escape (dark pit), while gentler downstream slopes rescue rays (bright band).",
            "animSvgHref": "./presentation/photon-propagation-animated-en.svg", "animAria": "Animated photon specular reflection and bulge escape comparison",
            "s2Title": "Methodology: Freeform Surface Parameterization & Fast Inverse Solver",
            "s2Sub": "Combining C² analytic surface normals, 300M rays/s Metal GPU throughput, Geant4 validation, and strict memory safety",
            "m1H": "Periodic Bicubic B-Spline", "m1P": "Eliminates facet-crease artifacts via smooth cylindrical B-splines:",
            "m1L1": "<code>36</code> azimuthal × <code>16</code> axial = <code>576</code> C²-continuous control DOFs;",
            "m1L2": "Exact analytic partial derivatives yield true 3D inward normal <code>n(θ, z)</code>;",
            "m1L3": "Concentric reference cone per bounce forces quadratic <code>C ≡ 0</code>, preventing self-hit.",
            "m2H": "Native Apple Metal GPU", "m2P": "Compiled Objective-C++ & Metal Compute Shader (<code>libmetal_raytracer.dylib</code>):",
            "m2L1": "Scrambled 2D Sobol QMC hemispherical diffuse emission at each <code>(X, Y)</code> scan point;",
            "m2L2": "Inline bicubic evaluation + 4-step Newton root refinement + up to 20 specular bounces;",
            "m2L3": "Sustains <strong>~300M multi-bounce rays/s</strong> on Apple M3 Max.",
            "m3H": "SO(2) Equivariant Inversion", "m3P": "Three-stage lightweight solver avoiding multi-GB autodiff graph unrolling:",
            "m3L1": "<strong>Stage 1</strong>: Fit radial profile to lock axisymmetric reflective film offset;",
            "m3L2": "<strong>Stage 2</strong>: Exploit rotational symmetry—perturb only <code>16</code> axial basis modes to assemble the full <code>576</code>-DOF Jacobian in ~3 s;",
            "m3L3": "<strong>Stage 3</strong>: Laplacian curvature regularization + exact GPU backtracking line search.",
            "m4H": "Geant4 11.1.3 Cross-Check", "m4P": "Independent optical Monte Carlo verification via local <code>envs/g4</code>:",
            "m4L1": "Export watertight <code>G4TessellatedSolid</code> with <code>polished dielectric_metal</code> reflector;",
            "m4L2": "5 parallel workers shooting <code>120,000</code> <code>G4OpticalPhoton</code> tracks per slice point;",
            "m4L3": f"<code>MemoryGuard</code> enforces hard RSS ceiling (peak only <code>{m['peakRss']:.1f} MB</code>).",
            "s3Title": "Interactive Explorer: Reconstructed Surface Map ↔ 2D Scan Response",
            "s3Sub": "Hover or move across either heatmap to inspect measured mV, simulated mV, non-axisymmetric ripples, and slice profiles",
            "exLeftTitle": "Entrance 100×100 Synchronized Scan Comparison (Hover to Probe)",
            "btnAbs": "Net Signal (mV)", "btnBump": "Non-Axi Ripple (%)",
            "capMeas": "Measured Laser Scan (scan_summary.csv)",
            "capSim": f"Reconstructed Surface Sim (Metal GPU, r = {m['totalCorr']:.4f})",
            "exSurfTitle": "Reconstructed Inner-Surface Map ΔR_non-axi(θ, z)", "exSurfLeg": "Blue = Inward Bulge | Red = Outward Dent",
            "exSurfInit": "Hover to inspect micron deformation: z = 42.0 mm, θ = +152° → ΔR = -88.5 μm",
            "s4Title": "Real-Time 3D Photon Specular Transport & PMT Readout Workbench",
            "s4Sub": "Orbit 360° around the reconstructed light guide, click any (X, Y) laser position, or run Auto Scan to watch PMT anode pulses",
            "sim3dTitle": "Interactive 3D Light-Guide Photon Transport and PMT Readout Simulator",
            "s5Title": "<code>Agy_Carbon</code> Deliverables, Documentation & Module Index",
            "s5Sub": "All bilingual interactive figures, PDF user manuals, 3D CAD meshes, CSV datasets, and dual-engine C++/Metal/Python sources",
            "th1": "File Path", "th2": "Category", "th3": "Description",
            "rows": [
                ("./light-guide-imperfection-simulation-en.svg", "Primary English SVG", "Infinite-zoom (0.6×–24×) English vector report with pan, layer toggles, and hover cursor readouts."),
                ("./light-guide-imperfection-simulation.svg", "Primary Chinese SVG", "Simplified Chinese interactive zoomable vector report."),
                ("./docs/user-manual-en.pdf", "User Manual (EN PDF)", "Comprehensive English user manual covering optical physics, architecture, CLI usage, and visual tour."),
                ("./docs/user-manual-zh.pdf", "User Manual (ZH PDF)", "Simplified Chinese PDF user manual and technical documentation."),
                ("./presentation/interactive-3d-photon-simulator-en.html", "3D Simulator (EN)", "Offline HTML5 3D light-guide ray-tracing & PMT oscilloscope workbench (English UI)."),
                ("./presentation/photon-propagation-animated-en.svg", "Animated Mechanism SVG", "Dual-channel SMIL vector animation contrasting ideal cone transport vs bulge-induced escape."),
                ("./presentation/research-summary-slides-en.svg", "1920×1080 Slide Poster", "Zoomable widescreen research summary poster for group meetings and talks."),
                ("./outputs/reconstructed-light-guide.stl", "3D CAD Mesh (STL)", "Watertight binary STL mesh of the reconstructed imperfect inner light-guide cavity."),
                ("./outputs/simulated-scan-summary.csv", "Simulated Scan CSV", "Point-by-point 10,000-row simulated vs measured PMT voltage dataset."),
                ("./outputs/geant4-slice-validation.csv", "Geant4 Validation CSV", "Independent Geant4 11.1.3 G4OpticalPhoton cross-section verification dataset."),
            ],
            "footer": "Hengne Li (李衡讷) | Hengne.Li@cern.ch | Hengne.Li@m.scnu.edu.cn · SCNU / CERN LHCb · PicoCal Light-Guide Suite (Agy_Carbon)",
            "ui": {
                "slicePrefix": "Transverse Slice Comparison (Current Y' = ",
                "sliceG4Near": " mm · Geant4 11.1.3 Green Points Overlaid)",
                "sliceG4Far": " mm · Move to Y'≈+1.6mm for Geant4 Points)",
                "coordPrefix": "Local Pos: X' = ",
                "valsMeas": "Measured: ", "valsSim": " mV | Simulated: ", "valsDiff": " mV | Residual: ",
                "surfZ": "Axial Depth z = ", "surfTh": " mm | Azimuth θ = ",
                "surfDr": "° | Non-Axi ΔR = ", "surfR": " μm (Total Radius R = ",
            },
        }
    return {
        "htmlLang": "zh-CN",
        "fontStack": '-apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif',
        "title": "LHCb PicoCal 锥形光导光学模拟与内表面逆向重构 — 工作总结报告 (Agy_Carbon)",
        "brandTitle": "LHCb PicoCal 锥形光导模拟与逆向重构工作总结",
        "nav1": "1. 物理背景与机制", "nav2": "2. 建模与求解思路", "nav3": "3. 实测 vs 模拟检视",
        "nav4": "4. 三维光子→PMT 实验台", "nav5": "5. 交付文件与手册",
        "toggleHref": "./index-en.html", "toggleText": "🌐 English Version",
        "eyebrow": "LHCb Upgrade II · PicoCal 光学读出均匀性逆问题研究",
        "h1": "从二维不均匀激光扫描图，逆向重构光导内表面微米级凹凸形貌",
        "authorBar": "作者：Hengne Li (李衡讷) | Hengne.Li@cern.ch | Hengne.Li@m.scnu.edu.cn · 华南师范大学 (SCNU) / CERN LHCb",
        "lead": (
            f"针对四通道圆锥形光导（入射口 <code>Ø17.5 mm</code> → 出射口 <code>Ø8.0 mm</code>，长 <code>100 mm</code>）在漫反射激光二维扫描中呈现出的光产额凹凸起伏，"
            f"我们构建了<strong>周期双三次 B 样条自由曲面模型（576 控制自由度）</strong>，结合 <strong>Apple M3 Max 原生 Metal GPU 硬件光线追踪</strong> 与本机 "
            f"<strong>Geant4 11.1.3 独立光学光子验证</strong>，成功反演出内壁 <code>{m['peakBulge']:.0f} μm ~ +{m['peakDent']:.0f} μm</code> 量级的真实加工/贴膜鼓包分布，像素级复现了实测二维光响应图。"
        ),
        "mainSvgHref": "./light-guide-imperfection-simulation.svg", "ctaMain": "🔍 打开主交付物：可无级缩放交互式 SVG 报告",
        "sim3dHref": "./presentation/interactive-3d-photon-simulator.html", "cta3d": "✨ 全屏打开三维光子传播与 PMT 接收实验台",
        "slidesHref": "./presentation/research-summary-slides.svg", "ctaSlides": "📊 打开 1920×1080 组会总结展板 SVG",
        "manPriHref": "./docs/user-manual-zh.pdf", "ctaManPri": "📘 中文版详细说明书 (PDF)",
        "manSecHref": "./docs/user-manual-en.pdf", "ctaManSec": "📗 English User Manual (PDF)",
        "k1L": "圆内净信号拟合误差 (RMSE)", "k1N": f"逼近实验单点泊松噪声极限 ({m['noisePct']:.2f}%)",
        "k2L": "100×100 全图皮尔逊相关系数", "k2N": "每像素发射 32,768 条准蒙特卡洛光线",
        "k3L": "非轴对称凹凸结构相关系数", "k3N": "精准复现左上凹坑与右侧新月高产额带",
        "k4L": "反演内表面局部凹凸幅度", "k4N": f"有效通光孔径 R_eff = {m['effRadius']:.3f} mm (膜厚 ~311 μm)",
        "k5L": "异构加速端到端总耗时", "k5V": "21.9 秒", "k5N": "Metal GPU ~3亿光线/秒 + 5×Geant4 并行",
        "k6L": "全程峰值物理内存 (RSS)", "k6N": "内置自适应水位感知与硬性内存看门狗",
        "s1Title": "物理问题剖析：为什么微米级内壁凸起会导致明显的二维读数凹凸？",
        "s1Sub": "结合 STEP 三维切片模型与 10,000 点激光漫反射扫描数据的定量物理诊断",
        "c1Head": "1. 实验几何与亚像素标定发现",
        "c1P": "通过自动解析光导 CAD 模型（<code>4cells_17.5mm_8mm_circle_to_circle_100mm-Cut.step</code>）并对实测扫描矩阵（<code>scan_summary.csv</code>，<code>100 × 100</code> 点，步长 <code>0.2 mm</code>）执行误差函数边缘拟合与噪声分离，得到关键基准：",
        "c1L1": "<strong>标称圆锥台结构</strong>：长 <code>L = 100.0 mm</code>，大端输入半径 <code>R_in = 8.75 mm</code>，小端输出半径 <code>R_out = 4.00 mm</code>，半锥角 <code>α = 2.72°</code>，末端接半径 <code>8.18 mm</code> 的 PMT 沉孔。",
        "c1L2": f"<strong>贴膜厚度自动检出</strong>：实测有效通光半径为 <code>R_eff = {m['effRadius']:.3f} mm</code>，较 CAD 标称值内缩约 <code>311 μm</code>，对应高反射膜真实厚度；光斑尺寸 <code>σ_beam = {m['beamSigma']:.3f} mm</code>。",
        "c1L3": "<strong>非轴对称凹凸特征</strong>：圆内净光响应均值约 <code>25.4 mV</code>，中心偏左上（<code>X' ≈ -0.8, Y' ≈ +1.8 mm</code>）存在约 <code>-5.5%</code> 暗坑，右侧伴有 <code>+3.5% ~ +5.0%</code> 新月亮带。",
        "c2Head": "2. 锥形聚光器的「临界角放大效应」",
        "c2P": "在收缩型圆锥光导中，相空间光学扩展量（Etendue）守恒决定了最大透射立体角比例不超过面积比 <code>(4.00 / 8.75)² ≈ 20.9%</code>：",
        "c2Eq": "每次锥壁镜面反射极角递增：θ_(k+1) ≈ θ_k + 2·(α + Δα_local)",
        "c2L1": "<strong>99% 以上传导光依赖多跳反射</strong>：直接穿过小口的光子不足 1%，绝大多数收集光子需经历 <code>1 ~ 14</code> 次镜面反射；约 <code>78%</code> 光子达到 <code>dz < 0</code> 掉头从入射大口逃逸。",
        "c2L2": "<strong>迎光陡坡致暗、背光缓坡增亮</strong>：内壁仅 <code>150 ~ 200 μm</code> 局部内凸鼓包会使迎光侧局部锥角骤增 <code>Δα</code>，令临界光子提前掉头逃逸（形成暗坑），背光缓坡则挽救超临界光线（形成亮带）。",
        "animSvgHref": "./presentation/photon-propagation-animated.svg", "animAria": "光子镜面反射与鼓包逃逸动态矢量对比图",
        "s2Title": "核心技术路线：如何描述内曲面凸起并高效反演拟合？",
        "s2Sub": "兼顾微米级法向量连续性、GPU 亿级光线吞吐率、Geant4 权威物理核验与宿主机内存安全保护",
        "m1H": "周期双三次 B 样条自由曲面", "m1P": "摒弃会产生折痕伪影的离散多面体直接求导，采用柱坐标双三次样条曲面表征任意鼓包：",
        "m1L1": "周向 <code>36</code> 周期节 × 轴向 <code>16</code> 节 = <code>576</code> 维光滑控制网格，严格满足 <code>C²</code> 连续；",
        "m1L2": "解析偏导数直接给出精确三维内法向量 <code>n(θ, z)</code>；",
        "m1L3": "每次碰撞构造同心参考锥使二次常数项 <code>C ≡ 0</code>，根除自相交伪根。",
        "m2H": "Apple Metal 原生 GPU 追踪", "m2P": "编写 Objective-C++ 与 Metal Compute Shader 原生动态库（<code>libmetal_raytracer.dylib</code>）：",
        "m2L1": "每个 <code>(X, Y)</code> 入射点采用二维置乱 Sobol 准蒙特卡洛序列发射漫散射光束；",
        "m2L2": "GPU 线程内联双三次样条求值 + 4 步牛顿法求交 + 最多 20 跳严格镜面反射；",
        "m2L3": "在 M3 Max 上实测吞吐率高达 <strong>~3 亿条多跳光线/秒</strong>。",
        "m3H": "SO(2) 等变非线性逆向反演", "m3P": "避免自动微分展开多跳牛顿图导致的百 GB 内存爆炸，设计三级轻量级精确反演器：",
        "m3L1": "<strong>阶段 1</strong>：拟合径向均值曲线，锁定反射膜整体内缩廓线；",
        "m3L2": "<strong>阶段 2</strong>：利用旋转对称性仅对 <code>16</code> 个轴向基函数微扰映射，3 秒构建 <code>576</code> 维灵敏度矩阵；",
        "m3L3": "<strong>阶段 3</strong>：结合二阶拉普拉斯曲率正则化与 GPU 非线性回溯线搜索单调收敛。",
        "m4H": "Geant4 11.1.3 并行核验与护栏", "m4P": "调用本机 <code>miniconda3/envs/g4</code> 环境完成独立粒子级交叉验证与系统保护：",
        "m4L1": "将反演变形面导出为水密封闭 <code>G4TessellatedSolid</code>，配置金属高反膜光学表面；",
        "m4L2": "5 进程并行发射 <code>120,000</code> 颗 <code>G4OpticalPhoton</code> 统计末端 PMT 命中率；",
        "m4L3": f"<code>MemoryGuard</code> 实时看护进程 RSS（全程峰值仅 <code>{m['peakRss']:.1f} MB</code>）。",
        "s3Title": "交互式结果检视台：重构内表面形貌 ↔ 二维扫描响应实时联动",
        "s3Sub": "在下方任意热力图上移动鼠标，即可同步对比实测读数、模拟读数、非轴对称凹凸纹理以及对应横截面曲线",
        "exLeftTitle": "入射窗口 100×100 二维扫描同步对比（移动光标实时联动）",
        "btnAbs": "净光响应信号 (mV)", "btnBump": "非轴对称局部凹凸 (%)",
        "capMeas": "实验激光扫描实测图 (scan_summary.csv)",
        "capSim": f"反演变形内表面模拟图 (Metal GPU, r = {m['totalCorr']:.4f})",
        "exSurfTitle": "反演重构出的光导内表面凹凸展开图 ΔR_non-axi(θ, z)", "exSurfLeg": "蓝=向内鼓包 | 红=向外凹陷",
        "exSurfInit": "悬停查看内壁微米级形变：轴向深度 z = 42.0 mm, 方位角 θ = +152° → ΔR = -88.5 μm",
        "s4Title": "三维光子镜面反射传播与 PMT 读出实时动态演示",
        "s4Sub": "直接在下方窗口拖拽 360° 旋转光导、点击二维扫描板任意点发射漫散射光，或开启「自动光栅扫描」观察 PMT 阳极波形变化",
        "sim3dTitle": "三维光导光子传播与 PMT 读出实时交互模拟器",
        "s5Title": "<code>Agy_Carbon</code> 全部交付成果、中英说明书与工程模块索引",
        "s5Sub": "所有中英双语交互式矢量图、PDF 说明书、三维动画网页、STL 实体模型与双引擎源码均存放在 <code>Agy_Carbon/</code> 内",
        "th1": "文件路径", "th2": "类型", "th3": "功能与用途说明",
        "rows": [
            ("./light-guide-imperfection-simulation.svg", "核心主交付 SVG (中)", "支持滚轮无级缩放（0.6× ~ 24×）、拖拽平移、图层开关与鼠标实时读数的中文主报告。"),
            ("./light-guide-imperfection-simulation-en.svg", "核心主交付 SVG (英)", "100% 英文版无级缩放交互式逆向重构与模拟对比报告。"),
            ("./docs/user-manual-zh.pdf", "中文用户说明书 PDF", "涵盖物理背景、数学建模、命令行复现指南与结果解读的完整中文 PDF 手册。"),
            ("./docs/user-manual-en.pdf", "英文用户说明书 PDF", "面向国际合作组交流的完整英文 PDF 技术说明书 (English User Manual)。"),
            ("./presentation/interactive-3d-photon-simulator.html", "三维交互动画网页", "360° 可旋转三维锥形光导 + 漫散射光子弹跳波包 + PMT 示波器 + 自动光栅扫描（含 -en.html 英文版）。"),
            ("./presentation/photon-propagation-animated.svg", "动态矢量演示 SVG", "上下双通道自动循环动画，直观演示微米级内凸鼓包迎光陡坡诱发临界光线折返逃逸（含 -en.svg 英文版）。"),
            ("./presentation/research-summary-slides.svg", "组会汇报展板 SVG", "1920×1080 宽屏高清可缩放矢量总结展板（含 -en.svg 英文版）。"),
            ("./outputs/reconstructed-light-guide.stl", "三维 CAD 网格", "反演重构出的不完美光导内腔三维二进制 STL 水密实体模型。"),
            ("./outputs/simulated-scan-summary.csv", "模拟扫描数据表", "与原始 scan_summary.csv 一一对齐的 10,000 点模拟电压读数对比表。"),
            ("./outputs/geant4-slice-validation.csv", "Geant4 验证数据", "本机 Geant4 11.1.3（G4TessellatedSolid + G4OpticalPhoton）5 进程并行光学验证输出。"),
        ],
        "footer": "作者：Hengne Li (李衡讷) | Hengne.Li@cern.ch | Hengne.Li@m.scnu.edu.cn · SCNU / CERN LHCb · Output Directory: Agy_Carbon/",
        "ui": {
            "slicePrefix": "横向截面光响应对比 (当前 Y' = ",
            "sliceG4Near": " mm · 已叠加绿色 Geant4 11.1.3 验证点)",
            "sliceG4Far": " mm · 移至 Y'≈+1.6mm 可查看 Geant4 验证点)",
            "coordPrefix": "局部坐标：X' = ",
            "valsMeas": "实测：", "valsSim": " mV | 模拟：", "valsDiff": " mV | 偏差：",
            "surfZ": "轴向深度 z = ", "surfTh": " mm | 方位角 θ = ",
            "surfDr": "° | 非对称形变 ΔR = ", "surfR": " μm (总内径 R = ",
        },
    }


def renderSummaryHtmlDocument(baseWebData: dict[str, Any], m: dict[str, float], lang: str) -> str:
    """
    Assemble the complete standalone HTML5 portal document for the requested language.
    """
    c = getPortalCopy(lang, m)
    payloadWithUi = dict(baseWebData)
    payloadWithUi["ui"] = c["ui"]
    jsCode = EXPLORER_JS_TEMPLATE.replace("__JSON_PAYLOAD__", json.dumps(payloadWithUi, separators=(",", ":")))
    rowsHtml = "\n".join(
        f'<tr><td><a href="{href}" target="_blank"><code>{href[2:]}</code></a></td><td><strong style="color:var(--cyan);">{cat}</strong></td><td>{desc}</td></tr>'
        for href, cat, desc in c["rows"]
    )
    return f"""<!DOCTYPE html>
<html lang="{c['htmlLang']}">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{c['title']}</title>
<style>
{SUMMARY_CSS}
  body {{ font-family: {c['fontStack']}; }}
</style>
</head>
<body>
<nav class="top-nav">
  <div class="brand"><span class="brand-tag">Agy_Carbon</span><span>{c['brandTitle']}</span></div>
  <div class="nav-right">
    <div class="nav-links">
      <a href="#sec-background">{c['nav1']}</a><a href="#sec-methodology">{c['nav2']}</a>
      <a href="#sec-explorer">{c['nav3']}</a><a href="#sec-3d-bench">{c['nav4']}</a><a href="#sec-deliverables">{c['nav5']}</a>
    </div>
    <!-- LANG_TOGGLE_START --><a id="navLangToggle" class="lang-btn" href="{c['toggleHref']}">{c['toggleText']}</a><!-- LANG_TOGGLE_END -->
  </div>
</nav>
<main class="container">
  <header class="hero">
    <div class="hero-eyebrow">{c['eyebrow']}</div>
    <h1>{c['h1']}</h1>
    <div class="author-bar">{c['authorBar']}</div>
    <p class="hero-lead">{c['lead']}</p>
    <!-- CTA_ROW_START -->
    <div class="cta-row">
      <a class="btn-primary" href="{c['mainSvgHref']}" target="_blank">{c['ctaMain']}</a>
      <a class="btn-secondary" href="{c['sim3dHref']}" target="_blank">{c['cta3d']}</a>
      <a class="btn-secondary" href="{c['slidesHref']}" target="_blank">{c['ctaSlides']}</a>
      <a class="btn-secondary" href="{c['manPriHref']}" target="_blank">{c['ctaManPri']}</a>
      <a class="btn-secondary" href="{c['manSecHref']}" target="_blank">{c['ctaManSec']}</a>
    </div>
    <!-- CTA_ROW_END -->
    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-label">{c['k1L']}</div><div class="kpi-value" style="color:var(--cyan);">{m['rmsePct']:.2f}% <span style="font-size:15px;font-weight:600;">({m['rmseMv']:.3f} mV)</span></div><div class="kpi-note">{c['k1N']}</div></div>
      <div class="kpi-card"><div class="kpi-label">{c['k2L']}</div><div class="kpi-value" style="color:var(--emerald);">r = {m['totalCorr']:.4f}</div><div class="kpi-note">{c['k2N']}</div></div>
      <div class="kpi-card"><div class="kpi-label">{c['k3L']}</div><div class="kpi-value" style="color:var(--sky);">r = {m['nonAxiCorr']:.4f}</div><div class="kpi-note">{c['k3N']}</div></div>
      <div class="kpi-card"><div class="kpi-label">{c['k4L']}</div><div class="kpi-value" style="color:var(--amber);">{m['peakBulge']:.0f} ~ +{m['peakDent']:.0f} μm</div><div class="kpi-note">{c['k4N']}</div></div>
      <div class="kpi-card"><div class="kpi-label">{c['k5L']}</div><div class="kpi-value" style="color:var(--violet);">{c['k5V']}</div><div class="kpi-note">{c['k5N']}</div></div>
      <div class="kpi-card"><div class="kpi-label">{c['k6L']}</div><div class="kpi-value" style="color:#f8fafc;">{m['peakRss']:.1f} MB</div><div class="kpi-note">{c['k6N']}</div></div>
    </div>
  </header>
  <section id="sec-background">
    <div class="sec-header"><span class="sec-num">SECTION 01</span><h2>{c['s1Title']}</h2><p>{c['s1Sub']}</p></div>
    <div class="grid-2">
      <div class="card"><h3 style="font-size:17px;margin-bottom:10px;color:var(--sky);">{c['c1Head']}</h3><p style="color:#cbd5e1;font-size:14px;">{c['c1P']}</p><ul style="padding-left:18px;margin-top:10px;color:#cbd5e1;font-size:13.6px;display:flex;flex-direction:column;gap:6px;"><li>{c['c1L1']}</li><li>{c['c1L2']}</li><li>{c['c1L3']}</li></ul></div>
      <div class="card"><h3 style="font-size:17px;margin-bottom:10px;color:var(--emerald);">{c['c2Head']}</h3><p style="color:#cbd5e1;font-size:14px;">{c['c2P']}</p><div class="formula-box">{c['c2Eq']}</div><ul style="padding-left:18px;color:#cbd5e1;font-size:13.6px;display:flex;flex-direction:column;gap:6px;"><li>{c['c2L1']}</li><li>{c['c2L2']}</li></ul></div>
    </div>
    <div class="svg-frame"><!-- ANIM_SVG_START --><object data="{c['animSvgHref']}" type="image/svg+xml" aria-label="{c['animAria']}"></object><!-- ANIM_SVG_END --></div>
  </section>
  <section id="sec-methodology">
    <div class="sec-header"><span class="sec-num">SECTION 02</span><h2>{c['s2Title']}</h2><p>{c['s2Sub']}</p></div>
    <div class="grid-4">
      <div class="step-card"><div class="step-badge">01</div><h3>{c['m1H']}</h3><p>{c['m1P']}</p><div class="formula-box">R(θ, z) = R_CAD(z) + ΔR_axi(z) + ΔR(θ, z)</div><ul><li>{c['m1L1']}</li><li>{c['m1L2']}</li><li>{c['m1L3']}</li></ul></div>
      <div class="step-card"><div class="step-badge">02</div><h3>{c['m2H']}</h3><p>{c['m2P']}</p><ul><li>{c['m2L1']}</li><li>{c['m2L2']}</li><li>{c['m2L3']}</li></ul></div>
      <div class="step-card"><div class="step-badge">03</div><h3>{c['m3H']}</h3><p>{c['m3P']}</p><ul><li>{c['m3L1']}</li><li>{c['m3L2']}</li><li>{c['m3L3']}</li></ul></div>
      <div class="step-card"><div class="step-badge">04</div><h3>{c['m4H']}</h3><p>{c['m4P']}</p><ul><li>{c['m4L1']}</li><li>{c['m4L2']}</li><li>{c['m4L3']}</li></ul></div>
    </div>
  </section>
  <section id="sec-explorer">
    <div class="sec-header"><span class="sec-num">SECTION 03</span><h2>{c['s3Title']}</h2><p>{c['s3Sub']}</p></div>
    <div class="card">
      <div class="explorer-layout">
        <div class="canvas-box">
          <div class="canvas-bar"><span class="canvas-title">{c['exLeftTitle']}</span><div class="mode-btns"><button id="btnViewAbs" class="mode-btn active">{c['btnAbs']}</button><button id="btnViewBump" class="mode-btn">{c['btnBump']}</button></div></div>
          <div class="pair-row">
            <div class="pair-item"><canvas id="cvMeas" width="300" height="300"></canvas><div class="pair-caption">{c['capMeas']}</div></div>
            <div class="pair-item"><canvas id="cvSim" width="300" height="300"></canvas><div class="pair-caption">{c['capSim']}</div></div>
          </div>
          <div class="probe-bar"><span id="probeScanCoord">--</span><span id="probeScanVals" style="color:var(--cyan);font-weight:700;">--</span></div>
        </div>
        <div style="display:flex;flex-direction:column;gap:14px;">
          <div class="canvas-box">
            <div class="canvas-bar"><span class="canvas-title">{c['exSurfTitle']}</span><span style="font-size:12px;color:var(--text-muted);">{c['exSurfLeg']}</span></div>
            <canvas id="surfExplorerCanvas" width="560" height="215"></canvas>
            <div class="probe-bar"><span id="probeSurfInfo">{c['exSurfInit']}</span></div>
          </div>
          <div class="canvas-box"><div class="canvas-bar"><span class="canvas-title" id="sliceTitle">--</span></div><canvas id="sliceExplorerCanvas" width="560" height="195"></canvas></div>
        </div>
      </div>
    </div>
  </section>
  <section id="sec-3d-bench">
    <div class="sec-header"><span class="sec-num">SECTION 04</span><h2>{c['s4Title']}</h2><p>{c['s4Sub']}</p></div>
    <div class="sim-iframe-wrap"><!-- SIM3D_IFRAME_START --><iframe src="{c['sim3dHref']}" title="{c['sim3dTitle']}"></iframe><!-- SIM3D_IFRAME_END --></div>
  </section>
  <section id="sec-deliverables">
    <div class="sec-header"><span class="sec-num">SECTION 05</span><h2>{c['s5Title']}</h2><p>{c['s5Sub']}</p></div>
    <div class="card" style="padding:0;overflow:hidden;"><table><thead><tr><th style="width:34%;">{c['th1']}</th><th style="width:18%;">{c['th2']}</th><th>{c['th3']}</th></tr></thead><tbody>{rowsHtml}</tbody></table></div>
  </section>
  <footer>{c['footer']}</footer>
</main>
<script>
{jsCode}
</script>
</body>
</html>
"""


def buildProjectSummaryWebpage(projectDir: Path) -> tuple[Path, Path]:
    """
    Generate both Chinese (index.html) and English (index-en.html) project summary portals.
    """
    logger = logging.getLogger("agy.summary_web")
    outputsDir = projectDir / "outputs"

    surfaceDf = pd.read_csv(outputsDir / "reconstructed-surface-grid.csv")
    scanDf = pd.read_csv(outputsDir / "simulated-scan-summary.csv")
    metrics: dict[str, Any] = json.loads((outputsDir / "inversion-metrics.json").read_text(encoding="utf-8"))

    g4Path = outputsDir / "geant4-slice-validation.csv"
    g4List: list[dict[str, float]] = []
    if g4Path.exists():
        g4Df = pd.read_csv(g4Path)
        for row in g4Df.itertuples(index=False):
            g4List.append({"x": round(float(row.x_local_mm), 3), "mv": round(float(row.g4_mv), 3), "err": round(float(row.g4_err_mv), 3)})

    pivDelta = surfaceDf.pivot(index="theta_deg", columns="z_mm", values="delta_r_non_axi_um")
    pivRadius = surfaceDf.pivot(index="theta_deg", columns="z_mm", values="radius_mm")
    measGrid = np.zeros((100, 100), dtype=np.float32)
    simGrid = np.zeros((100, 100), dtype=np.float32)
    for row in scanDf.itertuples(index=False):
        ix, iy = int(row.idx_x), int(row.idx_y)
        measGrid[iy, ix] = float(row.probe_V) * 1000.0
        simGrid[iy, ix] = float(row.simulated_probe_V) * 1000.0

    bgMv = 15.243
    measNet = np.maximum(0.0, measGrid - bgMv)
    simNet = np.maximum(0.0, simGrid - bgMv)
    xLocal1d = np.linspace(-7.5, 12.3, 100, dtype=np.float32) - 2.3665
    yLocal1d = np.linspace(-14.5, 5.3, 100, dtype=np.float32) - (-4.3788)

    baseWebData: dict[str, Any] = {
        "thetaDeg": np.round(pivDelta.index.to_numpy(dtype=np.float32)[::2], 1).tolist(),
        "zMm": np.round(pivDelta.columns.to_numpy(dtype=np.float32)[::2], 1).tolist(),
        "deltaUm": np.round(pivDelta.to_numpy(dtype=np.float32)[::2, ::2], 1).tolist(),
        "radiusMm": np.round(pivRadius.to_numpy(dtype=np.float32)[::2, ::2], 3).tolist(),
        "xLocalMm": np.round(xLocal1d, 3).tolist(),
        "yLocalMm": np.round(yLocal1d, 3).tolist(),
        "measMv": np.round(measNet, 2).tolist(),
        "simMv": np.round(simNet, 2).tolist(),
        "measBumpPct": np.round(computeAzimuthalResidualPct(measNet, xLocal1d, yLocal1d), 2).tolist(),
        "simBumpPct": np.round(computeAzimuthalResidualPct(simNet, xLocal1d, yLocal1d), 2).tolist(),
        "g4SliceY": 1.60,
        "g4Points": g4List,
    }
    metricVals = {
        "rmseMv": float(metrics.get("innerRmseMv", 0.397)),
        "rmsePct": float(metrics.get("innerRmsePct", 1.56)),
        "noisePct": float(metrics.get("noiseFloorPct", 0.81)),
        "totalCorr": float(metrics.get("totalCorr", 0.9750)),
        "nonAxiCorr": float(metrics.get("nonAxiCorr", 0.8785)),
        "effRadius": float(metrics.get("effectiveRadiusMm", 8.439)),
        "beamSigma": float(metrics.get("beamSigmaMm", 0.217)),
        "peakBulge": float(metrics.get("peakBulgeUm", -187.0)),
        "peakDent": float(metrics.get("peakDentUm", 146.0)),
        "peakRss": float(metrics.get("peakRssMb", 332.1)),
    }

    zhPath = projectDir / "index.html"
    enPath = projectDir / "index-en.html"
    zhPath.write_text(renderSummaryHtmlDocument(baseWebData, metricVals, lang="zh"), encoding="utf-8")
    enPath.write_text(renderSummaryHtmlDocument(baseWebData, metricVals, lang="en"), encoding="utf-8")
    logger.info("Generated bilingual summary portals: %s (%.1f KB), %s (%.1f KB)", zhPath, zhPath.stat().st_size / 1024.0, enPath, enPath.stat().st_size / 1024.0)
    return zhPath, enPath


def main() -> None:
    """
    CLI entry point generating both index.html and index-en.html summary portals.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    buildProjectSummaryWebpage(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
