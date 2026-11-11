"""
Zero-external-dependency single-file standalone HTML report bundler.

Packages the summary portal, inline SMIL animated photon diagram, interactive
4-canvas data explorer, full 3D photon-to-PMT simulator (via escaped srcdoc),
and Gzip-compressed zoomable vector reports (via browser-native
DecompressionStream fullscreen modal viewer) into two self-contained single-file
HTML bundles ready for offline sharing:
  - Agy_Carbon/lhcb-picocal-light-guide-report-standalone-zh.html
  - Agy_Carbon/lhcb-picocal-light-guide-report-standalone-en.html

Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""

import base64
import gzip
import html
import logging
import re
import sys
from pathlib import Path
from typing import Any

__author__ = "Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>"


def prepareSvgForHtmlSrcdoc(svgRawText: str, htmlLang: str = "zh-CN") -> str:
    """
    Convert a standalone XML SVG document into a complete HTML5 srcdoc string
    preserving wheel zoom, drag panning, layer toggles, and cursor readouts.
    """
    cleaned = re.sub(r"<\?xml[^>]*\?>", "", svgRawText, count=1).strip()
    cleaned = cleaned.replace("<![CDATA[", "").replace("]]>", "")
    cleaned = cleaned.replace(
        "const svgEl = document.documentElement;",
        "const svgEl = document.querySelector('svg') || document.documentElement;",
    )
    return (
        f"<!DOCTYPE html><html lang='{htmlLang}' style='width:100%;height:100%;margin:0;padding:0;background:#020617;overflow:hidden;'>"
        "<body style='width:100%;height:100%;margin:0;padding:0;overflow:hidden;display:flex;'>"
        f"{cleaned}"
        "</body></html>"
    )


def compressToBase64Gzip(textContent: str) -> str:
    """
    Compress UTF-8 text at maximum Gzip level 9 and encode as ASCII Base64
    for client-side decompression via DecompressionStream('gzip').
    """
    gzBytes = gzip.compress(textContent.encode("utf-8"), compresslevel=9)
    return base64.b64encode(gzBytes).decode("ascii")


def getStandaloneBundleSpec(projectDir: Path, lang: str) -> dict[str, Any]:
    """
    Resolve input asset paths, output path, and localized modal strings for 'zh' or 'en'.
    """
    if lang == "en":
        enMainSvg = projectDir / "light-guide-imperfection-simulation-en.svg"
        mainSvgPath = enMainSvg if enMainSvg.exists() else (projectDir / "light-guide-imperfection-simulation.svg")
        return {
            "htmlLang": "en",
            "indexPath": projectDir / "index-en.html",
            "animSvgPath": projectDir / "presentation" / "photon-propagation-animated-en.svg",
            "sim3dPath": projectDir / "presentation" / "interactive-3d-photon-simulator-en.html",
            "mainSvgPath": mainSvgPath,
            "mainSvgFilename": mainSvgPath.name,
            "slidesSvgPath": projectDir / "presentation" / "research-summary-slides-en.svg",
            "slidesSvgFilename": "research-summary-slides-en.svg",
            "outPath": projectDir / "lhcb-picocal-light-guide-report-standalone-en.html",
            "badgeText": "📦 Standalone Offline Bundle",
            "sim3dTitle": "Interactive 3D Light-Guide Photon Transport and PMT Readout Simulator",
            "btnMain": "🔍 Fullscreen View: Zoomable Primary SVG Report (Embedded)",
            "btn3d": "✨ Fullscreen Bench: 3D Photon &amp; PMT Simulator (Embedded)",
            "btnAnim": "🎬 Jump to Dual-Channel Photon Escape Animation",
            "btnSlides": "📊 Fullscreen View: 1920×1080 Summary Slide Poster (Embedded)",
            "modalDefaultTitle": "Interactive Fullscreen Viewer",
            "modalSaveBtn": "💾 Export &amp; Download Standalone SVG",
            "modalCloseBtn": "✕ Close Fullscreen (Esc)",
            "modalFrameTitle": "Interactive Fullscreen Viewport",
            "titleMain": "LHCb PicoCal Interactive Primary SVG Report — Wheel Zoom (0.6×–24×) / Drag Pan / Hover Readout",
            "titleSlides": "1920×1080 Widescreen Research Summary Slide Poster — Wheel Zoom & Drag Pan",
            "title3d": "3D Light-Guide Specular Photon Transport & PMT Readout Bench (360° Orbit / Click 2D Scan Pad)",
        }
    return {
        "htmlLang": "zh-CN",
        "indexPath": projectDir / "index.html",
        "animSvgPath": projectDir / "presentation" / "photon-propagation-animated.svg",
        "sim3dPath": projectDir / "presentation" / "interactive-3d-photon-simulator.html",
        "mainSvgPath": projectDir / "light-guide-imperfection-simulation.svg",
        "mainSvgFilename": "light-guide-imperfection-simulation.svg",
        "slidesSvgPath": projectDir / "presentation" / "research-summary-slides.svg",
        "slidesSvgFilename": "research-summary-slides.svg",
        "outPath": projectDir / "lhcb-picocal-light-guide-report-standalone-zh.html",
        "badgeText": "📦 单文件离线便携版",
        "sim3dTitle": "三维光导光子传播与 PMT 读出实时交互模拟器",
        "btnMain": "🔍 全屏交互查看：可无级缩放主报告 SVG（已内嵌）",
        "btn3d": "✨ 全屏沉浸操作：三维光子传播与 PMT 实验台（已内嵌）",
        "btnAnim": "🎬 查看双通道动态光子折返动画",
        "btnSlides": "📊 全屏交互查看：1920×1080 组会总结展板 SVG（已内嵌）",
        "modalDefaultTitle": "交互式全屏视图",
        "modalSaveBtn": "💾 导出保存此独立 SVG 文件",
        "modalCloseBtn": "✕ 关闭全屏视图 (Esc)",
        "modalFrameTitle": "全屏交互视图",
        "titleMain": "LHCb PicoCal 光导逆向重构交互式主报告 SVG — 滚轮无级缩放 (0.6×~24×) / 左键拖拽 / 悬停读数",
        "titleSlides": "1920×1080 组会汇报全景总结展板 SVG — 支持滚轮无级缩放与左键拖拽平移",
        "title3d": "三维光导光子镜面反射与 PMT 读出实时交互模拟台（左键 360° 旋转 / 点击右侧圆盘移动激光）",
    }


def buildSingleStandaloneReport(projectDir: Path, lang: str = "zh") -> Path:
    """
    Synthesize one 100%-self-contained single-file HTML bundle for the requested language.
    """
    logger = logging.getLogger("agy.standalone")
    spec = getStandaloneBundleSpec(projectDir, lang)

    indexHtml = Path(spec["indexPath"]).read_text(encoding="utf-8")
    animSvgRaw = Path(spec["animSvgPath"]).read_text(encoding="utf-8")
    sim3dHtmlRaw = Path(spec["sim3dPath"]).read_text(encoding="utf-8")
    mainSvgRaw = Path(spec["mainSvgPath"]).read_text(encoding="utf-8")
    slidesSvgRaw = Path(spec["slidesSvgPath"]).read_text(encoding="utf-8")

    inlineAnimSvg = re.sub(r"<\?xml[^>]*\?>", "", animSvgRaw, count=1).strip()
    escapedSim3dSrcdoc = html.escape(sim3dHtmlRaw, quote=True)

    mainSvgGzB64 = compressToBase64Gzip(prepareSvgForHtmlSrcdoc(mainSvgRaw, spec["htmlLang"]))
    slidesSvgGzB64 = compressToBase64Gzip(prepareSvgForHtmlSrcdoc(slidesSvgRaw, spec["htmlLang"]))
    mainSvgDownloadGzB64 = compressToBase64Gzip(mainSvgRaw)
    slidesSvgDownloadGzB64 = compressToBase64Gzip(slidesSvgRaw)

    standaloneHtml = re.sub(
        r"<!-- LANG_TOGGLE_START -->.*?<!-- LANG_TOGGLE_END -->",
        f'<span class="lang-btn">{spec["badgeText"]}</span>',
        indexHtml,
        flags=re.DOTALL,
    )
    standaloneHtml = re.sub(
        r"<!-- ANIM_SVG_START -->.*?<!-- ANIM_SVG_END -->",
        lambda _: inlineAnimSvg,
        standaloneHtml,
        flags=re.DOTALL,
    )
    standaloneHtml = re.sub(
        r"<!-- SIM3D_IFRAME_START -->.*?<!-- SIM3D_IFRAME_END -->",
        lambda _: f'<iframe id="embedded3dFrame" srcdoc="{escapedSim3dSrcdoc}" title="{spec["sim3dTitle"]}"></iframe>',
        standaloneHtml,
        flags=re.DOTALL,
    )

    ctaReplacement = f"""    <div class="cta-row">
      <button type="button" class="btn-primary" style="cursor:pointer;" id="btnOpenMainSvgModal">{spec['btnMain']}</button>
      <button type="button" class="btn-secondary" style="cursor:pointer;" id="btnOpen3dModal">{spec['btn3d']}</button>
      <a class="btn-secondary" href="#sec-background">{spec['btnAnim']}</a>
      <button type="button" class="btn-secondary" style="cursor:pointer;" id="btnOpenSlidesSvgModal">{spec['btnSlides']}</button>
    </div>"""
    standaloneHtml = re.sub(
        r"<!-- CTA_ROW_START -->.*?<!-- CTA_ROW_END -->",
        ctaReplacement,
        standaloneHtml,
        flags=re.DOTALL,
    )

    # Strip all remaining relative ./ file anchors so zero broken offline links remain
    standaloneHtml = re.sub(
        r'<a\s+[^>]*href="\./[^"]*"[^>]*>(.*?)</a>',
        r"\1",
        standaloneHtml,
        flags=re.DOTALL,
    )

    modalConfigJson = (
        "{"
        f'"mainViewerGz":"{mainSvgGzB64}",'
        f'"mainRawGz":"{mainSvgDownloadGzB64}",'
        f'"mainFile":"{spec["mainSvgFilename"]}",'
        f'"slidesViewerGz":"{slidesSvgGzB64}",'
        f'"slidesRawGz":"{slidesSvgDownloadGzB64}",'
        f'"slidesFile":"{spec["slidesSvgFilename"]}",'
        f'"titleMain":"{spec["titleMain"]}",'
        f'"titleSlides":"{spec["titleSlides"]}",'
        f'"title3d":"{spec["title3d"]}"'
        "}"
    )

    modalInjection = f"""
<style>
  .fs-modal {{ position: fixed; inset: 0; z-index: 9999; background: rgba(2, 6, 23, 0.96); display: none; flex-direction: column; }}
  .fs-modal.open {{ display: flex; }}
  .fs-modal-bar {{ height: 50px; background: #0f172a; border-bottom: 1px solid #334155; padding: 0 18px; display: flex; justify-content: space-between; align-items: center; gap: 12px; }}
  .fs-modal-title {{ font-size: 14px; font-weight: 700; color: #f8fafc; }}
  .fs-modal-actions {{ display: flex; gap: 10px; align-items: center; }}
  .fs-btn {{ background: #1e293b; color: #f8fafc; border: 1px solid #475569; border-radius: 7px; padding: 6px 13px; font-size: 12.5px; font-weight: 700; cursor: pointer; }}
  .fs-btn:hover {{ border-color: #22d3ee; }}
  .fs-btn-close {{ background: #be123c; border-color: #fb7185; }}
  .fs-modal-body {{ flex: 1; width: 100%; border: 0; background: #020617; }}
</style>
<div id="agyFullscreenModal" class="fs-modal" role="dialog" aria-modal="true">
  <div class="fs-modal-bar">
    <span id="agyModalTitle" class="fs-modal-title">{spec['modalDefaultTitle']}</span>
    <div class="fs-modal-actions">
      <button type="button" id="agyModalDownloadBtn" class="fs-btn" style="display:none;">{spec['modalSaveBtn']}</button>
      <button type="button" id="agyModalCloseBtn" class="fs-btn fs-btn-close">{spec['modalCloseBtn']}</button>
    </div>
  </div>
  <iframe id="agyModalFrame" class="fs-modal-body" title="{spec['modalFrameTitle']}"></iframe>
</div>
<script>
(function() {{
  "use strict";
  const PAYLOADS = {modalConfigJson};

  async function gunzipBase64ToText(b64Str) {{
    const binStr = atob(b64Str);
    const bytes = new Uint8Array(binStr.length);
    for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);
    const ds = new DecompressionStream("gzip");
    const stream = new Blob([bytes]).stream().pipeThrough(ds);
    return await new Response(stream).text();
  }}

  const modal = document.getElementById("agyFullscreenModal");
  const modalTitle = document.getElementById("agyModalTitle");
  const modalFrame = document.getElementById("agyModalFrame");
  const downloadBtn = document.getElementById("agyModalDownloadBtn");
  let currentDownloadInfo = null;

  function closeModal() {{
    modal.classList.remove("open");
    modalFrame.removeAttribute("srcdoc");
    currentDownloadInfo = null;
  }}

  document.getElementById("agyModalCloseBtn").addEventListener("click", closeModal);
  window.addEventListener("keydown", function(ev) {{
    if (ev.key === "Escape" && modal.classList.contains("open")) closeModal();
  }});

  downloadBtn.addEventListener("click", async function() {{
    if (!currentDownloadInfo) return;
    const svgText = await gunzipBase64ToText(currentDownloadInfo.rawGz);
    const blob = new Blob([svgText], {{ type: "image/svg+xml;charset=utf-8" }});
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = currentDownloadInfo.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(function() {{ URL.revokeObjectURL(url); }}, 1500);
  }});

  document.getElementById("btnOpenMainSvgModal").addEventListener("click", async function() {{
    modalTitle.textContent = PAYLOADS.titleMain;
    downloadBtn.style.display = "inline-block";
    currentDownloadInfo = {{ rawGz: PAYLOADS.mainRawGz, filename: PAYLOADS.mainFile }};
    modal.classList.add("open");
    modalFrame.srcdoc = await gunzipBase64ToText(PAYLOADS.mainViewerGz);
  }});

  document.getElementById("btnOpenSlidesSvgModal").addEventListener("click", async function() {{
    modalTitle.textContent = PAYLOADS.titleSlides;
    downloadBtn.style.display = "inline-block";
    currentDownloadInfo = {{ rawGz: PAYLOADS.slidesRawGz, filename: PAYLOADS.slidesFile }};
    modal.classList.add("open");
    modalFrame.srcdoc = await gunzipBase64ToText(PAYLOADS.slidesViewerGz);
  }});

  document.getElementById("btnOpen3dModal").addEventListener("click", function() {{
    modalTitle.textContent = PAYLOADS.title3d;
    downloadBtn.style.display = "none";
    currentDownloadInfo = null;
    modal.classList.add("open");
    modalFrame.srcdoc = document.getElementById("embedded3dFrame").getAttribute("srcdoc");
  }});
}})();
</script>
</body>
"""
    standaloneHtml = standaloneHtml.replace("</body>", modalInjection)

    outPath = Path(spec["outPath"])
    outPath.write_text(standaloneHtml, encoding="utf-8")
    logger.info(
        "Built standalone single-file HTML report (%s): %s (%.2f MB)",
        lang,
        outPath,
        outPath.stat().st_size / (1024.0 * 1024.0),
    )
    return outPath


def buildStandaloneReportHtml(projectDir: Path) -> tuple[Path, Path]:
    """
    Build both Chinese (-zh.html) and English (-en.html) standalone single-file HTML bundles.
    """
    zhOut = buildSingleStandaloneReport(projectDir, lang="zh")
    enOut = buildSingleStandaloneReport(projectDir, lang="en")
    return zhOut, enOut


def main() -> None:
    """
    CLI entry point generating both Chinese and English single-file standalone HTML reports.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    buildStandaloneReportHtml(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
