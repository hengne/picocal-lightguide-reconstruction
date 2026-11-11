#!/usr/bin/env bash
# ==============================================================================
# LHCb PicoCal Conical Light-Guide Simulation & Inverse Reconstruction Suite
# End-to-End Cross-Platform Reproduction Pipeline
#
# Author: Hengne Li <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
#
# Usage:
#   bash run-all.sh [pipeline-flags...]
#
# Examples:
#   bash run-all.sh
#   bash run-all.sh --skip-geant4
#   bash run-all.sh --scan-csv ./data/scan_summary.csv --skip-geant4
# ==============================================================================

set -euo pipefail

# Resolve repository root directory regardless of caller working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ------------------------------------------------------------------------------
# 1. Auto-detect Python 3 interpreter
# ------------------------------------------------------------------------------
resolve_python() {
    if [[ -n "${PYTHON_BIN:-}" ]] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "${PYTHON_BIN}"
        return 0
    fi
    for candidate in python3 python; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            if "${candidate}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then
                echo "${candidate}"
                return 0
            fi
        fi
    done
    return 1
}

if ! PYTHON_CMD="$(resolve_python)"; then
    echo "[ERROR] Python >= 3.10 was not found on PATH (checked 'python3' and 'python')." >&2
    echo "        Please activate your virtual/conda environment or set PYTHON_BIN=/path/to/python3." >&2
    exit 1
fi

echo "=============================================================================="
echo " LHCb PicoCal Light-Guide Simulation & Inverse Reconstruction Pipeline"
echo " Working directory : ${SCRIPT_DIR}"
echo " Python interpreter: ${PYTHON_CMD} ($(${PYTHON_CMD} --version 2>&1))"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# 2. Compile native hardware ray-tracing accelerators (Metal GPU / C++17 CPU)
# ------------------------------------------------------------------------------
echo ""
echo "[1/5] Compiling native ray-tracing accelerators (compile-accelerators.py)..."
"${PYTHON_CMD}" compile-accelerators.py

# ------------------------------------------------------------------------------
# 3. Run simulation, B-spline inverse reconstruction, validation & master SVGs
# ------------------------------------------------------------------------------
echo ""
echo "[2/5] Running simulation & inverse reconstruction pipeline (run-simulation-pipeline.py $*)..."
"${PYTHON_CMD}" run-simulation-pipeline.py "$@"

# ------------------------------------------------------------------------------
# 4. Build bilingual presentation SVGs, 3D photon simulator & web portals
# ------------------------------------------------------------------------------
echo ""
echo "[3/5] Building interactive 3D simulator, animated SVGs & web portals (build-presentation-suite.py)..."
"${PYTHON_CMD}" build-presentation-suite.py

# ------------------------------------------------------------------------------
# 5. Build zero-dependency single-file standalone HTML report bundles
# ------------------------------------------------------------------------------
echo ""
echo "[4/5] Building zero-dependency standalone HTML bundles (build-standalone-report.py)..."
"${PYTHON_CMD}" build-standalone-report.py

# ------------------------------------------------------------------------------
# 6. Build bilingual PDF user manuals if WeasyPrint is installed
# ------------------------------------------------------------------------------
echo ""
echo "[5/5] Checking WeasyPrint availability for PDF user manuals (build-user-manuals.py)..."
if "${PYTHON_CMD}" -c "import weasyprint" >/dev/null 2>&1; then
    "${PYTHON_CMD}" build-user-manuals.py
else
    echo "[INFO] Optional package 'weasyprint' is not installed (or system Pango/Cairo libraries are unavailable)."
    echo "       Skipping PDF regeneration; pre-built manuals remain available in docs/user-manual-{en,zh}.pdf."
fi

# ------------------------------------------------------------------------------
# 7. Print clean summary of generated deliverables
# ------------------------------------------------------------------------------
print_artifact_status() {
    local rel_path="$1"
    local label="$2"
    if [[ -f "${SCRIPT_DIR}/${rel_path}" ]]; then
        local size_hr
        size_hr="$(du -h "${SCRIPT_DIR}/${rel_path}" | awk '{print $1}')"
        printf "  [OK] %-56s (%s) - %s\n" "${rel_path}" "${size_hr}" "${label}"
    else
        printf "  [--] %-56s (not present) - %s\n" "${rel_path}" "${label}"
    fi
}

echo ""
echo "=============================================================================="
echo " Summary of Generated Deliverables"
echo "=============================================================================="
echo "Web Portals & Standalone Bundles:"
print_artifact_status "index.html" "Chinese Interactive Portal"
print_artifact_status "index-en.html" "English Interactive Portal"
print_artifact_status "lhcb-picocal-light-guide-report-standalone-zh.html" "Standalone Bundle (ZH)"
print_artifact_status "lhcb-picocal-light-guide-report-standalone-en.html" "Standalone Bundle (EN)"

echo ""
echo "Interactive 3D Simulators & Vector SVG Reports:"
print_artifact_status "light-guide-imperfection-simulation.svg" "Master Zoomable SVG Report (ZH)"
print_artifact_status "light-guide-imperfection-simulation-en.svg" "Master Zoomable SVG Report (EN)"
print_artifact_status "presentation/interactive-3d-photon-simulator.html" "3D Photon & PMT Bench (ZH)"
print_artifact_status "presentation/interactive-3d-photon-simulator-en.html" "3D Photon & PMT Bench (EN)"
print_artifact_status "presentation/photon-propagation-animated.svg" "Animated Escape SVG (ZH)"
print_artifact_status "presentation/photon-propagation-animated-en.svg" "Animated Escape SVG (EN)"
print_artifact_status "presentation/research-summary-slides.svg" "1920x1080 Summary Slides (ZH)"
print_artifact_status "presentation/research-summary-slides-en.svg" "1920x1080 Summary Slides (EN)"

echo ""
echo "Reconstructed 3D CAD Mesh & Quantitative Data:"
print_artifact_status "outputs/reconstructed-light-guide.stl" "Watertight 3D CAD STL Mesh"
print_artifact_status "outputs/reconstructed-surface-grid.csv" "Unwrapped Surface Grid CSV"
print_artifact_status "outputs/simulated-scan-summary.csv" "100x100 Simulated Scan CSV"
print_artifact_status "outputs/geant4-slice-validation.csv" "Geant4 Optical Slice CSV"
print_artifact_status "outputs/inversion-metrics.json" "Calibration & Fit Metrics JSON"

echo ""
echo "Bilingual PDF User Manuals:"
print_artifact_status "docs/user-manual-en.pdf" "English User Manual (PDF)"
print_artifact_status "docs/user-manual-zh.pdf" "Chinese User Manual (PDF)"
echo "=============================================================================="
