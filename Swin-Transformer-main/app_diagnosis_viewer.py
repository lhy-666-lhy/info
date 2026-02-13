"""
Cell AI Diagnosis Viewer - Streamlit
Displays three analysis images and the Gemma diagnosis report.
"""
from __future__ import annotations

import streamlit as st
from pathlib import Path

# Default results directory (evaluation_results under script dir)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "evaluation_results"

# Page config
st.set_page_config(
    page_title="Cell AI Diagnosis",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Cell AI Diagnosis Viewer")

# Scan available diagnosis result directories
def get_result_folders(base_dir: Path) -> list[tuple[str, Path]]:
    """Return (display_name, path) list, sorted by mtime descending."""
    if not base_dir.exists():
        return []
    folders = []
    for p in base_dir.iterdir():
        if p.is_dir() and (p / "1_original.png").exists():
            folders.append((p.name, p))
    folders.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
    return folders


results_dir = st.sidebar.text_input(
    "Results root directory",
    value=str(DEFAULT_RESULTS_DIR),
    help="Directory containing evaluation_results",
)
base = Path(results_dir)
folders = get_result_folders(base)

if not folders:
    st.warning(
        f"No diagnosis result directories found in `{results_dir}`.\n\n"
        "Please run `bash dignose.sh <image_path>` first to generate analysis results."
    )
    st.stop()

# Sidebar: select result
selected_name = st.sidebar.selectbox(
    "Select diagnosis result",
    options=[f[0] for f in folders],
    format_func=lambda x: x,
)
selected_path = next(p for n, p in folders if n == selected_name)

# Load three images
img1 = selected_path / "1_original.png"
img2 = selected_path / "2_gradcam_overlay.png"
img3 = selected_path / "3_attention_overlay.png"

# Display three images
st.subheader(f"📷 Analysis images: {selected_name}")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**1. Original cell image**")
    if img1.exists():
        st.image(str(img1), use_container_width=True)
    else:
        st.error("1_original.png not found")

with col2:
    st.markdown("**2. Grad-CAM heatmap overlay**")
    if img2.exists():
        st.image(str(img2), use_container_width=True)
    else:
        st.error("2_gradcam_overlay.png not found")

with col3:
    st.markdown("**3. Attention visualization overlay**")
    if img3.exists():
        st.image(str(img3), use_container_width=True)
    else:
        st.error("3_attention_overlay.png not found")

# Display diagnosis report
st.subheader("📋 Diagnosis report")
report_file = selected_path / "gemma_diagnosis.md"
if report_file.exists():
    report_text = report_file.read_text(encoding="utf-8")
    st.markdown(report_text)
else:
    st.info("Gemma diagnosis report (gemma_diagnosis.md) not found. Please install ollama and pull gemma3:12b, then re-run dignose.sh.")
