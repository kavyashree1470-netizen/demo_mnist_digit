"""
MNIST Digit Collector — Streamlit App
Uses streamlit-drawable-canvas for real-time drawing with native data bridge.
Converts drawings to 28×28 MNIST-standard grayscale and saves to CSV.
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import base64
from datetime import datetime
from io import BytesIO

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST Digit Collector",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark theme */
    .stApp { background-color: #0f0f1a; color: #e0e0e0; }
    .main .block-container { padding: 1.2rem 2rem 2rem; max-width: 1400px; }
    h1, h2, h3 { color: #c084fc; font-family: 'Segoe UI', sans-serif; }
    h3 { font-size: 1.05rem; margin-bottom: 0.4rem; }

    /* Metric cards */
    .metric-card {
        background: #1e1e2e; border: 1px solid #3b3b5c; border-radius: 12px;
        padding: 12px 16px; text-align: center; margin: 4px 0;
    }
    .metric-card .value { font-size: 1.7rem; font-weight: 700; color: #c084fc; }
    .metric-card .label { font-size: 0.73rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }

    /* MNIST pixel grid */
    .mnist-grid {
        display: grid; grid-template-columns: repeat(28, 9px); gap: 0;
        border: 2px solid #3b3b5c; border-radius: 4px;
        width: fit-content; margin: 0 auto;
    }
    .mnist-cell { width: 9px; height: 9px; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: white; border: none; border-radius: 8px;
        padding: 9px 20px; font-weight: 600; font-size: 0.9rem;
        transition: all 0.2s; width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(168,85,247,0.4);
    }
    div[data-testid="stButton"]:has(button.clear-btn) button {
        background: #2d2d3e !important;
        color: #aaa !important;
    }

    /* Info badges */
    .info-badge {
        display: inline-block; background: #1e1040; color: #c084fc;
        border: 1px solid #6d28d9; border-radius: 20px;
        padding: 2px 10px; font-size: 0.75rem; margin: 2px;
    }

    /* Instructions box */
    .instructions {
        background: #1a1a2e; border-left: 3px solid #7c3aed;
        border-radius: 0 8px 8px 0; padding: 10px 14px;
        font-size: 0.83rem; color: #bbb; margin-bottom: 10px; line-height: 1.6;
    }

    /* Success / warning banners */
    .banner-success {
        background: #052e16; border: 1px solid #16a34a; border-radius: 8px;
        padding: 12px 16px; color: #86efac; font-size: 0.88rem;
    }
    .banner-warn {
        background: #2d1a00; border: 1px solid #d97706; border-radius: 8px;
        padding: 10px 14px; color: #fde68a; font-size: 0.85rem;
    }

    /* Pixel value pre */
    .pixel-pre {
        font-family: 'Courier New', monospace; font-size: 0.52rem;
        line-height: 1.15; background: #0a0a14; color: #777;
        padding: 8px; border-radius: 6px; overflow-x: auto;
        border: 1px solid #2a2a3e;
    }

    /* Canvas wrapper — dark background */
    canvas { border-radius: 8px !important; }
    div[data-testid="stCanvasComponent"] > div { border-radius: 10px; overflow: hidden; }

    /* Sidebar override */
    section[data-testid="stSidebar"] { background: #12121f; }

    /* Selectbox / radio */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stRadio"] label { color: #a0aec0 !important; font-size: 0.85rem; }

    /* Divider */
    hr { border-color: #2a2a3e; margin: 0.8rem 0; }

    /* Dataframe */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── constants ──────────────────────────────────────────────────────────────────
CSV_FILE  = "mnist_dataset.csv"
CANVAS_SZ = 280      # px – display canvas
STROKE_W  = 18       # pen thickness
STROKE_CLR = "#FFFFFF"

# ── core image processing ──────────────────────────────────────────────────────

def image_to_mnist(rgba):

    # Convert RGB to grayscale
    rgb = rgba[:, :, :3]

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    # White strokes on black background
    stroke = gray.astype(np.float32)

    # Ignore very dark pixels
    stroke[stroke < 20] = 0

    if np.max(stroke) == 0:
        return None

    # Anti-alias
    stroke = cv2.GaussianBlur(
        stroke,
        (5,5),
        1
    )

    # 3. bounding box + padding
    mask = (stroke > 20).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    pad = max(int(max(w, h) * 0.20), 3)
    x1 = max(0, x - pad);       y1 = max(0, y - pad)
    x2 = min(stroke.shape[1], x + w + pad)
    y2 = min(stroke.shape[0], y + h + pad)
    cropped = stroke[y1:y2, x1:x2]
    if cropped.size == 0:
        return None

    # 4. resize → 20×20
    d20 = cv2.resize(cropped, (20, 20), interpolation=cv2.INTER_AREA)
    d20 = np.clip(d20, 0, 255)
    d20 = cv2.normalize(d20, d20, 0, 255, cv2.NORM_MINMAX)

    # 5. centre-of-mass centering
    M = cv2.moments(d20)
    out = np.zeros((28, 28), dtype=np.float32)
    cx  = int(M["m10"] / M["m00"]) if M["m00"] else 10
    cy  = int(M["m01"] / M["m00"]) if M["m00"] else 10
    ox, oy = 14 - cx, 14 - cy
    for r in range(20):
        for c in range(20):
            nr, nc = r + oy, c + ox
            if 0 <= nr < 28 and 0 <= nc < 28:
                out[nr, nc] = d20[r, c]

    # 6. finalise
    return out.astype(np.uint8)


# ── display helpers ────────────────────────────────────────────────────────────

def mnist_html_grid(arr: np.ndarray) -> str:
    cells = ""
    for row in arr:
        for v in row:
            v = int(v)
            if v > 10:
                r = min(255, 80  + int(v * 0.70))
                g = min(255,       int(v * 0.35))
                b = min(255, 120 + int(v * 0.50))
                color = f"#{r:02x}{g:02x}{b:02x}"
            else:
                color = "#0a0a14"
            cells += f'<div class="mnist-cell" style="background:{color}"></div>'
    return f'<div class="mnist-grid">{cells}</div>'


def pixel_table(arr: np.ndarray) -> str:
    return "\n".join(" ".join(f"{v:3d}" for v in row) for row in arr)


def arr_to_png_bytes(arr: np.ndarray, scale: int = 10) -> bytes:
    big = cv2.resize(arr, (28 * scale, 28 * scale), interpolation=cv2.INTER_NEAREST)
    _, buf = cv2.imencode(".png", big)
    return buf.tobytes()


# ── CSV helpers ────────────────────────────────────────────────────────────────
PIXEL_COLS = [f"pixel_{i}" for i in range(784)]

def init_csv():
    if not os.path.exists(CSV_FILE):
        pd.DataFrame(
            columns=[
               "id",
               "username",
               "label",
               "timestamp",
               "split"
              ] + PIXEL_COLS
                     ).to_csv(CSV_FILE, index=False)

def load_csv() -> pd.DataFrame:
    init_csv()
    return pd.read_csv(CSV_FILE)

def save_sample(
        username: str,
        label: int,
        pixels: np.ndarray,
        split: str):
    df  = load_csv()
    sid = int(df["id"].max()) + 1 if len(df) else 1
    row = {
    "id": sid,
    "username": username,
    "label": label,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "split": split
}
    row.update({f"pixel_{i}": int(v) for i, v in enumerate(pixels.flatten())})
    pd.DataFrame([row]).to_csv(CSV_FILE, mode="a", header=False, index=False)
    return sid


# ── session state ──────────────────────────────────────────────────────────────
for k, v in [("mnist_arr", None), ("canvas_key", 0),
             ("last_id", None), ("save_msg", ""), ("warn_msg", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

init_csv()

# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# ✏️ MNIST Digit Collector")
st.markdown(
    '<span class="info-badge">28×28 px</span>'
    '<span class="info-badge">Grayscale 0-255</span>'
    '<span class="info-badge">Anti-aliased</span>'
    '<span class="info-badge">CoM-centred</span>'
    '<span class="info-badge">MNIST standard</span>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  THREE-COLUMN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
col_draw, col_preview, col_save = st.columns([2, 1.5, 1.3], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
#  LEFT — Drawing canvas
# ─────────────────────────────────────────────────────────────────────────────
with col_draw:
    st.markdown("### 🖊️ Draw a Digit (0–9)")
    st.markdown("""
    <div class="instructions">
        • Draw <b>one digit</b> on the black canvas.<br>
        • Use <b>thick, deliberate strokes</b> — similar to MNIST source digits.<br>
        • The conversion updates <b>live</b> as you draw.<br>
        • Click <b>🗑 Clear</b> to erase and start again.
    </div>
    """, unsafe_allow_html=True)

    # ── streamlit-drawable-canvas ──────────────────────────────────────────
    canvas_result = st_canvas(
        fill_color   = "rgba(0,0,0,0)",
        stroke_width = STROKE_W,
        stroke_color = STROKE_CLR,
        background_color = "#000000",
        height       = CANVAS_SZ,
        width        = CANVAS_SZ,
        drawing_mode = "freedraw",
        key          = f"canvas_{st.session_state.canvas_key}",
        update_streamlit = True,          # live updates on every stroke
        display_toolbar  = False,         # hide built-in toolbar (we add our own)
    )

    # ── process canvas data ────────────────────────────────────────────────
    if canvas_result.image_data is not None:

     rgba = canvas_result.image_data.astype(np.uint8)

     # Check whether the user has actually drawn something.
     gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)

     has_drawing = (
        np.max(gray) > 5 or
        np.max(rgba[:, :, 3]) > 5
     )

     if has_drawing:

        result = image_to_mnist(rgba)

        if result is not None:
            st.session_state.mnist_arr = result
            st.session_state.warn_msg = ""

     else:

        st.session_state.mnist_arr = None

    # ── clear button ───────────────────────────────────────────────────────
    if st.button("🗑️  Clear Canvas", key="clear_btn", use_container_width=True):
        st.session_state.canvas_key  += 1
        st.session_state.mnist_arr   = None
        st.session_state.save_msg    = ""
        st.session_state.warn_msg    = ""
        st.rerun()

    # ── OR: upload image ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Or upload an existing digit image:**")
    uploaded = st.file_uploader(
        "Upload PNG/JPG", type=["png","jpg","jpeg"],
        label_visibility="collapsed", key="uploader"
    )
    if uploaded:
        img  = Image.open(uploaded).convert("RGBA")
        rgba = np.array(img)
        res  = image_to_mnist(rgba)
        if res is not None:
            st.session_state.mnist_arr = res
            st.session_state.warn_msg  = ""
            st.success("✅ Uploaded image converted to MNIST format.")
        else:
            st.error("Could not detect a digit in the uploaded image.")

# ─────────────────────────────────────────────────────────────────────────────
#  MIDDLE — MNIST preview
# ─────────────────────────────────────────────────────────────────────────────
with col_preview:
    st.markdown("### 🔍 MNIST Preview")

    arr = st.session_state.mnist_arr

    if arr is not None:
        # Pixel grid
        st.markdown("**28×28 pixel grid:**")
        st.markdown(mnist_html_grid(arr), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Stats
        nz   = int(np.count_nonzero(arr))
        mean = float(arr.mean())
        mx   = int(arr.max())

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="value">{nz}</div>'
                        f'<div class="label">Active px</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="value">{mean:.1f}</div>'
                        f'<div class="label">Mean val</div></div>', unsafe_allow_html=True)

        # Show MNIST image via PIL
        pil_big = Image.fromarray(
            cv2.resize(arr, (280, 280), interpolation=cv2.INTER_NEAREST)
        )
        st.image(
       arr,
       width=280,
       clamp=True,
       caption="28×28 MNIST (10×)"
    )

        # Download PNG
        st.download_button(
            "⬇️ Download 28×28 PNG",
            data=arr_to_png_bytes(arr, scale=1),
            file_name="digit_28x28.png",
            mime="image/png",
            use_container_width=True,
        )

        # Raw pixel values
        with st.expander("🔢 Raw pixel values (28×28)"):
            st.markdown(
                f'<div class="pixel-pre"><pre>{pixel_table(arr)}</pre></div>',
                unsafe_allow_html=True,
            )

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 10px;color:#444;
                    border:2px dashed #2a2a3e;border-radius:12px;margin-top:8px;">
            <div style="font-size:2.5rem;">🖼️</div>
            <div style="margin-top:8px;font-size:0.85rem;">
                MNIST preview appears<br>as you draw
            </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  RIGHT — Label, split, save + stats
# ─────────────────────────────────────────────────────────────────────────────
with col_save:
    st.markdown("### 💾 Label & Save")

    username = st.text_input(
        "👤 User Name",
        value="",
        placeholder="Enter your name",
        key="username"
    )

    label = st.selectbox(
        "Digit label (0–9)",
        options=list(range(10)),
        key="label_sel",
    )

    split = st.radio(
        "Dataset split",
        options=["train", "test", "validation"],
        key="split_radio",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Save only when the button is pressed
    if st.button("💾 Save Sample", use_container_width=True):

        if username.strip() == "":
            st.session_state.warn_msg = "⚠ Please enter your user name."

        elif st.session_state.mnist_arr is None:
            st.session_state.warn_msg = "⚠ Draw a digit first."

        else:
            sid = save_sample(
                username=username.strip(),
                label=label,
                pixels=st.session_state.mnist_arr,
                split=split,
            )

            st.session_state.save_msg = (
                f"✅ Sample #{sid} saved successfully.<br>"
                f"👤 User: <b>{username}</b><br>"
                f"🔢 Digit: <b>{label}</b>"
            )

            st.session_state.warn_msg = ""


            # Optional: clear the canvas after saving
            st.session_state.canvas_key += 1
            st.session_state.mnist_arr = None

            st.rerun()

    if st.session_state.save_msg:
        st.markdown(
            f'<div class="banner-success">{st.session_state.save_msg}</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.warn_msg:
        st.markdown(
            f'<div class="banner-warn">{st.session_state.warn_msg}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📊 Dataset Stats")

    df    = load_csv()
    total = len(df)
    st.markdown(
        f'<div class="metric-card"><div class="value">{total}</div>'
        f'<div class="label">Total Samples</div></div>',
        unsafe_allow_html=True,
    )

    if total > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        for sp in ["train", "test", "validation"]:
            cnt = len(df[df["split"] == sp])
            if cnt:
                pct = cnt / total * 100
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:5px 2px;border-bottom:1px solid #2a2a3e;font-size:0.82rem;">'
                    f'<span style="color:#a0aec0">{sp}</span>'
                    f'<span style="color:#c084fc;font-weight:600">{cnt}'
                    f' <span style="color:#555;font-size:0.72rem">({pct:.0f}%)</span></span></div>',
                    unsafe_allow_html=True,
                )

        # Per-digit bar chart
        st.markdown("<br>**Per digit:**", unsafe_allow_html=True)
        lc  = df["label"].value_counts().sort_index()
        mx2 = lc.max()
        for digit, count in lc.items():
            bw = int(count / mx2 * 100)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px;'
                f'margin:3px 0;font-size:0.8rem;">'
                f'<span style="color:#c084fc;width:12px;font-weight:700">{digit}</span>'
                f'<div style="flex:1;background:#1e1e2e;border-radius:3px;height:9px;">'
                f'<div style="width:{bw}%;background:linear-gradient(90deg,#7c3aed,#a855f7);'
                f'height:9px;border-radius:3px"></div></div>'
                f'<span style="color:#666;width:18px;text-align:right">{count}</span></div>',
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
#  BOTTOM — Dataset table + download
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📋 Dataset Records")

df = load_csv()
c1, c2, c3 = st.columns([2, 2, 1])

with c1:
    if len(df):
        st.download_button(
            "⬇️ Download Full CSV",
            data=df.to_csv(index=False).encode(),
            file_name="mnist_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

with c2:
    filt = st.multiselect("Filter by digit label", options=list(range(10)), default=[])

with c3:
    if st.button("🗑️ Clear ALL data", use_container_width=True):
        if os.path.exists(CSV_FILE):
            os.remove(CSV_FILE)
        init_csv()
        st.session_state.save_msg = ""
        st.rerun()

if len(df):
    disp = df[
    [
        "id",
        "username",
        "label",
        "timestamp",
        "split"
    ]
]
    if filt:
        disp = disp[disp["label"].isin(filt)]
    st.dataframe(
        disp.sort_values("id", ascending=False).head(100),
        use_container_width=True,
        hide_index=True,
        column_config={
            "id":        st.column_config.NumberColumn("ID",       width="small"),
            "label":     st.column_config.NumberColumn("Digit",    width="small"),
            "timestamp": st.column_config.TextColumn("Timestamp"),
            "split":     st.column_config.TextColumn("Split",      width="small"),
        },
    )
    st.caption(f"Showing last 100 of {len(df)} records · {len(disp)} after filter")
else:
    st.info("No samples saved yet — draw and save digits above.")

# ── footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#444;font-size:0.78rem;">'
    'MNIST Digit Collector · 28×28 · Grayscale 0-255 · Anti-aliased · Centre-of-mass centred'
    '</div>',
    unsafe_allow_html=True,
)
