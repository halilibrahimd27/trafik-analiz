"""
Trafik Isareti Tanima - Streamlit Web Arayuzu
Kullanim: python3 -m streamlit run src/app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import pandas as pd

from src.config import MODEL_PATH, IMG_SIZE, CLASS_NAMES, NUM_CLASSES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_DIR = os.path.join(DATA_DIR, "Test")

st.set_page_config(page_title="Trafik Isareti Tanima", page_icon="🚦", layout="wide")

# -- CSS -------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #0a0f1e 0%, #111827 100%); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #0d1321; border-right: 1px solid #1e293b; }

/* Header */
.hero { text-align: center; padding: 1.5rem 1rem 0.5rem; }
.hero h1 {
    font-size: 2.4rem; font-weight: 900; margin-bottom: 0.2rem;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #64748b; font-size: 0.9rem; margin: 0; }
.stats-row {
    display: flex; justify-content: center; gap: 24px; flex-wrap: wrap;
    margin: 1rem 0 1.2rem;
}
.stat-item { text-align: center; }
.stat-val { color: #e2e8f0; font-size: 1.3rem; font-weight: 700; }
.stat-label { color: #475569; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.5px; }

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed #334155 !important;
    border-radius: 16px !important;
    background: #111827 !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #3b82f6 !important; }
.upload-hint { text-align: center; color: #475569; font-size: 0.78rem; margin-top: 2px; }

/* Result card */
.result-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155; border-radius: 20px;
    padding: 2rem; text-align: center; position: relative; overflow: hidden;
}
.result-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
}
.result-icon { font-size: 3rem; margin-bottom: 0.3rem; }
.result-name { font-size: 1.5rem; font-weight: 800; color: #f1f5f9; margin: 0.3rem 0; line-height: 1.3; }

/* Confidence gauge */
.gauge-wrap { margin: 0.8rem auto; max-width: 260px; }
.gauge-track { background: #1e293b; border-radius: 8px; height: 10px; overflow: hidden; }
.gauge-fill { height: 100%; border-radius: 8px; transition: width 0.6s ease; }
.gauge-label { display: flex; justify-content: space-between; margin-top: 4px; }
.gauge-pct { font-weight: 700; font-size: 1.1rem; }
.gauge-text { color: #64748b; font-size: 0.78rem; }
.g-hi  { color: #4ade80; }
.g-mid { color: #facc15; }
.g-lo  { color: #f87171; }

/* Bar chart */
.bars { background: #0c1525; border-radius: 14px; padding: 1rem 1.2rem; margin-top: 0.6rem; }
.bar-row { display: flex; align-items: center; gap: 8px; margin: 5px 0; }
.bar-lbl { color: #cbd5e1; font-size: 0.8rem; width: 180px; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-track { flex: 1; background: #1e293b; border-radius: 5px; height: 12px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 5px; transition: width 0.4s; }
.bar-pct { color: #64748b; font-size: 0.76rem; width: 38px; text-align: right; flex-shrink: 0; }

/* Sidebar */
.cat-header {
    color: #60a5fa; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 1px; text-transform: uppercase;
    border-left: 3px solid #3b82f6; padding-left: 8px;
    margin: 0.8rem 0 0.3rem;
}
[data-testid="stSidebar"] button[kind="secondary"] {
    background: #131c30 !important; border: 1px solid #1e2d48 !important;
    color: #8b9cc0 !important; border-radius: 8px !important;
    font-size: 0.76rem !important; text-align: left !important;
    padding: 4px 10px !important; margin: 1px 0 !important;
    transition: all 0.15s;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: #1e2d4a !important; color: #e2e8f0 !important;
    border-color: #3b82f6 !important; transform: translateX(3px);
}

/* Example grid */
.ex-name {
    text-align: center; color: #475569; font-size: 0.65rem; margin-top: 2px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* Info panel */
.info-panel {
    background: #111827; border: 1px solid #1e293b; border-radius: 14px;
    padding: 1.2rem; margin-top: 1rem;
}
.info-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1e293b; }
.info-key { color: #64748b; font-size: 0.82rem; }
.info-val { color: #e2e8f0; font-size: 0.82rem; font-weight: 600; }

/* Tabs */
button[data-baseweb="tab"] { color: #94a3b8 !important; font-weight: 600 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #60a5fa !important; }

.empty-state { text-align: center; padding: 3rem; }
.footer { text-align: center; color: #1e293b; font-size: 0.75rem; padding: 1rem 0 0.3rem; }
</style>
""", unsafe_allow_html=True)


# -- Model / helpers -------------------------------------------------------

@st.cache_resource(show_spinner="Model yukleniyor...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def get_class_image_map():
    csv_path = os.path.join(DATA_DIR, "Test.csv")
    if not os.path.isfile(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        cid = int(row["ClassId"])
        if cid not in mapping:
            mapping[cid] = os.path.join(DATA_DIR, str(row["Path"]))
    return mapping


def apply_clahe_single(img_np):
    """CLAHE kontrast iyilestirme (opsiyonel, cv2 varsa)."""
    try:
        import cv2
        img_uint8 = (img_np * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced.astype(np.float32) / 255.0
    except ImportError:
        return img_np


def get_icon(name):
    if "Hiz Limiti" in name or "Hız Limiti" in name: return "🚗"
    if "Dur" in name:            return "🛑"
    if "Giremez" in name:        return "⛔"
    if "Gecis Yasagi" in name or "Geçiş Yasağı" in name: return "🚫"
    if "Yasak" in name and "Ton" not in name and "Sonu" not in name: return "🚫"
    if "Ton" in name:            return "🚛"
    if any(k in name for k in ["Tehlike", "Viraj", "Kaygan", "Buzlanma", "Engebeli"]): return "⚠️"
    if any(k in name for k in ["Dönünüz", "Gidiniz", "Geçiniz", "İleri"]): return "↗️"
    if "Yaya" in name:           return "🚶"
    if "Cocuk" in name or "Çocuk" in name: return "👶"
    if "Bisiklet" in name:       return "🚲"
    if "Kavşak" in name or "Dönel" in name: return "🔄"
    if "Öncelikli" in name:      return "💛"
    if "Yol Ver" in name:        return "🔺"
    if "Hayvan" in name:         return "🦌"
    if "Çalışma" in name:        return "🔧"
    if "Işık" in name:           return "🚦"
    if "Sonu" in name:           return "✅"
    return "🚦"


BAR_COLORS = ["#3b82f6", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b"]

CATEGORIES = [
    ("🚗 Hiz Limitleri",      [0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ("🚫 Yasaklayici",        [9, 10, 15, 16, 17]),
    ("🛑 Dur / Yol Ver",      [13, 14]),
    ("⚠️ Tehlike Uyarilari",  [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]),
    ("↗️ Yon & Zorunlu",      [33, 34, 35, 36, 37, 38, 39, 40]),
    ("✅ Yasaklarin Sonu",    [32, 41, 42]),
    ("💛 Oncelik",            [12]),
]


def run_predict(image):
    model = load_model()
    img = image.convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]))
    x = np.array(img, dtype=np.float32) / 255.0
    x = apply_clahe_single(x)
    probs = model.predict(x[np.newaxis], verbose=0)[0]
    top5 = np.argsort(probs)[::-1][:5]
    return probs, top5


class_image_map = get_class_image_map()

# -- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🗂️ Desteklenen Levhalar")
    st.markdown("<small style='color:#3b5278'>Tikla → ornek yukle</small>", unsafe_allow_html=True)

    for cat_name, ids in CATEGORIES:
        st.markdown(f"<div class='cat-header'>{cat_name}</div>", unsafe_allow_html=True)
        for cid in ids:
            if cid not in CLASS_NAMES:
                continue
            name = CLASS_NAMES[cid]
            if st.button(f"{get_icon(name)} {name}", key=f"sb_{cid}", use_container_width=True):
                if cid in class_image_map and os.path.isfile(class_image_map[cid]):
                    st.session_state["active_image"] = class_image_map[cid]
                    st.session_state["active_source"] = name

# -- Header ----------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🚦 Trafik Isareti Tanima</h1>
  <p>Fotografinizi yukleyin — yapay zeka aninda tanisin</p>
</div>
<div class="stats-row">
  <div class="stat-item"><div class="stat-val">43</div><div class="stat-label">Sinif</div></div>
  <div class="stat-item"><div class="stat-val">%97.6</div><div class="stat-label">Dogruluk</div></div>
  <div class="stat-item"><div class="stat-val">51K+</div><div class="stat-label">Egitim Ornegi</div></div>
  <div class="stat-item"><div class="stat-val">CNN v2</div><div class="stat-label">Model</div></div>
</div>
""", unsafe_allow_html=True)

# -- Tabs ------------------------------------------------------------------
tab_predict, tab_about = st.tabs(["📸 Tani", "ℹ️ Model Bilgisi"])

with tab_predict:
    upload_col, example_col = st.columns([2, 3], gap="large")

    with upload_col:
        uploaded = st.file_uploader(
            "Fotograf yukle",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )
        st.markdown('<p class="upload-hint">JPG · PNG · BMP · WebP</p>', unsafe_allow_html=True)

    with example_col:
        st.markdown("**Hazir ornekler:**")
        ex_nums = [0, 10, 50, 100, 200, 300, 400, 500, 700, 1000, 1200, 1500,
                   2000, 2500, 3000, 4000, 5000, 6000]
        ex_paths = [os.path.join(TEST_DIR, f"{n:05d}.png") for n in ex_nums
                    if os.path.isfile(os.path.join(TEST_DIR, f"{n:05d}.png"))]

        # 3 satirda 6'sar
        for row_start in range(0, min(len(ex_paths), 18), 6):
            row_paths = ex_paths[row_start:row_start + 6]
            if not row_paths:
                break
            row_cols = st.columns(len(row_paths))
            for col, path in zip(row_cols, row_paths):
                with col:
                    st.image(Image.open(path).resize((64, 64)), use_container_width=True)
                    if st.button("▶", key=f"ex_{path}", use_container_width=True):
                        st.session_state["active_image"] = path
                        st.session_state["active_source"] = os.path.basename(path)

    # Determine active image
    if uploaded is not None:
        image = Image.open(uploaded)
        source = uploaded.name
        st.session_state.pop("active_image", None)
    elif "active_image" in st.session_state:
        image = Image.open(st.session_state["active_image"])
        source = st.session_state.get("active_source", "")
    else:
        image = None
        source = None

    st.markdown("<hr style='border-color:#1a2035;margin:0.8rem 0'>", unsafe_allow_html=True)

    if image is None:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:3.5rem;opacity:0.6">📷</div>
            <p style="color:#3b5278;font-size:1rem;margin-top:0.5rem">
                Fotograf yukle, ornek sec veya sol panelden bir levhaya tikla
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        img_col, res_col = st.columns([1, 1], gap="large")

        with img_col:
            st.image(image, caption=source, use_container_width=True)

        with res_col:
            with st.spinner("Taniniyor..."):
                probs, top5 = run_predict(image)

            best_idx = int(top5[0])
            best_name = CLASS_NAMES.get(best_idx, f"Sinif {best_idx}")
            best_conf = float(probs[best_idx]) * 100
            icon = get_icon(best_name)

            if best_conf >= 90:
                g_cls, g_color, g_text = "g-hi", "#4ade80", "Cok yuksek guven"
            elif best_conf >= 70:
                g_cls, g_color, g_text = "g-mid", "#facc15", "Yuksek guven"
            else:
                g_cls, g_color, g_text = "g-lo", "#f87171", "Dusuk guven"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-icon">{icon}</div>
                <div class="result-name">{best_name}</div>
                <div class="gauge-wrap">
                    <div class="gauge-track">
                        <div class="gauge-fill" style="width:{best_conf:.1f}%;background:{g_color}"></div>
                    </div>
                    <div class="gauge-label">
                        <span class="gauge-pct {g_cls}">%{best_conf:.1f}</span>
                        <span class="gauge-text">{g_text}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**En olasi 5 tahmin:**")
            bars = '<div class="bars">'
            for rank, idx in enumerate(top5):
                name = CLASS_NAMES.get(int(idx), f"Sinif {idx}")
                pct = float(probs[idx]) * 100
                trunc = (name[:22] + "..") if len(name) > 24 else name
                bars += f"""
                <div class="bar-row">
                  <div class="bar-lbl">{get_icon(name)} {trunc}</div>
                  <div class="bar-track">
                    <div class="bar-fill" style="width:{max(pct, 0.5):.1f}%;background:{BAR_COLORS[rank]}"></div>
                  </div>
                  <div class="bar-pct">%{pct:.1f}</div>
                </div>"""
            bars += "</div>"
            st.markdown(bars, unsafe_allow_html=True)

            if best_conf < 70:
                st.warning("Dusuk guven — farkli aci veya daha net fotograf deneyin.")

with tab_about:
    st.markdown("### Model Detaylari")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-panel">
            <div class="info-row"><span class="info-key">Mimari</span><span class="info-val">VGG-Stili CNN v2</span></div>
            <div class="info-row"><span class="info-key">Conv Blok</span><span class="info-val">4 blok (64→128→256→512)</span></div>
            <div class="info-row"><span class="info-key">Parametre</span><span class="info-val">~3.5M (tamami egitilebilir)</span></div>
            <div class="info-row"><span class="info-key">Giris Boyutu</span><span class="info-val">48 x 48 x 3 piksel</span></div>
            <div class="info-row"><span class="info-key">Optimizer</span><span class="info-val">Adam + Cosine Annealing</span></div>
            <div class="info-row"><span class="info-key">Regularizasyon</span><span class="info-val">SpatialDropout + BatchNorm</span></div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-panel">
            <div class="info-row"><span class="info-key">Veri Seti</span><span class="info-val">GTSRB (Alman Trafik Isaretleri)</span></div>
            <div class="info-row"><span class="info-key">Egitim Ornegi</span><span class="info-val">51K+ (oversampled & augmented)</span></div>
            <div class="info-row"><span class="info-key">Test Ornegi</span><span class="info-val">12,630</span></div>
            <div class="info-row"><span class="info-key">Sinif Sayisi</span><span class="info-val">43</span></div>
            <div class="info-row"><span class="info-key">Onisleme</span><span class="info-val">CLAHE + Normalizasyon</span></div>
            <div class="info-row"><span class="info-key">Augmentation</span><span class="info-val">Rotation, Zoom, Brightness, Contrast</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Veri Genisletme Teknikleri")
    st.markdown("""
    - **Oversampling**: Az ornekli siniflar (min 150) hedef sayiya (1200) augmentation ile cogaltildi
    - **CLAHE**: Dusuk kontrastli ve golgeli fotograflarda detay artirildi
    - **Sinif Agirliklari**: Egitimde dengesiz siniflara daha yuksek agirlik verildi
    - **Online Augmentation**: Her epoch'ta farkli donusumler (rotation, zoom, brightness, contrast, translation)
    """)

    st.markdown("### Desteklenen 43 Isaret")
    for cat_name, ids in CATEGORIES:
        names = [f"{get_icon(CLASS_NAMES[i])} {CLASS_NAMES[i]}" for i in ids if i in CLASS_NAMES]
        st.markdown(f"**{cat_name}**: {' · '.join(names)}")

st.markdown('<div class="footer">GTSRB · CNN v2 · TensorFlow · Apple M4 Metal</div>', unsafe_allow_html=True)
