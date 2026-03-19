"""
Trafik İşareti Tanıma — Streamlit Web Arayüzü
Kullanım: python3 -m streamlit run src/app.py
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
DATA_DIR  = os.path.join(BASE_DIR, "data")
TEST_DIR  = os.path.join(DATA_DIR, "Test")

st.set_page_config(page_title="Trafik İşareti Tanıma", page_icon="🚦", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f172a; }
[data-testid="stHeader"]           { background: transparent; }
[data-testid="stSidebar"]          { background: #0a1120; border-right: 1px solid #1e293b; }

.hero { text-align:center; padding:2rem 1rem 0.8rem; }
.hero h1 {
    font-size:2.6rem; font-weight:900;
    background:linear-gradient(90deg,#60a5fa,#a78bfa);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p { color:#64748b; font-size:0.95rem; }
.badges { display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin:0.8rem 0 1.4rem; }
.badge { background:#1e293b; border:1px solid #334155; color:#94a3b8; border-radius:20px; padding:4px 16px; font-size:0.82rem; }

.result-card {
    background:linear-gradient(135deg,#1e293b,#0f172a);
    border:1px solid #334155; border-radius:18px; padding:1.8rem; text-align:center;
}
.result-icon { font-size:3rem; line-height:1.2; }
.result-name { font-size:1.55rem; font-weight:800; color:#f1f5f9; margin:0.3rem 0; }
.conf-hi  { color:#4ade80; font-weight:700; font-size:1.25rem; }
.conf-mid { color:#facc15; font-weight:700; font-size:1.25rem; }
.conf-lo  { color:#f87171; font-weight:700; font-size:1.25rem; }

.bars { background:#0c1525; border-radius:12px; padding:1rem 1.2rem; margin-top:0.8rem; }
.bar-row { display:flex; align-items:center; gap:8px; margin:5px 0; }
.bar-lbl { color:#cbd5e1; font-size:0.81rem; width:185px; flex-shrink:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bar-track { flex:1; background:#1e293b; border-radius:5px; height:12px; overflow:hidden; }
.bar-fill  { height:100%; border-radius:5px; }
.bar-pct   { color:#64748b; font-size:0.78rem; width:36px; text-align:right; flex-shrink:0; }

.cat-header { color:#60a5fa; font-size:0.72rem; font-weight:700; letter-spacing:1px; text-transform:uppercase; border-left:3px solid #3b82f6; padding-left:8px; margin:1rem 0 0.3rem; }

[data-testid="stSidebar"] button {
    background:#1a2540 !important; border:1px solid #2d3f5c !important;
    color:#94a3b8 !important; border-radius:7px !important;
    font-size:0.78rem !important; text-align:left !important; margin:1px 0 !important;
}
[data-testid="stSidebar"] button:hover { background:#243355 !important; color:#e2e8f0 !important; border-color:#3b82f6 !important; }

.upload-hint { text-align:center; color:#475569; font-size:0.8rem; margin-top:4px; }
.empty-state { text-align:center; padding:3rem; }
.footer { text-align:center; color:#334155; font-size:0.78rem; padding:1.5rem 0 0.3rem; }
hr.div { border:none; border-top:1px solid #1e293b; margin:1.2rem 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Model yükleniyor...")
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


def get_icon(name):
    if "Hız Limiti" in name:    return "🚗"
    if "Dur" in name:            return "🛑"
    if "Giremez" in name:        return "⛔"
    if "Geçiş Yasağı" in name or ("Yasak" in name and "Ton" not in name and "Sonu" not in name): return "🚫"
    if "Ton" in name:            return "🚛"
    if "Tehlike" in name or "Viraj" in name or "Kaygan" in name or "Buzlanma" in name or "Engebeli" in name: return "⚠️"
    if "Dönünüz" in name or "Gidiniz" in name or "Geçiniz" in name or "İleri" in name: return "↗️"
    if "Yaya" in name:           return "🚶"
    if "Çocuk" in name:          return "👶"
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
    ("🚗 Hız Limitleri",     [0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ("🚫 Yasaklayıcı",       [9, 10, 15, 16, 17]),
    ("🛑 Dur / Yol Ver",     [13, 14]),
    ("⚠️ Tehlike Uyarıları", [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]),
    ("↗️ Yön & Zorunlu",     [33, 34, 35, 36, 37, 38, 39, 40]),
    ("✅ Yasakların Sonu",    [32, 41, 42]),
    ("💛 Öncelik",           [12]),
]


def run_predict(image):
    model = load_model()
    img   = image.convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]))
    x     = np.array(img, dtype=np.float32) / 255.0
    probs = model.predict(x[np.newaxis], verbose=0)[0]
    top5  = np.argsort(probs)[::-1][:5]
    return probs, top5


class_image_map = get_class_image_map()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗂️ Desteklenen Levhalar")
    st.markdown("<small style='color:#475569'>Tıklayarak örnek görüntü yükle</small>", unsafe_allow_html=True)

    for cat_name, ids in CATEGORIES:
        st.markdown(f"<div class='cat-header'>{cat_name}</div>", unsafe_allow_html=True)
        for cid in ids:
            if cid not in CLASS_NAMES:
                continue
            name = CLASS_NAMES[cid]
            if st.button(f"{get_icon(name)} {name}", key=f"sb_{cid}", use_container_width=True):
                if cid in class_image_map and os.path.isfile(class_image_map[cid]):
                    st.session_state["active_image"]  = class_image_map[cid]
                    st.session_state["active_source"] = name

# ── Ana Sayfa ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚦 Trafik İşareti Tanıma</h1>
  <p>Fotoğraf yükle veya sol panelden bir levhaya tıkla — yapay zeka anında tanısın</p>
</div>
<div class="badges">
  <span class="badge">🎯 43 Sınıf</span>
  <span class="badge">📊 %97.55 Doğruluk</span>
  <span class="badge">⚡ Apple M4 GPU</span>
  <span class="badge">🧠 VGG-Stili CNN</span>
</div>
""", unsafe_allow_html=True)

# ── Yükleme + Örnekler ────────────────────────────────────────────────────────
upload_col, example_col = st.columns([2, 3], gap="large")

with upload_col:
    uploaded = st.file_uploader(
        "Fotoğraf yükle",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
    st.markdown('<p class="upload-hint">JPG · PNG · BMP · WebP — tek levha içeren net fotoğraf en iyi sonucu verir</p>', unsafe_allow_html=True)

with example_col:
    st.markdown("**Hazır örnekler — tıkla:**")
    # 12 farklı örnek: test setinden çeşitli dosya numaraları
    ex_nums = [0, 10, 50, 100, 200, 300, 400, 500, 700, 1000, 1200, 1500]
    ex_paths = [os.path.join(TEST_DIR, f"{n:05d}.png") for n in ex_nums
                if os.path.isfile(os.path.join(TEST_DIR, f"{n:05d}.png"))]

    rows = [ex_paths[:6], ex_paths[6:]]
    for row_paths in rows:
        if not row_paths:
            continue
        row_cols = st.columns(len(row_paths))
        for i, (col, path) in enumerate(zip(row_cols, row_paths)):
            with col:
                st.image(Image.open(path).resize((68, 68)), use_container_width=True)
                if st.button("▶", key=f"ex_{path}", use_container_width=True):
                    st.session_state["active_image"]  = path
                    st.session_state["active_source"] = f"Örnek — {os.path.basename(path)}"

# ── Aktif görüntü ─────────────────────────────────────────────────────────────
if uploaded is not None:
    image  = Image.open(uploaded)
    source = uploaded.name
    st.session_state.pop("active_image", None)
elif "active_image" in st.session_state:
    image  = Image.open(st.session_state["active_image"])
    source = st.session_state.get("active_source", "")
else:
    image  = None
    source = None

# ── Tahmin ────────────────────────────────────────────────────────────────────
st.markdown("<hr class='div'>", unsafe_allow_html=True)

if image is None:
    st.markdown("""
    <div class="empty-state">
        <div style="font-size:3.5rem">📷</div>
        <p style="color:#475569;font-size:1.05rem;margin-top:0.5rem">Fotoğraf yükle, örnek seç veya <b style='color:#60a5fa'>sol panelden</b> bir levhaya tıkla</p>
    </div>
    """, unsafe_allow_html=True)
else:
    img_col, res_col = st.columns([1, 1], gap="large")

    with img_col:
        st.image(image, caption=source, use_container_width=True)

    with res_col:
        with st.spinner("Tanıyor..."):
            probs, top5 = run_predict(image)

        best_idx  = int(top5[0])
        best_name = CLASS_NAMES.get(best_idx, f"Sınıf {best_idx}")
        best_conf = float(probs[best_idx]) * 100
        icon      = get_icon(best_name)

        if best_conf >= 90:
            conf_cls, conf_label = "conf-hi",  "✅ Çok yüksek güven"
        elif best_conf >= 70:
            conf_cls, conf_label = "conf-mid", "🟡 Yüksek güven"
        else:
            conf_cls, conf_label = "conf-lo",  "🔴 Düşük güven"

        st.markdown(f"""
        <div class="result-card">
            <div class="result-icon">{icon}</div>
            <div class="result-name">{best_name}</div>
            <div style="color:#94a3b8;font-size:0.92rem;margin-top:0.3rem">
                <span class="{conf_cls}">%{best_conf:.1f}</span>
                &nbsp;&nbsp;{conf_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**En olası 5 tahmin:**")
        bars = '<div class="bars">'
        for rank, idx in enumerate(top5):
            name  = CLASS_NAMES.get(int(idx), f"Sınıf {idx}")
            pct   = float(probs[idx]) * 100
            trunc = (name[:24] + "..") if len(name) > 26 else name
            bars += f"""
            <div class="bar-row">
              <div class="bar-lbl">{get_icon(name)} {trunc}</div>
              <div class="bar-track">
                <div class="bar-fill" style="width:{pct:.1f}%;background:{BAR_COLORS[rank]}"></div>
              </div>
              <div class="bar-pct">%{pct:.1f}</div>
            </div>"""
        bars += "</div>"
        st.markdown(bars, unsafe_allow_html=True)

        if best_conf < 70:
            st.info("💡 Net, yakın çekim ve sadece levhayı içeren fotoğraflar daha iyi sonuç verir.")

st.markdown('<div class="footer">GTSRB · Özel CNN (VGG-stili) · TensorFlow/Keras · Apple M4 Metal</div>', unsafe_allow_html=True)
