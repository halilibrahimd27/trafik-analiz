"""
Trafik Isareti Tanima - Streamlit Web Arayuzu (v3)

Ozellikler:
  - Gorsel tahmin (Top-5 + confidence gauge)
  - Grad-CAM gorsellestirmesi (modelin nereye baktigi)
  - TTA (Test-Time Augmentation) toggle
  - Egitim gecmisi grafigi + sinif dagilimi
  - Sinif bazli dogruluk tablosu
  - Model mimarisi ve yontem detaylari

Kullanim: python3 -m streamlit run src/app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import pandas as pd

from src.config import (
    MODEL_PATH, IMG_SIZE, CLASS_NAMES, NUM_CLASSES,
    HISTORY_JSON_PATH, RESULT_DIR,
)
from src.gradcam import make_gradcam_heatmap, overlay_heatmap, find_last_conv_layer
from src.dataset import get_tta_augmentations, get_class_counts

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

/* Metric box */
.metric-box {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155; border-radius: 14px;
    padding: 1rem; text-align: center;
}
.metric-val {
    font-size: 1.6rem; font-weight: 800; color: #f1f5f9;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-label { color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; }

/* Architecture block */
.arch-block {
    background: #0c1525; border: 1px solid #1e293b; border-radius: 10px;
    padding: 0.6rem 1rem; margin: 4px 0;
    display: flex; justify-content: space-between; align-items: center;
}
.arch-name  { color: #e2e8f0; font-weight: 600; font-size: 0.88rem; }
.arch-shape { color: #60a5fa; font-family: monospace; font-size: 0.82rem; }
.arch-params{ color: #64748b; font-size: 0.75rem; }

/* Tabs */
button[data-baseweb="tab"] { color: #94a3b8 !important; font-weight: 600 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #60a5fa !important; }

.empty-state { text-align: center; padding: 3rem; }
.footer { text-align: center; color: #1e293b; font-size: 0.75rem; padding: 1rem 0 0.3rem; }

/* Method card (ders sunumu icin) */
.method-card {
    background: #0c1525; border: 1px solid #1e2d48; border-radius: 12px;
    padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}
.method-title { color: #38bdf8; font-weight: 700; font-size: 0.95rem; margin-bottom: 0.3rem; }
.method-desc { color: #94a3b8; font-size: 0.82rem; line-height: 1.5; }

/* OOD / Unknown card */
.ood-card {
    background: linear-gradient(135deg, #3f1d1d 0%, #1a0a0a 100%);
    border: 1px solid #7f1d1d; border-radius: 20px;
    padding: 1.8rem; text-align: center; position: relative; overflow: hidden;
}
.ood-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #ef4444, #f97316, #eab308);
}
.ood-icon { font-size: 3rem; margin-bottom: 0.3rem; }
.ood-title { font-size: 1.4rem; font-weight: 800; color: #fca5a5; margin: 0.3rem 0; }
.ood-desc  { color: #fecaca; font-size: 0.85rem; line-height: 1.5; margin-top: 0.4rem; }
.ood-tag   { display: inline-block; background: #7f1d1d; color: #fee2e2;
             padding: 3px 10px; border-radius: 8px; font-size: 0.72rem; font-weight: 700;
             margin-top: 0.6rem; letter-spacing: 0.5px; }

/* Ambiguous card */
.amb-card {
    background: linear-gradient(135deg, #3f2e0f 0%, #1a1305 100%);
    border: 1px solid #a16207; border-radius: 20px;
    padding: 1.6rem; text-align: center;
}
.amb-title { font-size: 1.2rem; font-weight: 700; color: #fde047; }

/* Disclaimer */
.disclaimer {
    background: #0c1525; border-left: 3px solid #38bdf8;
    border-radius: 6px; padding: 0.5rem 0.8rem; margin-top: 0.8rem;
    color: #94a3b8; font-size: 0.75rem; line-height: 1.45;
}
</style>
""", unsafe_allow_html=True)


# -- Model / helpers -------------------------------------------------------

@st.cache_resource(show_spinner="Model yukleniyor...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def get_last_conv_name():
    try:
        return find_last_conv_layer(load_model())
    except Exception:
        return None


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


@st.cache_data
def load_history_json():
    if os.path.isfile(HISTORY_JSON_PATH):
        with open(HISTORY_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_classification_report_summary():
    report_path = os.path.join(RESULT_DIR, "classification_report.txt")
    if not os.path.isfile(report_path):
        return None
    summary = {}
    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()
    for key in ["Doğruluk", "Kesinlik", "Duyarlılık", "F1 Skoru"]:
        for line in text.splitlines():
            if line.startswith(key):
                # "Doğruluk  : 97.55%"
                val = line.split(":")[-1].strip().rstrip("%")
                try:
                    summary[key] = float(val)
                except ValueError:
                    pass
                break
    return summary


@st.cache_data
def load_class_counts():
    return get_class_counts(DATA_DIR, NUM_CLASSES)


def apply_clahe_single(img_np):
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


def preprocess_image(image: Image.Image, apply_clahe: bool = True):
    img = image.convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]))
    x = np.array(img, dtype=np.float32) / 255.0
    if apply_clahe:
        x = apply_clahe_single(x)
    return x


def run_predict(image, use_tta: bool = False):
    model = load_model()
    x = preprocess_image(image, apply_clahe=True)

    if use_tta:
        tfs = get_tta_augmentations()
        variants = [fn(tf.convert_to_tensor(x, dtype=tf.float32)).numpy() for fn in tfs]
        batch = np.stack(variants, axis=0)
        probs_all = model.predict(batch, verbose=0)
        probs = probs_all.mean(axis=0)
    else:
        probs = model.predict(x[np.newaxis], verbose=0)[0]

    top5 = np.argsort(probs)[::-1][:5]
    return probs, top5, x


class_image_map = get_class_image_map()

# -- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🗂️ Desteklenen Levhalar")
    st.markdown("<small style='color:#3b5278'>Tikla -> ornek yukle</small>", unsafe_allow_html=True)

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
_summary = load_classification_report_summary()
_acc_display = f"%{_summary['Doğruluk']:.1f}" if _summary and "Doğruluk" in _summary else "%98+"

st.markdown(f"""
<div class="hero">
  <h1>🚦 Trafik Isareti Tanima</h1>
  <p>GTSRB veri seti uzerinde egitilmis 4 bloklu CNN v3 ile 43 sinif tanima</p>
</div>
<div class="stats-row">
  <div class="stat-item"><div class="stat-val">43</div><div class="stat-label">Sinif</div></div>
  <div class="stat-item"><div class="stat-val">{_acc_display}</div><div class="stat-label">Test Dogrulugu</div></div>
  <div class="stat-item"><div class="stat-val">CNN v3</div><div class="stat-label">Mimari</div></div>
  <div class="stat-item"><div class="stat-val">GTSRB</div><div class="stat-label">Veri Seti</div></div>
</div>
""", unsafe_allow_html=True)


# -- Tabs ------------------------------------------------------------------
tab_predict, tab_gradcam, tab_stats, tab_arch, tab_about = st.tabs([
    "📸 Tahmin", "🔍 Grad-CAM Analizi", "📊 Egitim & Istatistik",
    "🧠 Model Mimarisi", "🎓 Yontem & Sunum"
])


# ===== Tab 1: Predict =====================================================
with tab_predict:
    upload_col, ctrl_col = st.columns([3, 2], gap="large")

    with upload_col:
        uploaded = st.file_uploader(
            "Fotograf yukle",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )
        st.markdown('<p class="upload-hint">JPG · PNG · BMP · WebP</p>', unsafe_allow_html=True)

    with ctrl_col:
        use_tta = st.toggle(
            "🔀 TTA (Test-Time Augmentation)",
            value=False,
            help="Ayni goruntunun 5 varyantini tahmin edip olasiliklari ortalar. "
                 "Daha kararli tahminler, ancak ~5x daha yavas."
        )
        ood_threshold = st.slider(
            "🚨 Bilinmeyen esigi (guven %)",
            min_value=50, max_value=95, value=80, step=5,
            help="Top-1 guven bu esigin altindaysa 'Bu levha GTSRB'nin 43 sinifinda yok' "
                 "uyarisi gosterilir. Model eminken yanlis olabilir — Grad-CAM ile dogrula."
        )
        st.caption("CLAHE kontrast iyilestirme her zaman aktiftir.")

    st.markdown("**Hazir ornekler:**")
    ex_nums = [0, 10, 50, 100, 200, 300, 400, 500, 700, 1000, 1200, 1500,
               2000, 2500, 3000, 4000, 5000, 6000]
    ex_paths = [os.path.join(TEST_DIR, f"{n:05d}.png") for n in ex_nums
                if os.path.isfile(os.path.join(TEST_DIR, f"{n:05d}.png"))]

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
            with st.spinner("TTA ile taniniyor..." if use_tta else "Taniniyor..."):
                probs, top5, processed_x = run_predict(image, use_tta=use_tta)

            # Aktif goruntuyu session'a kaydet (Grad-CAM sekmesi kullansin)
            st.session_state["last_processed"] = processed_x
            st.session_state["last_probs"] = probs

            best_idx = int(top5[0])
            second_idx = int(top5[1])
            best_name = CLASS_NAMES.get(best_idx, f"Sinif {best_idx}")
            best_conf  = float(probs[best_idx]) * 100
            second_conf = float(probs[second_idx]) * 100
            margin = best_conf - second_conf

            # Dagilim entropisi (0 = cok emin, log(43)≈3.76 = hic emin degil)
            eps = 1e-10
            entropy = float(-(probs * np.log(probs + eps)).sum())
            entropy_norm = entropy / np.log(NUM_CLASSES)  # 0..1 araligi

            # Siniflandirma: OOD / Belirsiz / Normal
            is_ood       = best_conf < ood_threshold
            is_ambiguous = (not is_ood) and (margin < 15.0)

            icon = get_icon(best_name)
            tta_badge = " · TTA 🔀" if use_tta else ""

            if is_ood:
                # Bilinmeyen / GTSRB disi levha uyarisi
                st.markdown(f"""
                <div class="ood-card">
                    <div class="ood-icon">❌</div>
                    <div class="ood-title">Bu levha tanınamadı</div>
                    <div class="ood-desc">
                        Model en iyi tahmini <b>{best_name}</b> ile yalnızca
                        <b>%{best_conf:.1f}</b> güvenle yaptı — belirlenen eşik
                        <b>%{ood_threshold}</b>'in altında.<br>
                        Büyük olasılıkla bu levha <b>GTSRB'nin 43 sınıfında yok</b>
                        veya görüntü kalitesi düşük.
                    </div>
                    <div class="ood-tag">⚠️ OUT-OF-DISTRIBUTION</div>
                </div>
                """, unsafe_allow_html=True)
            elif is_ambiguous:
                st.markdown(f"""
                <div class="amb-card">
                    <div class="result-icon">🤔</div>
                    <div class="amb-title">Belirsiz tahmin</div>
                    <div style="color:#fde68a;font-size:0.85rem;margin-top:0.4rem">
                        Top-1 ({best_name}) ile Top-2 arasındaki fark sadece
                        <b>%{margin:.1f}</b>. Görüntü birden fazla sınıfa benziyor olabilir.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Normal sonuc karti
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
                    <div style="color:#64748b;font-size:0.75rem;margin-top:0.2rem">Sinif #{best_idx}{tta_badge}</div>
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

            # Her durumda top-5 ve metrikleri goster
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

            # Detayli metrikler
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric("Top-1", f"%{best_conf:.1f}")
            with mcol2:
                st.metric("Top-1 → Top-2 fark", f"%{margin:.1f}",
                          help="Buyuk fark = model emin. Kucuk fark = belirsiz.")
            with mcol3:
                st.metric("Entropi", f"{entropy_norm:.3f}",
                          help="0=cok emin, 1=hic emin degil. 0.3 ustu OOD sinyali olabilir.")

            # Her zaman disclaimer
            st.markdown(
                '<div class="disclaimer">'
                '<b>ℹ️ Not:</b> Bu model yalnizca <b>GTSRB veri setinin 43 sinifiyla</b> '
                'egitildi. Bu sinif listesinde olmayan bir levha goruntuledigini '
                'dusunuyorsan, model onu <b>tanidigi en benzer sinifa atar</b> — '
                'bu durumda yuksek guven bile yaniltici olabilir. Emin degilsen '
                '<b>🔍 Grad-CAM Analizi</b> sekmesinde modelin nereye baktigini incele.'
                '</div>',
                unsafe_allow_html=True,
            )


# ===== Tab 2: Grad-CAM =====================================================
with tab_gradcam:
    st.markdown("### 🔍 Grad-CAM — Modelin Dikkat Haritasi")
    st.markdown(
        "<p style='color:#94a3b8;font-size:0.88rem'>"
        "Grad-CAM (<i>Gradient-weighted Class Activation Mapping</i>), modelin "
        "tahmin yaparken <b>goruntunun hangi bolgelerine dikkat ettigini</b> "
        "gosteren bir yorumlanabilirlik teknigidir. Sicak renkler (kirmizi/sari) "
        "yuksek dikkat, soguk renkler (mavi) dusuk dikkat anlamina gelir."
        "</p>",
        unsafe_allow_html=True,
    )

    if "last_processed" not in st.session_state:
        st.info("Once **📸 Tahmin** sekmesinden bir goruntu secin, ardindan buraya donun.")
    else:
        target_col, _ = st.columns([2, 1])
        with target_col:
            target_options = ["Model tahmininin sinifi"] + [
                f"{i} — {CLASS_NAMES[i]}" for i in range(NUM_CLASSES)
            ]
            target_choice = st.selectbox(
                "Hangi sinif icin dikkat haritasi?",
                target_options,
                index=0,
                help="Modelin en guclu tahminine gore veya secilen sinifa gore heatmap uretilir."
            )

        processed = st.session_state["last_processed"]
        probs = st.session_state["last_probs"]

        if target_choice == "Model tahmininin sinifi":
            target_idx = int(np.argmax(probs))
        else:
            target_idx = int(target_choice.split(" — ")[0])

        try:
            with st.spinner("Grad-CAM hesaplaniyor..."):
                model = load_model()
                last_conv = get_last_conv_name()
                heatmap = make_gradcam_heatmap(
                    model,
                    processed[np.newaxis],
                    last_conv_name=last_conv,
                    pred_index=target_idx,
                )
                overlay = overlay_heatmap(processed, heatmap, alpha=0.5)
                raw_cmap_img = (heatmap * 255).astype(np.uint8)

            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                st.image((processed * 255).astype(np.uint8), caption="Orijinal (on islenmis)", use_container_width=True)
            with c2:
                st.image(raw_cmap_img, caption="Isi haritasi (ham)", use_container_width=True, clamp=True)
            with c3:
                st.image(overlay, caption="Overlay", use_container_width=True)

            target_name = CLASS_NAMES.get(target_idx, f"Sinif {target_idx}")
            conf = float(probs[target_idx]) * 100
            st.success(
                f"**Hedef sinif:** {get_icon(target_name)} {target_name} "
                f"(#{target_idx}) — Modelin bu sinifa guveni: **%{conf:.1f}** · "
                f"Son konvolusyon katmani: `{last_conv}`"
            )

            st.markdown(
                "<p style='color:#64748b;font-size:0.82rem;margin-top:1rem'>"
                "<b>Yorum:</b> Heatmap, modelin kararini verirken goruntudeki "
                "hangi bolgeleri 'dikkate' aldigini gosterir. Iyi egitilmis bir "
                "model, levhadaki sembol / sekil bolgesine yogunlasmalidir."
                "</p>", unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Grad-CAM hesaplanamadi: {e}")


# ===== Tab 3: Stats / Training ============================================
with tab_stats:
    st.markdown("### 📊 Egitim Gecmisi ve Sinif Istatistikleri")

    hist = load_history_json()
    if hist is None:
        st.info(
            "Egitim gecmisi bulunamadi. `python src/train.py` ile egitim "
            "baslatildiktan sonra bu sekmede grafikler gorunecek."
        )
    else:
        meta = hist.get("_meta", {})
        # Ust metrikler
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            val_best = meta.get("best_val_accuracy", 0) * 100
            st.markdown(
                f'<div class="metric-box"><div class="metric-val">%{val_best:.2f}</div>'
                f'<div class="metric-label">En Iyi Val Acc</div></div>',
                unsafe_allow_html=True,
            )
        with mc2:
            n_epochs = len(hist.get("accuracy", []))
            st.markdown(
                f'<div class="metric-box"><div class="metric-val">{n_epochs}</div>'
                f'<div class="metric-label">Epoch</div></div>',
                unsafe_allow_html=True,
            )
        with mc3:
            elapsed_min = meta.get("elapsed_sec", 0) / 60
            st.markdown(
                f'<div class="metric-box"><div class="metric-val">{elapsed_min:.1f}\'</div>'
                f'<div class="metric-label">Egitim Suresi</div></div>',
                unsafe_allow_html=True,
            )
        with mc4:
            bs = meta.get("batch_size", 64)
            st.markdown(
                f'<div class="metric-box"><div class="metric-val">{bs}</div>'
                f'<div class="metric-label">Batch Size</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("&nbsp;")

        # Egitim grafigi
        g1, g2 = st.columns(2, gap="medium")
        epochs_list = list(range(1, len(hist.get("accuracy", [])) + 1))
        if epochs_list:
            df_acc = pd.DataFrame({
                "epoch": epochs_list,
                "Egitim": hist.get("accuracy", []),
                "Dogrulama": hist.get("val_accuracy", []),
            })
            df_loss = pd.DataFrame({
                "epoch": epochs_list,
                "Egitim": hist.get("loss", []),
                "Dogrulama": hist.get("val_loss", []),
            })

            with g1:
                st.markdown("**📈 Dogruluk Egrisi**")
                st.line_chart(df_acc.set_index("epoch"), height=280)
            with g2:
                st.markdown("**📉 Kayip Egrisi**")
                st.line_chart(df_loss.set_index("epoch"), height=280)

    # Sinif dagilimi
    st.markdown("---")
    st.markdown("### 📦 Egitim Seti Sinif Dagilimi")
    try:
        counts = load_class_counts()
        df_dist = pd.DataFrame({
            "Sinif": [CLASS_NAMES.get(i, str(i)) for i in range(NUM_CLASSES)],
            "Ornek Sayisi": counts.tolist(),
        }).sort_values("Ornek Sayisi", ascending=False).reset_index(drop=True)

        dc1, dc2 = st.columns([3, 2], gap="medium")
        with dc1:
            st.bar_chart(df_dist.set_index("Sinif"), height=420)
        with dc2:
            st.markdown("**En zengin 5 sinif:**")
            st.dataframe(df_dist.head(5), hide_index=True, use_container_width=True)
            st.markdown("**En az 5 sinif:**")
            st.dataframe(df_dist.tail(5), hide_index=True, use_container_width=True)

        total = int(df_dist["Ornek Sayisi"].sum())
        imbalance = int(df_dist["Ornek Sayisi"].max()) / max(1, int(df_dist["Ornek Sayisi"].min()))
        st.caption(
            f"Toplam {total:,} egitim ornegi · 43 sinif · Dengesizlik orani: "
            f"{imbalance:.1f}× (en bol / en az). Bu dengesizlik **oversampling** "
            f"ve opsiyonel **class weights** ile giderilmektedir."
        )
    except Exception as e:
        st.info(f"Sinif dagilimi hesaplanamadi: {e}")


# ===== Tab 4: Architecture ================================================
with tab_arch:
    st.markdown("### 🧠 CNN v3 Model Mimarisi")

    try:
        model = load_model()
        total_params = model.count_params()
        n_layers = len(model.layers)
        input_shape = model.input_shape

        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-val">{total_params/1e6:.2f}M</div>'
                f'<div class="metric-label">Toplam Parametre</div></div>',
                unsafe_allow_html=True,
            )
        with a2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-val">{n_layers}</div>'
                f'<div class="metric-label">Katman Sayisi</div></div>',
                unsafe_allow_html=True,
            )
        with a3:
            shape_str = f"{input_shape[1]}x{input_shape[2]}x{input_shape[3]}"
            st.markdown(
                f'<div class="metric-box"><div class="metric-val">{shape_str}</div>'
                f'<div class="metric-label">Giris Boyutu</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("&nbsp;")
        st.markdown("**Katman Yapisi:**")

        for layer in model.layers:
            try:
                out_shape = layer.output.shape
                shape_str = "x".join(str(s) for s in out_shape[1:])
            except Exception:
                shape_str = "?"
            params = layer.count_params()
            if params > 0:
                params_str = f"{params:,} param"
            else:
                params_str = "—"
            st.markdown(
                f'<div class="arch-block">'
                f'<span class="arch-name">{layer.name} '
                f'<small style="color:#64748b">({layer.__class__.__name__})</small></span>'
                f'<span class="arch-shape">→ {shape_str}</span>'
                f'<span class="arch-params">{params_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Model yuklenemedi: {e}")


# ===== Tab 5: About / Methodology =========================================
with tab_about:
    st.markdown("### 🎓 Yontem ve Sunum Notlari")
    st.markdown(
        "Bu bolum, derin ogrenme dersi sunumunda kullanmak icin proje boyunca "
        "uygulanan yontemleri ozetler."
    )

    method_list = [
        ("1. Veri Seti: GTSRB",
         "German Traffic Sign Recognition Benchmark — 43 sinif, 26,640 egitim + "
         "12,630 test gercek dunya fotografi. Sinif bazli dengesizlik mevcut "
         "(bazi siniflar 150, digerleri 2000 ornek)."),

        ("2. On Isleme: CLAHE",
         "Contrast Limited Adaptive Histogram Equalization — LAB renk "
         "uzayinda L (parlaklik) kanalina uygulanir. Dusuk kontrastli, golgeli "
         "fotograflarda sembol detaylarini belirginlestirir."),

        ("3. Sinif Dengeleme: Oversampling + Class Weights",
         "Az ornekli siniflar, augmentation ile hedef sayiya (1200) "
         "cogaltilir. Opsiyonel olarak class weights ile kayip fonksiyonu "
         "dengelenir. Boylece model azinlik siniflara da duyarli hale gelir."),

        ("4. Mimari: VGG-Stili 4 Bloklu CNN v3",
         "64→128→256→512 filtreli 4 konvolusyon blogu, BatchNormalization, "
         "SpatialDropout2D ve GlobalAveragePooling ile. Toplam ~4.1M "
         "egitilebilir parametre. He_normal agirlik baslatma."),

        ("5. Regularizasyon: SpatialDropout + Label Smoothing + L2",
         "SpatialDropout2D feature map bazli duzenlileme saglar (klasik "
         "dropout'tan conv katmanlar icin daha uygun). Label smoothing (0.1) "
         "modelin asiri guvenden kacinmasina yardim eder. AdamW optimizer "
         "L2 weight decay ile sinir agirliklarini kuculur tutar."),

        ("6. Ogrenme Orani: Cosine Annealing + Warmup",
         "Ilk 3 epoch linear warmup (0 -> 1e-3), ardindan cosine egrisi ile "
         "1e-3 -> 1e-6'ya yumusak azalma. Plateau'lardan kacinmak icin "
         "adaptif olmayan, gorunen onemli bir teknik."),

        ("7. Online Augmentation",
         "Egitim sirasinda her epoch icin farkli donusumler: rastgele dondurme "
         "(±8%), zoom (±10%), parlaklik/kontrast (±18%), oteleme. "
         "Horizontal flip KULLANILMAZ (yon isaretleri bozulur)."),

        ("8. Degerlendirme: Test Set + Confusion Matrix + Per-Class Acc",
         "12,630 hic gorulmemis test ornegi uzerinde accuracy, weighted "
         "precision / recall / F1. Karisim matrisi ve sinif bazli dogruluk "
         "tablosu ile zayif siniflar tespit edilir."),

        ("9. Test-Time Augmentation (TTA)",
         "Inference sirasinda ayni goruntunun birkac varyanti (zoom in/out, "
         "parlaklik degisimleri) tahmin edilir ve olasiliklar ortalanir. "
         "Daha kararli ve biraz daha dogru sonuclar uretir."),

        ("10. Yorumlanabilirlik: Grad-CAM",
         "Gradient-weighted Class Activation Mapping — son konvolusyon "
         "katmaninin aktivasyonlarini siniflandirma skorunun gradyanlari ile "
         "agirliklandirir. Sonuc: modelin 'nereye baktigini' gosteren sicaklik "
         "haritasi. Sinif kararinin gorsel olarak dogrulanmasi saglanir."),
    ]

    for title, desc in method_list:
        st.markdown(
            f'<div class="method-card">'
            f'<div class="method-title">{title}</div>'
            f'<div class="method-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🛠️ Teknoloji Yigini")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("""
        <div class="info-panel">
            <div class="info-row"><span class="info-key">Python</span><span class="info-val">3.9+</span></div>
            <div class="info-row"><span class="info-key">Framework</span><span class="info-val">TensorFlow / Keras 2.16</span></div>
            <div class="info-row"><span class="info-key">GPU</span><span class="info-val">Apple M4 Metal</span></div>
            <div class="info-row"><span class="info-key">Veri Seti</span><span class="info-val">GTSRB (Kaggle)</span></div>
        </div>
        """, unsafe_allow_html=True)
    with tc2:
        st.markdown("""
        <div class="info-panel">
            <div class="info-row"><span class="info-key">Onisleme</span><span class="info-val">OpenCV (CLAHE)</span></div>
            <div class="info-row"><span class="info-key">Arayuz</span><span class="info-val">Streamlit</span></div>
            <div class="info-row"><span class="info-key">Metrikler</span><span class="info-val">scikit-learn</span></div>
            <div class="info-row"><span class="info-key">Gorsellestirme</span><span class="info-val">Matplotlib / Seaborn</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 Referanslar")
    st.markdown("""
    - **Stallkamp et al. (2012)** — The German Traffic Sign Recognition Benchmark
    - **Simonyan & Zisserman (2014)** — VGG: Very Deep Convolutional Networks
    - **Ioffe & Szegedy (2015)** — Batch Normalization
    - **Tompson et al. (2015)** — Efficient Object Localization (SpatialDropout)
    - **Selvaraju et al. (2017)** — Grad-CAM: Visual Explanations
    - **Szegedy et al. (2016)** — Label Smoothing (Inception-v3)
    - **Loshchilov & Hutter (2017)** — SGDR: Stochastic Gradient Descent with Warm Restarts
    - **Loshchilov & Hutter (2019)** — Decoupled Weight Decay Regularization (AdamW)
    """)

st.markdown('<div class="footer">GTSRB · CNN v3 · TensorFlow · Grad-CAM · Apple M4 Metal</div>', unsafe_allow_html=True)
