"""
Trafik İşareti Tanıma — Kurumsal Web Arayüzü (CNN v3)

Tasarım: kurumsal tema (açık/koyu geçişli), SVG marka kimliği, düz kartlar.
İşlevler:
  - Görsel tahmin (Top-5, güven göstergesi, CLAHE öncesi/sonrası, çıkarım süresi)
  - Grad-CAM görselleştirmesi (modelin dikkat haritası)
  - TTA (Test-Time Augmentation) seçeneği + OOD / belirsiz uyarıları
  - Eğitim geçmişi, sonuç görselleri (confusion matrix, örnek tahminler),
    sınıf-bazlı performans tablosu, model mimarisi, yöntem notları

Kullanım: python3 -m streamlit run src/app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time

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
# Veri seti olmayan kurulumlar için repoya gömülü küçük örnek seti (<classid>.png)
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

st.set_page_config(
    page_title="Trafik İşareti Tanıma Sistemi",
    page_icon="🚦",
    layout="wide",
)


# ── Tema (açık / koyu) — CSS değişkenleri ─────────────────────────────────
def _theme_vars(dark: bool) -> str:
    if dark:
        v = ("--bg:#0b1220;--surface:#111c30;--surface2:#0c1525;"
             "--border:#1e2d48;--border2:#334155;--text:#e6edf7;"
             "--text2:#9fb0c9;--muted:#64748b;--accent:#3b82f6;"
             "--accent-soft:#16233f;--shadow:rgba(0,0,0,0.35);")
    else:
        v = ("--bg:#f8fafc;--surface:#ffffff;--surface2:#f1f5f9;"
             "--border:#e2e8f0;--border2:#cbd5e1;--text:#0f172a;"
             "--text2:#475569;--muted:#94a3b8;--accent:#1d4ed8;"
             "--accent-soft:#eff6ff;--shadow:rgba(15,23,42,0.04);")
    return "<style>:root{" + v + "}</style>"


_MAIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }

[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

[data-testid="stMetricValue"] { color: var(--text) !important; }
[data-testid="stMetricLabel"] { color: var(--text2) !important; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }

/* Üst başlık çubuğu */
.app-header {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.05rem 1.6rem; margin-bottom: 1.4rem;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 1rem; box-shadow: 0 1px 2px var(--shadow);
}
.brand { display: flex; align-items: center; gap: 0.9rem; }
.brand-logo { width: 44px; height: 44px; flex-shrink: 0; }
.brand-title { font-size: 1.25rem; font-weight: 700; color: var(--text); letter-spacing: -0.01em; line-height: 1.2; }
.brand-sub { color: var(--muted); font-size: 0.78rem; margin-top: 2px; }
.kpi-strip { display: flex; gap: 2.2rem; flex-wrap: wrap; }
.kpi { text-align: center; }
.kpi-val { font-size: 1.2rem; font-weight: 700; color: var(--text); line-height: 1.1; }
.kpi-label { font-size: 0.66rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em; margin-top: 3px; }

/* Bölüm başlıkları */
.section-title { font-size: 1.08rem; font-weight: 700; color: var(--text); margin: 0.2rem 0 0.3rem; }
.section-desc  { color: var(--text2); font-size: 0.86rem; line-height: 1.55; margin-bottom: 0.6rem; }
.eyebrow { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); }

/* Kartlar */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.4rem; box-shadow: 0 1px 2px var(--shadow); }

/* Sonuç kartı */
.result-card {
    background: var(--surface); border: 1px solid var(--border); border-top: 3px solid var(--accent);
    border-radius: 12px; padding: 1.5rem 1.6rem; text-align: center; box-shadow: 0 1px 3px var(--shadow);
}
.result-name { font-size: 1.45rem; font-weight: 700; color: var(--text); margin: 0.45rem 0 0.15rem; line-height: 1.25; }
.result-sub  { color: var(--muted); font-size: 0.76rem; }
.badge-v3 { display:inline-block; background: var(--accent-soft); color: var(--accent); font-size:0.66rem;
            font-weight:700; padding:2px 8px; border-radius:5px; letter-spacing:0.04em; margin-left:6px; }

/* Güven göstergesi */
.gauge-wrap { margin: 1rem auto 0; max-width: 280px; }
.gauge-track { background: var(--surface2); border-radius: 6px; height: 8px; overflow: hidden; }
.gauge-fill { height: 100%; border-radius: 6px; transition: width .5s ease; }
.gauge-label { display: flex; justify-content: space-between; margin-top: 6px; align-items: baseline; }
.gauge-pct { font-weight: 700; font-size: 1.05rem; }
.gauge-text { color: var(--muted); font-size: 0.76rem; }

/* Top-5 çubukları */
.bars { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.1rem 1.3rem; margin-top: 0.7rem; }
.bars-title { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.7rem; }
.bar-row { display: flex; align-items: center; gap: 10px; margin: 7px 0; }
.bar-id { font-family: 'Inter', monospace; font-size: 0.7rem; font-weight: 700; color: var(--accent);
          background: var(--accent-soft); border-radius: 5px; padding: 2px 6px; width: 34px; text-align: center; flex-shrink: 0; }
.bar-lbl { color: var(--text2); font-size: 0.82rem; width: 165px; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-track { flex: 1; background: var(--surface2); border-radius: 5px; height: 10px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 5px; transition: width .4s; }
.bar-pct { color: var(--text2); font-size: 0.78rem; font-weight: 600; width: 42px; text-align: right; flex-shrink: 0; }

/* Metrik kutusu */
.metric-box { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 1.1rem; text-align: center; }
.metric-val { font-size: 1.55rem; font-weight: 700; color: var(--accent); line-height: 1.1; }
.metric-label { color: var(--text2); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }

/* Bilgi paneli */
.info-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 0.6rem 1.2rem; }
.info-row { display: flex; justify-content: space-between; padding: 0.55rem 0; border-bottom: 1px solid var(--border); }
.info-row:last-child { border-bottom: none; }
.info-key { color: var(--text2); font-size: 0.83rem; }
.info-val { color: var(--text); font-size: 0.83rem; font-weight: 600; }

/* Mimari katman satırı */
.arch-block { background: var(--surface); border: 1px solid var(--border); border-radius: 9px;
    padding: 0.6rem 1rem; margin: 5px 0; display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
.arch-name  { color: var(--text); font-weight: 600; font-size: 0.85rem; }
.arch-type  { color: var(--muted); font-weight: 400; font-size: 0.76rem; }
.arch-shape { color: var(--accent); font-family: monospace; font-size: 0.8rem; }
.arch-params{ color: var(--muted); font-size: 0.74rem; min-width: 110px; text-align: right; }

/* Yöntem kartı */
.method-card { background: var(--surface); border: 1px solid var(--border); border-left: 3px solid var(--accent);
    border-radius: 9px; padding: 0.9rem 1.2rem; margin-bottom: 0.7rem; }
.method-title { color: var(--text); font-weight: 700; font-size: 0.92rem; margin-bottom: 0.25rem; }
.method-desc  { color: var(--text2); font-size: 0.83rem; line-height: 1.55; }

/* OOD / belirsiz kartları (her iki temada da çalışır) */
.ood-card { background: rgba(220,38,38,0.07); border: 1px solid rgba(220,38,38,0.30); border-top: 3px solid #dc2626;
    border-radius: 12px; padding: 1.5rem 1.6rem; text-align: center; }
.ood-badge { display: inline-block; background: rgba(220,38,38,0.15); color: #ef4444; font-size: 0.66rem; font-weight: 700;
             letter-spacing: 0.06em; padding: 3px 10px; border-radius: 6px; margin-bottom: 0.6rem; }
.ood-title { font-size: 1.3rem; font-weight: 700; color: #ef4444; margin: 0.2rem 0; }
.ood-desc  { color: var(--text2); font-size: 0.84rem; line-height: 1.55; margin-top: 0.4rem; }

.amb-card { background: rgba(217,119,6,0.08); border: 1px solid rgba(217,119,6,0.30); border-top: 3px solid #d97706;
    border-radius: 12px; padding: 1.4rem 1.6rem; text-align: center; }
.amb-badge { display: inline-block; background: rgba(217,119,6,0.15); color: #d97706; font-size: 0.66rem; font-weight: 700;
             letter-spacing: 0.06em; padding: 3px 10px; border-radius: 6px; margin-bottom: 0.5rem; }
.amb-title { font-size: 1.15rem; font-weight: 700; color: #f59e0b; }
.amb-desc  { color: var(--text2); font-size: 0.83rem; margin-top: 0.35rem; }

/* Uyarı / dipnot */
.disclaimer { background: var(--surface2); border: 1px solid var(--border); border-left: 3px solid var(--accent);
    border-radius: 8px; padding: 0.7rem 1rem; margin-top: 0.9rem; color: var(--text2); font-size: 0.78rem; line-height: 1.55; }

/* Dosya yükleyici */
[data-testid="stFileUploader"] { border: 2px dashed var(--border2) !important; border-radius: 12px !important;
    background: var(--surface) !important; transition: border-color .2s; }
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
.upload-hint { text-align: center; color: var(--muted); font-size: 0.76rem; margin-top: 4px; }

/* Kenar çubuğu */
.sb-title { font-size: 0.95rem; font-weight: 700; color: var(--text); margin-bottom: 0.1rem; }
.sb-sub { color: var(--muted); font-size: 0.74rem; margin-bottom: 0.6rem; }
.cat-header { color: var(--text2); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase; border-left: 3px solid var(--accent); padding-left: 8px; margin: 0.9rem 0 0.35rem; }
[data-testid="stSidebar"] button[kind="secondary"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    color: var(--text2) !important; border-radius: 7px !important;
    font-size: 0.78rem !important; text-align: left !important;
    padding: 5px 11px !important; margin: 2px 0 !important; transition: all .15s; }
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: var(--accent-soft) !important; color: var(--accent) !important; border-color: var(--accent) !important; }

/* Sekmeler */
button[data-baseweb="tab"] { color: var(--muted) !important; font-weight: 600 !important; font-size: 0.9rem !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent) !important; }
[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

.empty-state { text-align: center; padding: 3rem 1rem; border: 1px dashed var(--border2); border-radius: 12px; background: var(--surface); }
.empty-state p { color: var(--muted); font-size: 0.95rem; margin-top: 0.5rem; }
.footer { text-align: center; color: var(--muted); font-size: 0.74rem; padding: 1.4rem 0 0.4rem; border-top: 1px solid var(--border); margin-top: 1.5rem; }
</style>
"""

# Tema durumu (sidebar toggle ile yönetilir, üstte okunur)
_dark = bool(st.session_state.get("dark_theme", False))
st.markdown(_theme_vars(_dark), unsafe_allow_html=True)
st.markdown(_MAIN_CSS, unsafe_allow_html=True)


# Marka SVG logosu — TEK SATIR (çok satır olursa markdown araya boş satır koyup
# sonraki HTML'i kod bloğu sanıyor). Tema rengine uyum sağlar.
BRAND_LOGO = (
    '<svg class="brand-logo" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="44" height="44" rx="11" fill="var(--accent)"/>'
    '<circle cx="22" cy="22" r="12" fill="none" stroke="#ffffff" stroke-width="2.6"/>'
    '<path d="M22 14.5 L22 24" stroke="#ffffff" stroke-width="2.8" stroke-linecap="round"/>'
    '<circle cx="22" cy="29" r="1.7" fill="#ffffff"/>'
    '</svg>'
)


# ── Model / yardımcılar ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Model yükleniyor...")
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
    # Öncelik: tam GTSRB test seti (varsa)
    csv_path = os.path.join(DATA_DIR, "Test.csv")
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        mapping = {}
        for _, row in df.iterrows():
            cid = int(row["ClassId"])
            if cid not in mapping:
                mapping[cid] = os.path.join(DATA_DIR, str(row["Path"]))
        if mapping:
            return mapping
    # Fallback: repoya gömülü samples/<classid>.png (veri seti yoksa)
    mapping = {}
    if os.path.isdir(SAMPLES_DIR):
        for fn in sorted(os.listdir(SAMPLES_DIR)):
            if not fn.lower().endswith(".png"):
                continue
            try:
                cid = int(os.path.splitext(fn)[0])
            except ValueError:
                continue
            mapping[cid] = os.path.join(SAMPLES_DIR, fn)
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
                val = line.split(":")[-1].strip().rstrip("%")
                try:
                    summary[key] = float(val)
                except ValueError:
                    pass
                break
    return summary


@st.cache_data
def load_per_class_report():
    """classification_report.txt'i sınıf-bazlı tabloya ayrıştırır."""
    path = os.path.join(RESULT_DIR, "classification_report.txt")
    if not os.path.isfile(path):
        return None
    rows = []
    skip = {"accuracy", "macro avg", "weighted avg"}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                support = int(parts[-1])
                f1 = float(parts[-2])
                rec = float(parts[-3])
                prec = float(parts[-4])
            except ValueError:
                continue
            name = " ".join(parts[:-4]).strip()
            if not name or name.lower() in skip:
                continue
            rows.append({"Sınıf": name, "Precision": prec,
                         "Recall": rec, "F1": f1, "Destek": support})
    if not rows:
        return None
    return pd.DataFrame(rows)


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


# Lacivert tonları (Top-5 çubuk paleti)
BAR_COLORS = ["#1d4ed8", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]

CATEGORIES = [
    ("Hız Limitleri",      [0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ("Yasaklayıcı",        [9, 10, 15, 16, 17]),
    ("Dur / Yol Ver",      [13, 14]),
    ("Tehlike Uyarıları",  [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]),
    ("Yön & Zorunlu",      [33, 34, 35, 36, 37, 38, 39, 40]),
    ("Yasakların Sonu",    [32, 41, 42]),
    ("Öncelik",            [12]),
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

# ── Kenar çubuğu ──────────────────────────────────────────────────────────
with st.sidebar:
    st.toggle("Koyu tema", key="dark_theme",
              help="Açık kurumsal ↔ koyu kurumsal tema arasında geçiş yapar.")
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    st.markdown('<div class="sb-title">Desteklenen Levhalar</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Bir levhaya tıklayın → örnek görüntü yüklenir</div>',
                unsafe_allow_html=True)

    for cat_name, ids in CATEGORIES:
        st.markdown(f"<div class='cat-header'>{cat_name}</div>", unsafe_allow_html=True)
        for cid in ids:
            if cid not in CLASS_NAMES:
                continue
            name = CLASS_NAMES[cid]
            if st.button(f"{cid:>2} · {name}", key=f"sb_{cid}", width="stretch"):
                if cid in class_image_map and os.path.isfile(class_image_map[cid]):
                    st.session_state["active_image"] = class_image_map[cid]
                    st.session_state["active_source"] = name


# ── Üst başlık çubuğu ─────────────────────────────────────────────────────
_summary = load_classification_report_summary()
_acc_display = f"%{_summary['Doğruluk']:.2f}" if _summary and "Doğruluk" in _summary else "%99+"

st.markdown(f"""
<div class="app-header">
  <div class="brand">
    {BRAND_LOGO}
    <div>
      <div class="brand-title">Trafik İşareti Tanıma Sistemi</div>
      <div class="brand-sub">GTSRB · Derin CNN v3 · 43 sınıf görüntü sınıflandırma</div>
    </div>
  </div>
  <div class="kpi-strip">
    <div class="kpi"><div class="kpi-val">43</div><div class="kpi-label">Sınıf</div></div>
    <div class="kpi"><div class="kpi-val">{_acc_display}</div><div class="kpi-label">Test Doğruluğu</div></div>
    <div class="kpi"><div class="kpi-val">CNN v3</div><div class="kpi-label">Mimari</div></div>
    <div class="kpi"><div class="kpi-val">4.98M</div><div class="kpi-label">Parametre</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Sekmeler ──────────────────────────────────────────────────────────────
tab_predict, tab_gradcam, tab_stats, tab_arch, tab_about = st.tabs([
    "Tahmin", "Grad-CAM Analizi", "Eğitim & İstatistik",
    "Model Mimarisi", "Yöntem & Sunum",
])


# ===== Sekme 1: Tahmin ====================================================
with tab_predict:
    upload_col, ctrl_col = st.columns([3, 2], gap="large")

    with upload_col:
        st.markdown('<div class="eyebrow">Görüntü Girişi</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Fotoğraf yükle",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )
        st.markdown('<p class="upload-hint">JPG · PNG · BMP · WebP formatları desteklenir</p>',
                    unsafe_allow_html=True)

    with ctrl_col:
        st.markdown('<div class="eyebrow">Çıkarım Ayarları</div>', unsafe_allow_html=True)
        use_tta = st.toggle(
            "TTA — Test-Time Augmentation",
            value=False,
            help="Aynı görüntünün 5 varyantını tahmin edip olasılıkları ortalar. "
                 "Daha kararlı tahminler, ancak ~5× daha yavaş.",
        )
        ood_threshold = st.slider(
            "Bilinmeyen levha eşiği (güven %)",
            min_value=50, max_value=95, value=80, step=5,
            help="Top-1 güven bu eşiğin altındaysa 'GTSRB'nin 43 sınıfında yok' "
                 "uyarısı gösterilir. Model eminken yanlış olabilir — Grad-CAM ile doğrulayın.",
        )
        st.caption("CLAHE kontrast iyileştirme her zaman aktiftir.")

    st.markdown('<div class="eyebrow" style="margin-top:0.6rem">Hazır Örnekler</div>',
                unsafe_allow_html=True)
    ex_nums = [0, 10, 50, 100, 200, 300, 400, 500, 700, 1000, 1200, 1500,
               2000, 2500, 3000, 4000, 5000, 6000]
    ex_paths = [os.path.join(TEST_DIR, f"{n:05d}.png") for n in ex_nums
                if os.path.isfile(os.path.join(TEST_DIR, f"{n:05d}.png"))]
    # Veri seti yoksa repoya gömülü örneklere düş
    if not ex_paths and os.path.isdir(SAMPLES_DIR):
        ex_paths = [os.path.join(SAMPLES_DIR, f) for f in sorted(os.listdir(SAMPLES_DIR))
                    if f.lower().endswith(".png")]

    for row_start in range(0, min(len(ex_paths), 18), 6):
        row_paths = ex_paths[row_start:row_start + 6]
        if not row_paths:
            break
        row_cols = st.columns(len(row_paths))
        for col, path in zip(row_cols, row_paths):
            with col:
                st.image(Image.open(path).resize((64, 64)), width="stretch")
                if st.button("Seç", key=f"ex_{path}", width="stretch"):
                    st.session_state["active_image"] = path
                    st.session_state["active_source"] = os.path.basename(path)

    # Aktif görüntüyü belirle
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

    st.markdown("<hr style='border:none;border-top:1px solid var(--border);margin:1rem 0'>",
                unsafe_allow_html=True)

    if image is None:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:2.4rem;color:var(--border2)">◇</div>
            <p>Bir fotoğraf yükleyin, hazır örnek seçin ya da soldan bir levha kategorisine tıklayın.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        img_col, res_col = st.columns([1, 1], gap="large")

        with img_col:
            st.markdown('<div class="eyebrow">Girdi Görüntüsü</div>', unsafe_allow_html=True)
            st.image(image, caption=source, width="stretch")

            # CLAHE öncesi / sonrası karşılaştırma
            with st.expander("CLAHE ön işleme — öncesi / sonrası"):
                raw_x = preprocess_image(image, apply_clahe=False)
                clahe_x = preprocess_image(image, apply_clahe=True)
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.image((np.clip(raw_x, 0, 1) * 255).astype(np.uint8),
                             caption="Ham (yalnızca normalize)", width="stretch")
                with cc2:
                    st.image((np.clip(clahe_x, 0, 1) * 255).astype(np.uint8),
                             caption="CLAHE sonrası (modele giren)", width="stretch")
                st.caption("CLAHE, LAB uzayında parlaklık (L) kanalına uygulanır; "
                           "düşük kontrastlı görüntülerde sembol detaylarını belirginleştirir.")

        with res_col:
            t0 = time.time()
            with st.spinner("TTA ile değerlendiriliyor..." if use_tta else "Değerlendiriliyor..."):
                probs, top5, processed_x = run_predict(image, use_tta=use_tta)
            infer_ms = (time.time() - t0) * 1000.0

            st.session_state["last_processed"] = processed_x
            st.session_state["last_probs"] = probs

            best_idx = int(top5[0])
            second_idx = int(top5[1])
            best_name = CLASS_NAMES.get(best_idx, f"Sınıf {best_idx}")
            best_conf = float(probs[best_idx]) * 100
            second_conf = float(probs[second_idx]) * 100
            margin = best_conf - second_conf

            eps = 1e-10
            entropy = float(-(probs * np.log(probs + eps)).sum())
            entropy_norm = entropy / np.log(NUM_CLASSES)

            is_ood = best_conf < ood_threshold
            is_ambiguous = (not is_ood) and (margin < 15.0)
            tta_badge = " · TTA" if use_tta else ""

            if is_ood:
                st.markdown(f"""
                <div class="ood-card">
                    <div class="ood-badge">OUT-OF-DISTRIBUTION</div>
                    <div class="ood-title">Levha tanınamadı</div>
                    <div class="ood-desc">
                        Model en iyi tahmini <b>{best_name}</b> ile yalnızca
                        <b>%{best_conf:.1f}</b> güvenle yaptı — belirlenen eşik
                        <b>%{ood_threshold}</b>'in altında. Büyük olasılıkla bu levha
                        <b>GTSRB'nin 43 sınıfında yok</b> veya görüntü kalitesi düşük.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif is_ambiguous:
                st.markdown(f"""
                <div class="amb-card">
                    <div class="amb-badge">BELİRSİZ</div>
                    <div class="amb-title">Belirsiz tahmin</div>
                    <div class="amb-desc">
                        Top-1 (<b>{best_name}</b>) ile Top-2 arasındaki fark yalnızca
                        <b>%{margin:.1f}</b>. Görüntü birden fazla sınıfa benziyor olabilir.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                if best_conf >= 90:
                    g_color, g_text = "#16a34a", "Çok yüksek güven"
                elif best_conf >= 70:
                    g_color, g_text = "var(--accent)", "Yüksek güven"
                else:
                    g_color, g_text = "#d97706", "Düşük güven"

                st.markdown(f"""
                <div class="result-card">
                    <div class="eyebrow">Tahmin Sonucu<span class="badge-v3">CNN v3</span></div>
                    <div class="result-name">{best_name}</div>
                    <div class="result-sub">Sınıf #{best_idx} · {infer_ms:.0f} ms{tta_badge}</div>
                    <div class="gauge-wrap">
                        <div class="gauge-track">
                            <div class="gauge-fill" style="width:{best_conf:.1f}%;background:{g_color}"></div>
                        </div>
                        <div class="gauge-label">
                            <span class="gauge-pct" style="color:{g_color}">%{best_conf:.1f}</span>
                            <span class="gauge-text">{g_text}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Top-5
            bars = '<div class="bars"><div class="bars-title">En Olası 5 Tahmin</div>'
            for rank, idx in enumerate(top5):
                name = CLASS_NAMES.get(int(idx), f"Sınıf {idx}")
                pct = float(probs[idx]) * 100
                trunc = (name[:24] + "…") if len(name) > 25 else name
                bars += f"""
                <div class="bar-row">
                  <div class="bar-id">{int(idx)}</div>
                  <div class="bar-lbl">{trunc}</div>
                  <div class="bar-track">
                    <div class="bar-fill" style="width:{max(pct, 0.5):.1f}%;background:{BAR_COLORS[rank]}"></div>
                  </div>
                  <div class="bar-pct">%{pct:.1f}</div>
                </div>"""
            bars += "</div>"
            st.markdown(bars, unsafe_allow_html=True)

            # Metrikler
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric("Top-1 güven", f"%{best_conf:.1f}")
            with mcol2:
                st.metric("Top-1 → Top-2 fark", f"%{margin:.1f}",
                          help="Büyük fark = model emin. Küçük fark = belirsiz.")
            with mcol3:
                st.metric("Entropi", f"{entropy_norm:.3f}",
                          help="0 = çok emin, 1 = hiç emin değil. 0.3 üstü OOD sinyali olabilir.")

            st.markdown(
                '<div class="disclaimer">'
                '<b>Not:</b> Bu model yalnızca <b>GTSRB veri setinin 43 sınıfıyla</b> '
                'eğitildi. Bu listede olmayan bir levha gördüğünde, model onu '
                '<b>tanıdığı en benzer sınıfa atar</b> — bu durumda yüksek güven bile '
                'yanıltıcı olabilir. Emin değilseniz <b>Grad-CAM Analizi</b> sekmesinde '
                'modelin nereye baktığını inceleyin.'
                '</div>',
                unsafe_allow_html=True,
            )


# ===== Sekme 2: Grad-CAM ==================================================
with tab_gradcam:
    st.markdown('<div class="section-title">Grad-CAM — Modelin Dikkat Haritası</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">'
        'Grad-CAM (<i>Gradient-weighted Class Activation Mapping</i>), modelin tahmin '
        'yaparken görüntünün hangi bölgelerine dikkat ettiğini gösterir. Sıcak renkler '
        '(kırmızı/sarı) yüksek dikkat, soğuk renkler (mavi) düşük dikkat anlamına gelir.'
        '</div>',
        unsafe_allow_html=True,
    )

    if "last_processed" not in st.session_state:
        st.info("Önce **Tahmin** sekmesinden bir görüntü seçin, ardından bu sekmeye dönün.")
    else:
        target_col, _ = st.columns([2, 1])
        with target_col:
            target_options = ["Model tahmininin sınıfı"] + [
                f"{i} — {CLASS_NAMES[i]}" for i in range(NUM_CLASSES)
            ]
            target_choice = st.selectbox(
                "Hangi sınıf için dikkat haritası?",
                target_options,
                index=0,
                help="Modelin en güçlü tahminine göre veya seçilen sınıfa göre heatmap üretilir.",
            )

        processed = st.session_state["last_processed"]
        probs = st.session_state["last_probs"]

        if target_choice == "Model tahmininin sınıfı":
            target_idx = int(np.argmax(probs))
        else:
            target_idx = int(target_choice.split(" — ")[0])

        try:
            with st.spinner("Grad-CAM hesaplanıyor..."):
                model = load_model()
                last_conv = get_last_conv_name()
                heatmap = make_gradcam_heatmap(
                    model, processed[np.newaxis],
                    last_conv_name=last_conv, pred_index=target_idx,
                )
                overlay = overlay_heatmap(processed, heatmap, alpha=0.5)
                raw_cmap_img = (heatmap * 255).astype(np.uint8)

            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                st.image((processed * 255).astype(np.uint8),
                         caption="Orijinal (ön işlenmiş)", width="stretch")
            with c2:
                st.image(raw_cmap_img, caption="Isı haritası (ham)",
                         width="stretch", clamp=True)
            with c3:
                st.image(overlay, caption="Üst üste bindirme", width="stretch")

            target_name = CLASS_NAMES.get(target_idx, f"Sınıf {target_idx}")
            conf = float(probs[target_idx]) * 100
            st.success(
                f"**Hedef sınıf:** {target_name} (#{target_idx}) — "
                f"Modelin bu sınıfa güveni: **%{conf:.1f}** · "
                f"Son konvolüsyon katmanı: `{last_conv}`"
            )
            st.markdown(
                '<div class="section-desc" style="margin-top:0.8rem">'
                '<b>Yorum:</b> İyi eğitilmiş bir model, kararını verirken levhadaki '
                'sembol / şekil bölgesine yoğunlaşmalıdır. Dağınık veya arka plana '
                'kayan dikkat, zayıf bir karar sinyalidir.'
                '</div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Grad-CAM hesaplanamadı: {e}")


# ===== Sekme 3: İstatistik ================================================
with tab_stats:
    st.markdown('<div class="section-title">Eğitim Geçmişi ve Sınıf İstatistikleri</div>',
                unsafe_allow_html=True)

    hist = load_history_json()
    if hist is None:
        st.info("Eğitim geçmişi bulunamadı. `python src/train.py` ile eğitim "
                "başlatıldıktan sonra bu sekmede grafikler görünecek.")
    else:
        meta = hist.get("_meta", {})
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            val_best = meta.get("best_val_accuracy", 0) * 100
            st.markdown(f'<div class="metric-box"><div class="metric-val">%{val_best:.2f}</div>'
                        f'<div class="metric-label">En İyi Val Doğruluğu</div></div>',
                        unsafe_allow_html=True)
        with mc2:
            n_epochs = len(hist.get("accuracy", []))
            st.markdown(f'<div class="metric-box"><div class="metric-val">{n_epochs}</div>'
                        f'<div class="metric-label">Epoch</div></div>', unsafe_allow_html=True)
        with mc3:
            elapsed_min = meta.get("elapsed_sec", 0) / 60
            st.markdown(f'<div class="metric-box"><div class="metric-val">{elapsed_min:.0f} dk</div>'
                        f'<div class="metric-label">Eğitim Süresi</div></div>', unsafe_allow_html=True)
        with mc4:
            bs = meta.get("batch_size", 64)
            st.markdown(f'<div class="metric-box"><div class="metric-val">{bs}</div>'
                        f'<div class="metric-label">Batch Size</div></div>', unsafe_allow_html=True)

        st.markdown("&nbsp;")

        g1, g2 = st.columns(2, gap="large")
        epochs_list = list(range(1, len(hist.get("accuracy", [])) + 1))
        if epochs_list:
            df_acc = pd.DataFrame({
                "epoch": epochs_list,
                "Eğitim": hist.get("accuracy", []),
                "Doğrulama": hist.get("val_accuracy", []),
            })
            df_loss = pd.DataFrame({
                "epoch": epochs_list,
                "Eğitim": hist.get("loss", []),
                "Doğrulama": hist.get("val_loss", []),
            })
            with g1:
                st.markdown('<div class="eyebrow">Doğruluk Eğrisi</div>', unsafe_allow_html=True)
                st.line_chart(df_acc.set_index("epoch"), height=300, color=["#1d4ed8", "#16a34a"])
            with g2:
                st.markdown('<div class="eyebrow">Kayıp Eğrisi</div>', unsafe_allow_html=True)
                st.line_chart(df_loss.set_index("epoch"), height=300, color=["#1d4ed8", "#16a34a"])

    # ── Model değerlendirme görselleri ──
    st.markdown("<hr style='border:none;border-top:1px solid var(--border);margin:1.2rem 0'>",
                unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Değerlendirme Görselleri</div>',
                unsafe_allow_html=True)
    cm_path = os.path.join(RESULT_DIR, "confusion_matrix.png")
    sp_path = os.path.join(RESULT_DIR, "sample_predictions_eval.png")
    if not os.path.isfile(sp_path):
        sp_path = os.path.join(RESULT_DIR, "sample_predictions.png")

    rc1, rc2 = st.columns([3, 2], gap="large")
    with rc1:
        st.markdown('<div class="eyebrow">Karışıklık Matrisi (test, normalize)</div>',
                    unsafe_allow_html=True)
        if os.path.isfile(cm_path):
            st.image(cm_path, width="stretch")
        else:
            st.info("confusion_matrix.png bulunamadı — `python src/evaluate.py` ile üretilir.")
    with rc2:
        st.markdown('<div class="eyebrow">Örnek Tahminler</div>', unsafe_allow_html=True)
        if os.path.isfile(sp_path):
            st.image(sp_path, width="stretch")
        else:
            st.info("Örnek tahmin görseli bulunamadı.")

    # ── Sınıf-bazlı performans tablosu ──
    df_pc = load_per_class_report()
    if df_pc is not None:
        st.markdown('<div class="eyebrow" style="margin-top:0.8rem">'
                    'Sınıf-Bazlı Performans (en zayıf F1 üstte)</div>', unsafe_allow_html=True)
        df_pc_sorted = df_pc.sort_values("F1", ascending=True).reset_index(drop=True)
        st.dataframe(
            df_pc_sorted, hide_index=True, width="stretch", height=320,
            column_config={
                "Precision": st.column_config.NumberColumn("Precision", format="%.2f"),
                "Recall": st.column_config.ProgressColumn(
                    "Recall", format="%.2f", min_value=0.0, max_value=1.0),
                "F1": st.column_config.ProgressColumn(
                    "F1", format="%.2f", min_value=0.0, max_value=1.0),
                "Destek": st.column_config.NumberColumn("Destek", format="%d"),
            },
        )
        worst = df_pc_sorted.iloc[0]
        st.caption(f"En zayıf sınıf: **{worst['Sınıf']}** (F1 = {worst['F1']:.2f}, "
                   f"{int(worst['Destek'])} test örneği). GTSRB'de tarihsel olarak en zor sınıftır.")

    # ── Sınıf dağılımı ──
    st.markdown("<hr style='border:none;border-top:1px solid var(--border);margin:1.2rem 0'>",
                unsafe_allow_html=True)
    st.markdown('<div class="section-title">Eğitim Seti Sınıf Dağılımı</div>', unsafe_allow_html=True)
    try:
        counts = load_class_counts()
        df_dist = pd.DataFrame({
            "Sınıf": [CLASS_NAMES.get(i, str(i)) for i in range(NUM_CLASSES)],
            "Örnek Sayısı": counts.tolist(),
        }).sort_values("Örnek Sayısı", ascending=False).reset_index(drop=True)

        dc1, dc2 = st.columns([3, 2], gap="large")
        with dc1:
            st.bar_chart(df_dist.set_index("Sınıf"), height=420, color="#1d4ed8")
        with dc2:
            st.markdown('<div class="eyebrow">En çok örnekli 5 sınıf</div>', unsafe_allow_html=True)
            st.dataframe(df_dist.head(5), hide_index=True, width="stretch")
            st.markdown('<div class="eyebrow">En az örnekli 5 sınıf</div>', unsafe_allow_html=True)
            st.dataframe(df_dist.tail(5), hide_index=True, width="stretch")

        total = int(df_dist["Örnek Sayısı"].sum())
        imbalance = int(df_dist["Örnek Sayısı"].max()) / max(1, int(df_dist["Örnek Sayısı"].min()))
        st.caption(
            f"Toplam {total:,} eğitim örneği · 43 sınıf · Dengesizlik oranı: "
            f"{imbalance:.1f}× (en çok / en az). Bu dengesizlik oversampling ve "
            f"opsiyonel class weights ile giderilir."
        )
    except Exception as e:
        st.info(f"Sınıf dağılımı hesaplanamadı: {e}")


# ===== Sekme 4: Mimari ====================================================
with tab_arch:
    st.markdown('<div class="section-title">CNN v3 Model Mimarisi</div>', unsafe_allow_html=True)
    try:
        model = load_model()
        total_params = model.count_params()
        n_layers = len(model.layers)
        input_shape = model.input_shape

        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{total_params/1e6:.2f}M</div>'
                        f'<div class="metric-label">Toplam Parametre</div></div>', unsafe_allow_html=True)
        with a2:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{n_layers}</div>'
                        f'<div class="metric-label">Katman Sayısı</div></div>', unsafe_allow_html=True)
        with a3:
            shape_str = f"{input_shape[1]}×{input_shape[2]}×{input_shape[3]}"
            st.markdown(f'<div class="metric-box"><div class="metric-val">{shape_str}</div>'
                        f'<div class="metric-label">Giriş Boyutu</div></div>', unsafe_allow_html=True)

        st.markdown("&nbsp;")
        st.markdown('<div class="eyebrow">Katman Yapısı</div>', unsafe_allow_html=True)

        for layer in model.layers:
            try:
                out_shape = layer.output.shape
                shape_str = "×".join(str(s) for s in out_shape[1:])
            except Exception:
                shape_str = "?"
            params = layer.count_params()
            params_str = f"{params:,} param" if params > 0 else "—"
            st.markdown(
                f'<div class="arch-block">'
                f'<span class="arch-name">{layer.name} '
                f'<span class="arch-type">({layer.__class__.__name__})</span></span>'
                f'<span class="arch-shape">{shape_str}</span>'
                f'<span class="arch-params">{params_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")


# ===== Sekme 5: Yöntem ====================================================
with tab_about:
    st.markdown('<div class="section-title">Yöntem ve Sunum Notları</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Bu bölüm, proje boyunca uygulanan derin öğrenme '
        'yöntemlerini özetler ve ders sunumu için referans olarak kullanılabilir.</div>',
        unsafe_allow_html=True,
    )

    method_list = [
        ("1. Veri Seti: GTSRB",
         "German Traffic Sign Recognition Benchmark — 43 sınıf, 26.640 eğitim + "
         "12.630 test gerçek dünya fotoğrafı. Belirgin sınıf dengesizliği mevcut "
         "(bazı sınıflar ~150, diğerleri ~2.000 örnek)."),
        ("2. Ön İşleme: CLAHE",
         "Contrast Limited Adaptive Histogram Equalization — LAB renk uzayında L "
         "(parlaklık) kanalına uygulanır. Düşük kontrastlı, gölgeli fotoğraflarda "
         "sembol detaylarını belirginleştirir."),
        ("3. Sınıf Dengeleme: Oversampling + Class Weights",
         "Az örnekli sınıflar augmentation ile hedef sayıya (1.200) çoğaltılır. "
         "Oversampling yalnızca eğitim bölümüne uygulanır; doğrulama setine sızıntı olmaz."),
        ("4. Mimari: VGG-stili 4 Bloklu CNN v3",
         "64→128→256→512 filtreli 4 konvolüsyon bloğu, BatchNormalization, "
         "SpatialDropout2D ve GlobalAveragePooling ile. Toplam ~4.98M eğitilebilir "
         "parametre. He_normal ağırlık başlatma."),
        ("5. Regülarizasyon: SpatialDropout + Label Smoothing + L2",
         "SpatialDropout2D feature-map bazlı düzenlileme sağlar (konvolüsyon katmanları "
         "için klasik dropout'tan uygun). Label smoothing (0.1) aşırı güveni azaltır. "
         "AdamW optimizer L2 weight decay uygular."),
        ("6. Öğrenme Oranı: Cosine Annealing + Warmup",
         "İlk 3 epoch linear warmup (0 → 1e-3), ardından cosine eğrisi ile "
         "1e-3 → 1e-6'ya yumuşak azalma. Plato'lardan kaçınmaya yardım eder."),
        ("7. Online Augmentation",
         "Her epoch'ta farklı dönüşümler: rastgele döndürme (±8%), zoom (±10%), "
         "parlaklık/kontrast (±18%), öteleme. Horizontal flip KULLANILMAZ "
         "(yön işaretleri bozulur)."),
        ("8. Değerlendirme: Test Set + Confusion Matrix + Per-Class Acc",
         "12.630 hiç görülmemiş test örneği üzerinde accuracy, weighted "
         "precision / recall / F1. Karışıklık matrisi ve sınıf bazlı doğruluk "
         "tablosu ile zayıf sınıflar tespit edilir."),
        ("9. Test-Time Augmentation (TTA)",
         "Çıkarımda aynı görüntünün birkaç varyantı (zoom in/out, parlaklık "
         "değişimleri) tahmin edilir ve olasılıklar ortalanır. Daha kararlı sonuçlar."),
        ("10. Yorumlanabilirlik: Grad-CAM",
         "Gradient-weighted Class Activation Mapping — son konvolüsyon katmanının "
         "aktivasyonlarını sınıf skorunun gradyanlarıyla ağırlıklandırır. Sonuç: "
         "modelin nereye baktığını gösteren sıcaklık haritası."),
    ]
    for title, desc in method_list:
        st.markdown(
            f'<div class="method-card">'
            f'<div class="method-title">{title}</div>'
            f'<div class="method-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border:none;border-top:1px solid var(--border);margin:1.2rem 0'>",
                unsafe_allow_html=True)
    st.markdown('<div class="section-title">Teknoloji Yığını</div>', unsafe_allow_html=True)
    tc1, tc2 = st.columns(2, gap="large")
    with tc1:
        st.markdown("""
        <div class="info-panel">
            <div class="info-row"><span class="info-key">Dil</span><span class="info-val">Python 3.9+</span></div>
            <div class="info-row"><span class="info-key">Framework</span><span class="info-val">TensorFlow / Keras 2.16</span></div>
            <div class="info-row"><span class="info-key">GPU</span><span class="info-val">Apple M4 Metal</span></div>
            <div class="info-row"><span class="info-key">Veri Seti</span><span class="info-val">GTSRB (Kaggle)</span></div>
        </div>
        """, unsafe_allow_html=True)
    with tc2:
        st.markdown("""
        <div class="info-panel">
            <div class="info-row"><span class="info-key">Ön İşleme</span><span class="info-val">OpenCV (CLAHE)</span></div>
            <div class="info-row"><span class="info-key">Arayüz</span><span class="info-val">Streamlit</span></div>
            <div class="info-row"><span class="info-key">Metrikler</span><span class="info-val">scikit-learn</span></div>
            <div class="info-row"><span class="info-key">Dağıtım</span><span class="info-val">Docker</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid var(--border);margin:1.2rem 0'>",
                unsafe_allow_html=True)
    st.markdown('<div class="section-title">Referanslar</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Stallkamp et al. (2012)** — The German Traffic Sign Recognition Benchmark
    - **Simonyan & Zisserman (2014)** — VGG: Very Deep Convolutional Networks
    - **Ioffe & Szegedy (2015)** — Batch Normalization
    - **Tompson et al. (2015)** — Efficient Object Localization (SpatialDropout)
    - **Selvaraju et al. (2017)** — Grad-CAM: Visual Explanations
    - **Szegedy et al. (2016)** — Label Smoothing (Inception-v3)
    - **Loshchilov & Hutter (2019)** — Decoupled Weight Decay Regularization (AdamW)
    """)

st.markdown('<div class="footer">Trafik İşareti Tanıma Sistemi · GTSRB · CNN v3 · '
            'TensorFlow · Grad-CAM · Docker</div>', unsafe_allow_html=True)
