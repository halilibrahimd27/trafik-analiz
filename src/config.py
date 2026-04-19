"""
Proje yapılandırma ayarları
"""

import os

# ── Klasör yolları ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

for d in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Model parametreleri ─────────────────────────────────────────────────────
IMG_SIZE   = (64, 64)      # Giriş görüntü boyutu (v3: 48 → 64, daha fazla detay)
BATCH_SIZE = 64
EPOCHS     = 45
NUM_CLASSES = 43           # GTSRB sınıf sayısı
LEARNING_RATE = 1e-3
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 1e-4

# ── Dosya adları ────────────────────────────────────────────────────────────
MODEL_PATH        = os.path.join(MODEL_DIR, "trafik_model.keras")
HISTORY_PATH      = os.path.join(RESULT_DIR, "training_history.png")
HISTORY_JSON_PATH = os.path.join(RESULT_DIR, "training_history.json")
CONF_MATRIX_PATH  = os.path.join(RESULT_DIR, "confusion_matrix.png")
CLASS_DIST_PATH   = os.path.join(RESULT_DIR, "class_distribution.png")

# ── GTSRB sınıf adları (Türkçe) ─────────────────────────────────────────────
CLASS_NAMES = {
    0:  "Hız Limiti (20 km/s)",
    1:  "Hız Limiti (30 km/s)",
    2:  "Hız Limiti (50 km/s)",
    3:  "Hız Limiti (60 km/s)",
    4:  "Hız Limiti (70 km/s)",
    5:  "Hız Limiti (80 km/s)",
    6:  "Hız Limiti Sonu (80 km/s)",
    7:  "Hız Limiti (100 km/s)",
    8:  "Hız Limiti (120 km/s)",
    9:  "Geçiş Yasağı",
    10: "3.5 Ton Üzeri Araç Yasağı",
    11: "Öncelikli Yol Kavşağı",
    12: "Öncelikli Yol",
    13: "Yol Ver",
    14: "Dur",
    15: "Araç Giremez",
    16: "3.5 Ton Üzeri Araç Giremez",
    17: "Girilmez",
    18: "Genel Tehlike",
    19: "Sola Tehlikeli Viraj",
    20: "Sağa Tehlikeli Viraj",
    21: "Ardışık Virajlar",
    22: "Engebeli Yol",
    23: "Kaygan Yol",
    24: "Sağdan Daralan Yol",
    25: "Yol Çalışması",
    26: "Trafik Işıkları",
    27: "Yaya Geçidi",
    28: "Çocuk Geçidi",
    29: "Bisiklet Geçidi",
    30: "Buzlanma / Kar",
    31: "Yaban Hayvanı Geçidi",
    32: "Tüm Yasakların Sonu",
    33: "Sağa Dönünüz",
    34: "Sola Dönünüz",
    35: "İleri Gidiniz",
    36: "İleri veya Sağa",
    37: "İleri veya Sola",
    38: "Sağdan Geçiniz",
    39: "Soldan Geçiniz",
    40: "Dönel Kavşak",
    41: "Geçiş Yasağı Sonu",
    42: "3.5 Ton Üzeri Yasak Sonu",
}
