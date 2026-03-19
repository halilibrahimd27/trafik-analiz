"""
GTSRB veri setini indir ve hazırla.

Kaynak: https://benchmark.ini.rub.de/gtsrb_dataset.html
Kaggle üzerinden de indirilebilir:
  kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
"""

import os
import sys
import zipfile
import urllib.request
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR

# Kaggle alternatifi varsa kullan, yoksa tensorflow_datasets ile indir
def download_via_tensorflow_datasets():
    """tensorflow-datasets üzerinden GTSRB'yi indir."""
    try:
        import tensorflow_datasets as tfds
        print("tensorflow-datasets ile GTSRB indiriliyor...")
        ds, info = tfds.load(
            "gtsrb",
            split=["train", "test"],
            as_supervised=True,
            with_info=True,
            data_dir=DATA_DIR,
        )
        print(f"Veri seti indirildi! Eğitim: {info.splits['train'].num_examples}, "
              f"Test: {info.splits['test'].num_examples}")
        return True
    except Exception as e:
        print(f"tensorflow-datasets hatası: {e}")
        return False


def check_data_exists():
    """Verinin daha önce indirilip indirilmediğini kontrol et."""
    train_dir = os.path.join(DATA_DIR, "Train")
    test_dir  = os.path.join(DATA_DIR, "Test")
    return os.path.isdir(train_dir) and os.path.isdir(test_dir)


def print_instructions():
    """Manuel indirme talimatlarını yazdır."""
    print("\n" + "="*60)
    print("VERİ SETİ İNDİRME TALİMATLARI")
    print("="*60)
    print("""
Seçenek 1 — Kaggle CLI (önerilir):
  pip install kaggle
  kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
  unzip gtsrb-german-traffic-sign.zip -d data/

Seçenek 2 — Kaggle web sitesi:
  https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
  → İndir → data/ klasörüne çıkart

Beklenen klasör yapısı:
  data/
  ├── Train/
  │   ├── 0/   (43 sınıf, 0'dan 42'ye)
  │   ├── 1/
  │   └── ...
  └── Test/
      └── *.png  (+ Test.csv)

Veri indirildikten sonra şunu çalıştır:
  python src/train.py
""")
    print("="*60)


if __name__ == "__main__":
    if check_data_exists():
        print("Veri seti zaten mevcut. Eğitime geçebilirsiniz: python src/train.py")
    else:
        print("Veri seti bulunamadı.")
        print_instructions()
