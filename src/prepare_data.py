"""
GTSRB veri setini torchvision ile indirir ve
data/Train/<sinif_id>/ / data/Test/ yapısına kopyalar.

Kullanım:
    python src/prepare_data.py
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR

TRAIN_DIR = os.path.join(DATA_DIR, "Train")
TEST_DIR  = os.path.join(DATA_DIR, "Test")
TMP_DIR   = os.path.join(DATA_DIR, "_torchvision_tmp")


def download_and_prepare():
    # Zaten hazırsa atla
    if os.path.isdir(TRAIN_DIR) and len(os.listdir(TRAIN_DIR)) >= 43:
        print("[Veri] Train/ zaten hazır, atlanıyor.")
        return
    if os.path.isdir(TEST_DIR) and len(os.listdir(TEST_DIR)) > 0:
        pass  # test hazır ama train değil, devam et

    try:
        from torchvision.datasets import GTSRB
        from PIL import Image
        import numpy as np
    except ImportError:
        print("[Hata] torchvision veya Pillow bulunamadı.")
        print("  pip install torchvision Pillow")
        sys.exit(1)

    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # ── Eğitim seti ──────────────────────────────────────────────────────────
    print("[Veri] Eğitim seti indiriliyor (torchvision)... (~300 MB)")
    train_ds = GTSRB(root=TMP_DIR, split="train", download=True)

    print(f"[Veri] {len(train_ds)} eğitim örneği bulundu.")
    print("[Veri] Görüntüler Train/<sinif>/ klasörlerine kopyalanıyor...")

    counters = {}
    for idx, (img, label) in enumerate(train_ds):
        class_dir = os.path.join(TRAIN_DIR, str(label))
        os.makedirs(class_dir, exist_ok=True)
        count = counters.get(label, 0)
        out_path = os.path.join(class_dir, f"{count:05d}.png")
        img.save(out_path)
        counters[label] = count + 1
        if idx % 5000 == 0:
            print(f"  {idx}/{len(train_ds)} işlendi...")

    print(f"[Veri] Eğitim seti hazır: {sum(counters.values())} görüntü, {len(counters)} sınıf")

    # ── Test seti ────────────────────────────────────────────────────────────
    if not os.path.isdir(TEST_DIR) or len(os.listdir(TEST_DIR)) == 0:
        print("[Veri] Test seti indiriliyor...")
        test_ds = GTSRB(root=TMP_DIR, split="test", download=True)
        print(f"[Veri] {len(test_ds)} test örneği bulundu.")

        os.makedirs(TEST_DIR, exist_ok=True)
        labels = []
        for idx, (img, label) in enumerate(test_ds):
            out_path = os.path.join(TEST_DIR, f"{idx:05d}.png")
            img.save(out_path)
            labels.append(label)
            if idx % 3000 == 0:
                print(f"  {idx}/{len(test_ds)} işlendi...")

        # Test etiketlerini CSV olarak kaydet
        import csv
        csv_path = os.path.join(DATA_DIR, "Test.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Path", "ClassId"])
            for i, lbl in enumerate(labels):
                writer.writerow([os.path.join("Test", f"{i:05d}.png"), lbl])
        print(f"[Veri] Test etiketleri kaydedildi: {csv_path}")

    # ── Geçici dosyaları temizle ─────────────────────────────────────────────
    try:
        shutil.rmtree(TMP_DIR)
        print("[Veri] Geçici dosyalar temizlendi.")
    except Exception:
        pass

    print("\n[Veri] Veri seti hazırlama TAMAMLANDI.")
    print(f"  Train klasörü : {TRAIN_DIR}")
    print(f"  Test klasörü  : {TEST_DIR}")


if __name__ == "__main__":
    download_and_prepare()
