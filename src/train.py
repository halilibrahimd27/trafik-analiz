"""
Trafik işareti tanıma modeli — özel CNN eğitim hattı.

Kullanım:
    python src/train.py
    python src/train.py --epochs 30
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import tensorflow as tf

from src.config import (
    DATA_DIR, MODEL_DIR, RESULT_DIR, MODEL_PATH,
    IMG_SIZE, BATCH_SIZE, NUM_CLASSES, CLASS_NAMES,
)
from src.model import build_model
from src.dataset import load_gtsrb_from_directory, load_test_dataset


def _print_box(lines, width=60):
    border = "═" * width
    print(f"\n╔{border}╗")
    for line in lines:
        padding = width - len(line)
        print(f"║ {line}{' ' * max(0, padding - 1)}║")
    print(f"╚{border}╝\n")


def train(args):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    _print_box([
        "  TRAFİK İŞARETİ TANIMA — EĞİTİM BAŞLIYOR  ",
        f"  Epoch      : {args.epochs}",
        f"  Batch Size : {args.batch_size}",
        f"  Mimari     : Özel CNN (VGG-stili, sıfırdan)",
        f"  Görüntü    : {IMG_SIZE[0]}x{IMG_SIZE[1]} piksel",
        f"  Sınıf      : {NUM_CLASSES}",
        f"  GPU        : Apple M4 Metal",
    ])

    # ── Veri yükleme ─────────────────────────────────────────────────────────
    print("[Eğitim] Veri yükleniyor...")
    train_ds, val_ds = load_gtsrb_from_directory(DATA_DIR, IMG_SIZE, args.batch_size)
    test_ds = load_test_dataset(DATA_DIR, IMG_SIZE, args.batch_size, NUM_CLASSES)

    # ── Model ────────────────────────────────────────────────────────────────
    print("\n[Eğitim] Model oluşturuluyor (Özel CNN)...")
    model = build_model(NUM_CLASSES, IMG_SIZE, learning_rate=1e-3)
    model.summary(line_length=80)

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # ── Eğitim (augmentation olmadan, temiz pipeline) ─────────────────────────
    print(f"\n[Eğitim] {args.epochs} epoch başlıyor...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    best_val_acc = max(history.history.get("val_accuracy", [0]))
    print(f"\n[Eğitim] En iyi val doğruluğu: %{best_val_acc*100:.2f}")

    # ── En iyi modeli kaydet ──────────────────────────────────────────────────
    best_path = os.path.join(MODEL_DIR, "best_model.keras")
    if os.path.isfile(best_path):
        best_model = tf.keras.models.load_model(best_path)
    else:
        best_model = model
    best_model.save(MODEL_PATH)
    print(f"[Eğitim] Final model kaydedildi: {MODEL_PATH}")

    # ── Görselleştirme ────────────────────────────────────────────────────────
    try:
        from src.visualize import plot_training_history, plot_sample_predictions
        plot_training_history(history.history, os.path.join(RESULT_DIR, "training_history.png"))
        class_names_list = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
        plot_sample_predictions(best_model, val_ds, class_names_list,
                                os.path.join(RESULT_DIR, "sample_predictions.png"))
        print("[Eğitim] Grafikler oluşturuldu.")
    except Exception as e:
        print(f"[Uyarı] Görselleştirme hatası: {e}")

    # ── Son değerlendirme ─────────────────────────────────────────────────────
    eval_ds   = test_ds if test_ds is not None else val_ds
    eval_name = "Test"  if test_ds is not None else "Doğrulama"
    eval_loss, eval_acc = best_model.evaluate(eval_ds, verbose=0)

    _print_box([
        "  EĞİTİM TAMAMLANDI  ",
        f"  {eval_name} Doğruluğu  : %{eval_acc * 100:.2f}",
        f"  {eval_name} Kaybı      : {eval_loss:.4f}",
        f"  En İyi Val Acc      : %{best_val_acc*100:.2f}",
        f"  Model               : {MODEL_PATH}",
        f"  Sonuçlar            : {RESULT_DIR}",
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Trafik işareti CNN eğitimi")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, dest="batch_size")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
