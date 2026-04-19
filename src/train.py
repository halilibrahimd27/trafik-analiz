"""
Trafik isareti tanima modeli — gelistirilmis egitim hatti (v3).

Ozellikler:
  - 4 bloklu CNN v3 (64x64 giris, ~4M param)
  - Label smoothing (0.1) + AdamW
  - Oversampling ile sinif dengeleme
  - Online augmentation
  - CLAHE kontrast iyilestirme
  - Class weights (opsiyonel)
  - Cosine annealing LR schedule + warmup
  - Training history JSON olarak kaydedilir

Kullanim:
    python src/train.py
    python src/train.py --epochs 45 --augment
    python src/train.py --no-balanced --augment --class-weights
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
import time

import tensorflow as tf
import numpy as np

from src.config import (
    DATA_DIR, MODEL_DIR, RESULT_DIR, MODEL_PATH, HISTORY_JSON_PATH,
    IMG_SIZE, BATCH_SIZE, NUM_CLASSES, CLASS_NAMES,
    LABEL_SMOOTHING, WEIGHT_DECAY,
)
from src.model import build_model
from src.dataset import (
    load_gtsrb_from_directory,
    load_balanced_dataset,
    load_test_dataset,
    compute_class_weights,
    get_class_counts,
)


def _print_box(lines, width=65):
    border = "=" * width
    print(f"\n+{border}+")
    for line in lines:
        padding = width - len(line)
        print(f"| {line}{' ' * max(0, padding - 1)}|")
    print(f"+{border}+\n")


class CosineAnnealingSchedule(tf.keras.callbacks.Callback):
    """Cosine annealing learning rate + linear warmup."""
    def __init__(self, initial_lr=1e-3, min_lr=1e-6, warmup_epochs=3, total_epochs=45):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        try:
            self.model.optimizer.learning_rate.assign(lr)
        except AttributeError:
            self.model.optimizer.lr.assign(lr)
        print(f"\n  [LR] Epoch {epoch+1}: lr = {lr:.2e}")


def train(args):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    mode = "Dengelenmis + Aug" if args.balanced else ("Augmented" if args.augment else "Standart")

    _print_box([
        "  TRAFIK ISARETI TANIMA - EGITIM BASLIYOR (v3)",
        f"  Mod        : {mode}",
        f"  Epoch      : {args.epochs}",
        f"  Batch Size : {args.batch_size}",
        f"  Mimari     : CNN v3 (4 blok, SpatialDropout, AdamW)",
        f"  Goruntu    : {IMG_SIZE[0]}x{IMG_SIZE[1]} piksel",
        f"  Sinif      : {NUM_CLASSES}",
        f"  Label Smoothing: {LABEL_SMOOTHING}",
        f"  GPU        : Apple Metal (varsa)",
    ])

    # -- Veri yukleme --------------------------------------------------------
    print("[Egitim] Veri yukleniyor...")

    if args.balanced:
        train_ds, val_ds = load_balanced_dataset(
            DATA_DIR, IMG_SIZE, args.batch_size,
            target_per_class=args.target_per_class,
            apply_clahe=True, online_augment=True,
        )
    else:
        train_ds, val_ds = load_gtsrb_from_directory(
            DATA_DIR, IMG_SIZE, args.batch_size,
            augment=args.augment,
        )

    test_ds = load_test_dataset(DATA_DIR, IMG_SIZE, args.batch_size, NUM_CLASSES)

    # -- Class weights -------------------------------------------------------
    class_weights = None
    if args.class_weights and not args.balanced:
        print("[Egitim] Sinif agirliklari hesaplaniyor...")
        class_weights = compute_class_weights(DATA_DIR, NUM_CLASSES)
        min_w = min(class_weights.values())
        max_w = max(class_weights.values())
        print(f"  [Agirlik] Min: {min_w:.2f}, Max: {max_w:.2f}")
    elif args.balanced:
        print("[Egitim] Balanced mod — class weights atlaniyor (veri zaten dengeli)")

    # -- Sinif dagilimi gorsellestir -----------------------------------------
    try:
        from src.visualize import plot_class_distribution
        plot_class_distribution(
            DATA_DIR,
            [CLASS_NAMES[i] for i in range(NUM_CLASSES)],
            os.path.join(RESULT_DIR, "class_distribution.png"),
        )
    except Exception as e:
        print(f"[Uyari] Sinif dagilimi ciz hatasi: {e}")

    # -- Model ---------------------------------------------------------------
    print("\n[Egitim] Model olusturuluyor (CNN v3 - 4 blok)...")
    model = build_model(
        NUM_CLASSES, IMG_SIZE,
        learning_rate=args.lr,
        label_smoothing=LABEL_SMOOTHING,
        weight_decay=WEIGHT_DECAY,
    )
    model.summary(line_length=85)

    # -- Callbacks -----------------------------------------------------------
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        CosineAnnealingSchedule(
            initial_lr=args.lr,
            min_lr=1e-6,
            warmup_epochs=3,
            total_epochs=args.epochs,
        ),
    ]

    # -- Egitim --------------------------------------------------------------
    print(f"\n[Egitim] {args.epochs} epoch basliyor...")
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    elapsed = time.time() - t0

    best_val_acc = max(history.history.get("val_accuracy", [0]))
    print(f"\n[Egitim] En iyi val dogrulugu: %{best_val_acc*100:.2f} | Sure: {elapsed/60:.1f} dk")

    # -- Training history JSON ------------------------------------------------
    try:
        hist_to_save = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        hist_to_save["_meta"] = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_size": list(IMG_SIZE),
            "label_smoothing": LABEL_SMOOTHING,
            "weight_decay": WEIGHT_DECAY,
            "mode": mode,
            "elapsed_sec": elapsed,
            "best_val_accuracy": float(best_val_acc),
        }
        with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(hist_to_save, f, indent=2)
        print(f"[Egitim] Egitim gecmisi JSON kaydedildi: {HISTORY_JSON_PATH}")
    except Exception as e:
        print(f"[Uyari] JSON kaydetme hatasi: {e}")

    # -- En iyi modeli kaydet ------------------------------------------------
    best_path = os.path.join(MODEL_DIR, "best_model.keras")
    if os.path.isfile(best_path):
        best_model = tf.keras.models.load_model(best_path)
    else:
        best_model = model
    best_model.save(MODEL_PATH)
    print(f"[Egitim] Final model kaydedildi: {MODEL_PATH}")

    # -- Gorsellestirme ------------------------------------------------------
    try:
        from src.visualize import plot_training_history, plot_sample_predictions
        plot_training_history(history.history, os.path.join(RESULT_DIR, "training_history.png"))
        class_names_list = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
        plot_sample_predictions(best_model, val_ds, class_names_list,
                                os.path.join(RESULT_DIR, "sample_predictions.png"))
        print("[Egitim] Grafikler olusturuldu.")
    except Exception as e:
        print(f"[Uyari] Gorsellestirme hatasi: {e}")

    # -- Son degerlendirme ---------------------------------------------------
    eval_ds   = test_ds if test_ds is not None else val_ds
    eval_name = "Test"  if test_ds is not None else "Dogrulama"
    eval_results = best_model.evaluate(eval_ds, verbose=0, return_dict=True)
    eval_loss = eval_results.get("loss", 0.0)
    eval_acc  = eval_results.get("accuracy", 0.0)
    eval_top3 = eval_results.get("top3_acc", 0.0)

    _print_box([
        "  EGITIM TAMAMLANDI",
        f"  {eval_name} Dogrulugu  : %{eval_acc * 100:.2f}",
        f"  {eval_name} Top-3 Acc  : %{eval_top3 * 100:.2f}",
        f"  {eval_name} Kaybi      : {eval_loss:.4f}",
        f"  En Iyi Val Acc      : %{best_val_acc*100:.2f}",
        f"  Egitim Suresi       : {elapsed/60:.1f} dk",
        f"  Model               : {MODEL_PATH}",
        f"  Sonuclar            : {RESULT_DIR}",
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Trafik isareti CNN v3 egitimi")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, dest="batch_size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Baslangic ogrenme orani")
    parser.add_argument("--augment", action="store_true", help="Online augmentation (standart mod)")
    parser.add_argument("--balanced", action="store_true", default=True,
                        help="Dengelenmis veri seti (oversampling + CLAHE)")
    parser.add_argument("--no-balanced", action="store_false", dest="balanced")
    parser.add_argument("--class-weights", action="store_true", default=False, dest="class_weights")
    parser.add_argument("--target-per-class", type=int, default=1200, dest="target_per_class",
                        help="Oversampling hedef ornek sayisi")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
