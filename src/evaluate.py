"""
Trafik işareti tanıma modeli — değerlendirme scripti.

Kullanım:
    python src/evaluate.py
    python src/evaluate.py --model models/trafik_model.keras
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np

from src.config import (
    DATA_DIR,
    MODEL_PATH,
    RESULT_DIR,
    IMG_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
)


# ── Yardımcı: kutu yazdırma ───────────────────────────────────────────────────

def _print_box(lines, width=65):
    border = "═" * width
    print(f"\n╔{border}╗")
    for line in lines:
        pad = width - len(line)
        print(f"║ {line}{' ' * max(pad - 1, 0)}║")
    print(f"╚{border}╝\n")


# ── Değerlendirme ─────────────────────────────────────────────────────────────

def evaluate(model_path: str) -> None:
    """Modeli test veri kümesi üzerinde değerlendirir.

    Yapılan işlemler:
        1. Model yükleme
        2. Test verisi yükleme
        3. Genel metrikler (accuracy, precision, recall, f1)
        4. Karışıklık matrisi (results/ altına kaydedilir)
        5. Örnek tahmin ızgarası
        6. Sınıf bazlı doğruluk tablosu (en kötüden en iyiye)
        7. Tam classification report → results/classification_report.txt

    Args:
        model_path: Değerlendirilecek Keras model dosyasının yolu.
    """
    import tensorflow as tf
    from sklearn.metrics import (  # type: ignore
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )
    from src.dataset import load_test_arrays

    def load_test_data(data_dir, img_size):
        """Test setini CLAHE ile on islenmis numpy array olarak dondurur."""
        return load_test_arrays(data_dir, img_size=img_size, apply_clahe=True)
    from src.visualize import (
        plot_confusion_matrix,
        plot_sample_predictions,
    )

    os.makedirs(RESULT_DIR, exist_ok=True)

    # ── Model yükleme ────────────────────────────────────────────────────────
    if not os.path.isfile(model_path):
        print(
            f"[Evaluate] Hata: Model bulunamadı → {model_path}\n"
            "Önce modeli eğitin: python src/train.py"
        )
        sys.exit(1)

    print(f"[Evaluate] Model yükleniyor: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # ── Test verisi yükleme ──────────────────────────────────────────────────
    print("[Evaluate] Test verisi yükleniyor...")
    test_result = load_test_data(data_dir=DATA_DIR, img_size=IMG_SIZE)

    if test_result is None:
        print(
            "[Evaluate] Test verisi bulunamadı. Doğrulama seti üzerinde değerlendirme yapılıyor..."
        )
        _evaluate_on_val(model)
        return

    x_test, y_true = test_result
    print(f"[Evaluate] Test örneği: {len(y_true):,}")

    # ── Tahminler ────────────────────────────────────────────────────────────
    print("[Evaluate] Tahminler hesaplanıyor...")
    preds_proba = model.predict(x_test, batch_size=64, verbose=1)
    y_pred      = np.argmax(preds_proba, axis=-1)

    # ── Genel metrikler ──────────────────────────────────────────────────────
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    _print_box([
        "  DEĞERLENDİRME SONUÇLARI (Test Seti)  ",
        f"  Doğruluk (Accuracy)        : {acc * 100:.2f}%",
        f"  Kesinlik (Precision)        : {prec * 100:.2f}%",
        f"  Duyarlılık (Recall)         : {rec * 100:.2f}%",
        f"  F1 Skoru (Weighted)         : {f1 * 100:.2f}%",
        f"  Test Örnek Sayısı           : {len(y_true):,}",
    ])

    # ── Classification report (dosyaya kaydet) ───────────────────────────────
    class_names_list = [CLASS_NAMES.get(i, f"Sinif_{i}") for i in range(NUM_CLASSES)]
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names_list,
        zero_division=0,
    )

    report_path = os.path.join(RESULT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("GTSRB Trafik İşareti Tanıma — Sınıflandırma Raporu\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Doğruluk  : {acc * 100:.2f}%\n")
        f.write(f"Kesinlik  : {prec * 100:.2f}%\n")
        f.write(f"Duyarlılık: {rec * 100:.2f}%\n")
        f.write(f"F1 Skoru  : {f1 * 100:.2f}%\n\n")
        f.write(report)

    print(f"[Evaluate] Sınıflandırma raporu kaydedildi: {report_path}")

    # ── Karışıklık matrisi ───────────────────────────────────────────────────
    cm_path = os.path.join(RESULT_DIR, "confusion_matrix.png")
    print("[Evaluate] Karışıklık matrisi oluşturuluyor...")
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names_list,
        save_path=cm_path,
    )

    # ── Örnek tahmin ızgarası ────────────────────────────────────────────────
    print("[Evaluate] Örnek tahmin ızgarası oluşturuluyor...")

    # Küçük bir tf.data.Dataset oluştur (görselleştirme için)
    sample_ds = (
        tf.data.Dataset
        .from_tensor_slices((x_test[:64], tf.one_hot(y_true[:64], NUM_CLASSES)))
        .batch(16)
    )

    preds_path = os.path.join(RESULT_DIR, "sample_predictions_eval.png")
    plot_sample_predictions(
        model=model,
        dataset=sample_ds,
        class_names=class_names_list,
        save_path=preds_path,
    )

    # ── Sınıf bazlı doğruluk tablosu ────────────────────────────────────────
    _print_per_class_table(y_true, y_pred, class_names_list)


def _evaluate_on_val(model) -> None:
    """Test verisi yoksa doğrulama seti üzerinde basit değerlendirme yapar."""
    from src.dataset import load_gtsrb_from_directory
    from src.config import BATCH_SIZE

    print("[Evaluate] Doğrulama seti yükleniyor...")
    try:
        _, val_ds = load_gtsrb_from_directory(DATA_DIR, IMG_SIZE, BATCH_SIZE)
        loss, acc = model.evaluate(val_ds, verbose=1)
        _print_box([
            "  DEĞERLENDİRME (Doğrulama Seti)  ",
            f"  Kayıp    : {loss:.4f}",
            f"  Doğruluk : {acc * 100:.2f}%",
        ])
    except FileNotFoundError as exc:
        print(f"[Evaluate] Hata: {exc}")


def _print_per_class_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names,
) -> None:
    """Her sınıf için doğruluk oranını hesaplar ve en kötüden en iyiye yazdırır."""
    print("\n[Evaluate] Sınıf Bazlı Doğruluk Tablosu (En Kötüden → En İyiye)\n")
    header = f"{'Sıra':<5} {'Sınıf Adı':<40} {'Doğru':>6} {'Toplam':>7} {'Doğruluk':>9}"
    print(header)
    print("-" * len(header))

    per_class = []
    for cls_idx, cls_name in enumerate(class_names):
        mask    = y_true == cls_idx
        total   = int(mask.sum())
        if total == 0:
            continue
        correct = int((y_pred[mask] == cls_idx).sum())
        acc_cls = correct / total
        per_class.append((cls_idx, cls_name, correct, total, acc_cls))

    # En kötü performanstan en iyiye sırala
    per_class.sort(key=lambda x: x[4])

    for rank, (cls_idx, cls_name, correct, total, acc_cls) in enumerate(per_class, 1):
        name_trunc = cls_name[:38] + ".." if len(cls_name) > 38 else cls_name
        bar = "█" * int(acc_cls * 20)
        print(
            f"{rank:<5} {name_trunc:<40} {correct:>6} {total:>7} "
            f"{acc_cls * 100:>8.1f}%  {bar}"
        )


# ── Giriş noktası ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trafik işareti tanıma modeli değerlendirmesi"
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help=f"Keras model dosyası (varsayılan: {MODEL_PATH})",
    )
    args = parser.parse_args()
    evaluate(model_path=args.model)


if __name__ == "__main__":
    main()
