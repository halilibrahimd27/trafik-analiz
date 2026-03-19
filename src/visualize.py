"""
Trafik işareti tanıma projesi — görselleştirme fonksiyonları.

Kullanım:
    from src.visualize import (
        plot_training_history,
        plot_confusion_matrix,
        plot_sample_predictions,
        plot_class_distribution,
    )
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # GUI olmayan ortamlarda güvenli arka uç
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config import RESULT_DIR

# Seaborn isteğe bağlı; yoksa matplotlib ile devam et
try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False
    print("[Visualize] Uyarı: seaborn bulunamadı, matplotlib kullanılacak.")


# ── Ortak yardımcılar ────────────────────────────────────────────────────────

def _ensure_result_dir() -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)


def _save_and_close(fig: plt.Figure, save_path: str) -> None:
    """Figürü kaydeder ve belleği serbest bırakır."""
    _ensure_result_dir()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[Visualize] Kaydedildi: {save_path}")


# ── 1. Eğitim geçmişi grafiği ────────────────────────────────────────────────

def plot_training_history(
    history,
    save_path=None,
) -> None:
    """Eğitim ve doğrulama doğruluğu / kaybını 2x1 alt grafik olarak çizer.

    Args:
        history:   Keras History nesnesi veya ``{'accuracy': [...], ...}`` sözlüğü.
        save_path: Kaydedilecek dosya yolu. None ise ``results/training_history.png``.
    """
    if save_path is None:
        save_path = os.path.join(RESULT_DIR, "training_history.png")

    # History nesnesi veya sözlük desteklenir
    hist = history.history if hasattr(history, "history") else history

    epochs = range(1, len(hist.get("accuracy", hist.get("acc", []))) + 1)

    acc_key     = "accuracy"     if "accuracy"     in hist else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in hist else "val_acc"
    loss_key    = "loss"
    val_loss_key= "val_loss"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Eğitim Geçmişi", fontsize=15, fontweight="bold")

    # ── Doğruluk ────────────────────────────────────────────────────────────
    ax1.plot(epochs, hist[acc_key],     "b-o", label="Eğitim", linewidth=2, markersize=4)
    if val_acc_key in hist:
        ax1.plot(epochs, hist[val_acc_key], "r-s", label="Doğrulama", linewidth=2, markersize=4)
    ax1.set_title("Doğruluk (Accuracy)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Doğruluk")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # ── Kayıp ───────────────────────────────────────────────────────────────
    ax2.plot(epochs, hist[loss_key],     "b-o", label="Eğitim", linewidth=2, markersize=4)
    if val_loss_key in hist:
        ax2.plot(epochs, hist[val_loss_key], "r-s", label="Doğrulama", linewidth=2, markersize=4)
    ax2.set_title("Kayıp (Loss)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Kayıp")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_and_close(fig, save_path)


# ── 2. Karışıklık matrisi ─────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    save_path=None,
    normalize=True,
) -> None:
    """Seaborn ısı haritası olarak karışıklık matrisi çizer.

    Args:
        y_true:       Gerçek sınıf indeksleri (1-D).
        y_pred:       Tahmin edilen sınıf indeksleri (1-D).
        class_names:  Her sınıfa ait etiket listesi.
        save_path:    Kaydedilecek dosya yolu.
        normalize:    True ise satır bazında normalize edilir (oran gösterilir).
    """
    if save_path is None:
        save_path = os.path.join(RESULT_DIR, "confusion_matrix.png")

    from sklearn.metrics import confusion_matrix  # type: ignore

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fmt = ".2f"
        title = "Karışıklık Matrisi (normalize)"
    else:
        cm_display = cm
        fmt = "d"
        title = "Karışıklık Matrisi"

    n_classes = len(class_names)
    fig_size = max(12, n_classes * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    if _HAS_SNS:
        sns.heatmap(
            cm_display,
            annot=n_classes <= 20,          # Küçük matrisler için hücre değeri yaz
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            linewidths=0.3,
        )
    else:
        im = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=11)
    ax.set_ylabel("Gerçek Sınıf", fontsize=11)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    _save_and_close(fig, save_path)


# ── 3. Örnek tahmin ızgarası ─────────────────────────────────────────────────

def plot_sample_predictions(
    model,
    dataset,
    class_names,
    save_path=None,
    n: int = 16,
) -> None:
    """4×4 ızgara içinde örnek görüntüler ve tahminleri gösterir.

    Doğru tahminler yeşil, yanlış tahminler kırmızı çerçeve ile işaretlenir.

    Args:
        model:       Keras modeli.
        dataset:     ``(image, label)`` tensörleri döndüren tf.data.Dataset.
        class_names: Sınıf adı listesi.
        save_path:   Kaydedilecek dosya yolu.
        n:           Gösterilecek görüntü sayısı (en fazla 16 önerilir).
    """
    import tensorflow as tf

    if save_path is None:
        save_path = os.path.join(RESULT_DIR, "sample_predictions.png")

    images_list: list = []
    true_list: list   = []

    # Dataset'ten yeterli örnek topla
    for batch_imgs, batch_labels in dataset:
        images_list.append(batch_imgs.numpy())
        # one-hot ise argmax, skaler ise direkt kullan
        lbl = batch_labels.numpy()
        if lbl.ndim > 1:
            lbl = np.argmax(lbl, axis=-1)
        true_list.append(lbl)
        if sum(len(x) for x in images_list) >= n:
            break

    if not images_list:
        print("[Visualize] Uyarı: Dataset boş, örnek tahmin grafiği oluşturulamadı.")
        return

    images = np.concatenate(images_list, axis=0)[:n]
    y_true = np.concatenate(true_list,  axis=0)[:n]

    # Tahminler
    preds = model.predict(images, verbose=0)
    y_pred = np.argmax(preds, axis=-1)

    cols  = 4
    rows  = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle("Örnek Tahminler (Yeşil=Doğru / Kırmızı=Yanlış)", fontsize=13, fontweight="bold")

    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, ax in enumerate(axes):
        if i < len(images):
            img = images[i]
            # [0,1] aralığında olduğunu varsay; clip güvenlik için
            ax.imshow(np.clip(img, 0, 1))

            true_name = class_names[y_true[i]] if y_true[i] < len(class_names) else str(y_true[i])
            pred_name = class_names[y_pred[i]] if y_pred[i] < len(class_names) else str(y_pred[i])
            correct   = y_true[i] == y_pred[i]
            color     = "green" if correct else "red"
            conf      = preds[i][y_pred[i]] * 100

            ax.set_title(
                f"G: {true_name}\nT: {pred_name} ({conf:.1f}%)",
                fontsize=7,
                color=color,
                fontweight="bold",
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
        ax.axis("off")

    plt.tight_layout()
    _save_and_close(fig, save_path)


# ── 4. Sınıf dağılımı ────────────────────────────────────────────────────────

def plot_class_distribution(
    data_dir: str,
    class_names,
    save_path=None,
) -> None:
    """Her sınıftaki örnek sayısını gösteren yatay çubuk grafik oluşturur.

    Args:
        data_dir:    Kök veri dizini (Train/ alt klasörünü içermeli).
        class_names: Sınıf adı listesi.
        save_path:   Kaydedilecek dosya yolu.
    """
    if save_path is None:
        save_path = os.path.join(RESULT_DIR, "class_distribution.png")

    train_dir = os.path.join(data_dir, "Train")
    if not os.path.isdir(train_dir):
        print(f"[Visualize] Uyarı: Train/ bulunamadı ({train_dir}). Dağılım grafiği atlanıyor.")
        return

    counts = []
    labels_used = []

    for idx, name in enumerate(class_names):
        class_folder = os.path.join(train_dir, str(idx))
        if os.path.isdir(class_folder):
            n_files = sum(
                1 for f in os.listdir(class_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".ppm"))
            )
        else:
            n_files = 0
        counts.append(n_files)
        labels_used.append(f"{idx}: {name}")

    fig_height = max(8, len(class_names) * 0.38)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = range(len(class_names))
    bars  = ax.barh(y_pos, counts, color="steelblue", edgecolor="white", linewidth=0.5)

    # Değer etiketleri
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.005,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            ha="left",
            fontsize=7,
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels_used, fontsize=7)
    ax.set_xlabel("Örnek Sayısı", fontsize=11)
    ax.set_title("Sınıf Dağılımı (GTSRB — Train)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    _save_and_close(fig, save_path)
