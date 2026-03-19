"""
Trafik işareti tanıma — tek görüntü çıkarım (inference) scripti.

Kullanım:
    python src/predict.py --image yol/trafik_isareti.jpg
    python src/predict.py --image yol/trafik_isareti.jpg --model models/trafik_model.keras
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np

from src.config import (
    MODEL_PATH,
    IMG_SIZE,
    NUM_CLASSES,
    CLASS_NAMES,
    RESULT_DIR,
)


# ── Yardımcı: kutu yazdırma ───────────────────────────────────────────────────

def _print_box(title, rows, width=62):
    """Başlık ve anahtar-değer satırlarını kutu içinde yazdırır."""
    border = "═" * width
    print(f"\n╔{border}╗")
    title_pad = width - len(title)
    print(f"║{' ' * (title_pad // 2)}{title}{' ' * (title_pad - title_pad // 2)}║")
    print(f"╠{border}╣")
    for key, val in rows:
        line = f"  {key:<25} {val}"
        pad  = width - len(line)
        print(f"║{line}{' ' * max(pad, 1)}║")
    print(f"╚{border}╝\n")


# ── Çıkarım fonksiyonu ────────────────────────────────────────────────────────

def predict_image(
    model_path: str,
    image_path: str,
    class_names,
    img_size=IMG_SIZE,
    top_k: int = 5,
    show_plot: bool = True,
    save_plot: bool = True,
) -> list:
    """Tek bir görüntü üzerinde çıkarım yapar ve en yüksek K tahmini döner.

    Args:
        model_path:  Keras model dosyası (.keras / .h5).
        image_path:  Tahmin yapılacak görüntü dosyası.
        class_names: Sınıf adları (dict {int: str} veya liste).
        img_size:    Model giriş boyutu (yükseklik, genişlik).
        top_k:       Döndürülecek en iyi tahmin sayısı.
        show_plot:   True ise matplotlib penceresi açılır (GUI ortamı gerekir).
        save_plot:   True ise grafik results/ altına kaydedilir.

    Returns:
        ``[(sınıf_adı, güven_skoru), ...]`` — azalan güven sırasında.

    Raises:
        FileNotFoundError: Model veya görüntü dosyası bulunamazsa.
    """
    import tensorflow as tf
    from PIL import Image  # type: ignore
    import matplotlib.pyplot as plt
    import matplotlib
    if not show_plot:
        matplotlib.use("Agg")

    # ── Dosya kontrolleri ────────────────────────────────────────────────────
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"[Predict] Model bulunamadı: {model_path}\n"
            "Önce modeli eğitin: python src/train.py"
        )
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"[Predict] Görüntü bulunamadı: {image_path}")

    # ── Model yükleme ────────────────────────────────────────────────────────
    print(f"[Predict] Model yükleniyor: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # ── Görüntü ön işleme ────────────────────────────────────────────────────
    img_pil = Image.open(image_path).convert("RGB")
    img_resized = img_pil.resize(
        (img_size[1], img_size[0]),   # PIL: (genişlik, yükseklik)
        Image.BILINEAR,
    )
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch  = np.expand_dims(img_array, axis=0)

    # ── Çıkarım ──────────────────────────────────────────────────────────────
    predictions = model.predict(img_batch, verbose=0)[0]  # (num_classes,)

    # Top-K indeksleri
    top_indices = np.argsort(predictions)[::-1][:top_k]

    # Sınıf adını al (dict veya liste)
    def get_name(idx: int) -> str:
        if isinstance(class_names, dict):
            return class_names.get(idx, f"Sınıf {idx}")
        return class_names[idx] if idx < len(class_names) else f"Sınıf {idx}"

    results = [(get_name(i), float(predictions[i])) for i in top_indices]

    # ── Konsol çıktısı ───────────────────────────────────────────────────────
    rank_emojis = ["🥇", "🥈", "🥉", "4️⃣ ", "5️⃣ "]
    rows = []
    for rank, (name, conf) in enumerate(results):
        emoji = rank_emojis[rank] if rank < len(rank_emojis) else "  "
        rows.append((f"{emoji} {name}", f"%{conf * 100:.2f}"))

    _print_box(
        title="  TAHMİN SONUÇLARI  ",
        rows=[("Görüntü", os.path.basename(image_path))] + rows,
    )

    # ── Görselleştirme ────────────────────────────────────────────────────────
    _plot_prediction(
        img_array=img_array,
        results=results,
        image_path=image_path,
        show_plot=show_plot,
        save_plot=save_plot,
    )

    return results


def _plot_prediction(
    img_array: np.ndarray,
    results,
    image_path: str,
    show_plot: bool,
    save_plot: bool,
) -> None:
    """Görüntü ve Top-5 güven çubuklarını yan yana gösterir / kaydeder."""
    import matplotlib.pyplot as plt

    names  = [r[0] for r in results]
    confs  = [r[1] * 100 for r in results]
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(results))]

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Trafik İşareti Tahmin Sonucu", fontsize=14, fontweight="bold")

    # Görüntü paneli
    ax_img.imshow(np.clip(img_array, 0, 1))
    ax_img.set_title(
        f"En İyi Tahmin:\n{names[0]}\n({confs[0]:.1f}%)",
        fontsize=10,
        color="#27ae60",
        fontweight="bold",
    )
    ax_img.axis("off")

    # Güven çubuğu paneli
    y_pos = range(len(names))
    bars  = ax_bar.barh(list(y_pos), confs, color=colors, edgecolor="white")
    ax_bar.set_yticks(list(y_pos))
    ax_bar.set_yticklabels(
        [f"{n[:35]}..." if len(n) > 35 else n for n in names],
        fontsize=9,
    )
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Güven Skoru (%)")
    ax_bar.set_xlim([0, 105])
    ax_bar.set_title("Top-5 Tahmin", fontsize=11, fontweight="bold")
    ax_bar.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, confs):
        ax_bar.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_plot:
        os.makedirs(RESULT_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path  = os.path.join(RESULT_DIR, f"predict_{base_name}.png")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"[Predict] Grafik kaydedildi: {out_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ── Giriş noktası ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trafik işareti tanıma — tek görüntü tahmini"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Tahmin yapılacak görüntü dosyasının yolu",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help=f"Keras model dosyası (varsayılan: {MODEL_PATH})",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        dest="no_show",
        help="Matplotlib penceresini açma (sunucu/headless ortamı için)",
    )
    args = parser.parse_args()

    try:
        predict_image(
            model_path=args.model,
            image_path=args.image,
            class_names=CLASS_NAMES,
            img_size=IMG_SIZE,
            top_k=5,
            show_plot=not args.no_show,
            save_plot=True,
        )
    except FileNotFoundError as exc:
        print(f"\n[Hata] {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[Hata] Beklenmedik bir hata oluştu: {exc}")
        raise


if __name__ == "__main__":
    main()
