"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementasyonu.

Referans:
  Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization", ICCV 2017.

Amac:
  Siniflandirma kararinin hangi gorsel bolgelerden etkilendigini
  sicaklik haritasi (heatmap) olarak gostermek. Sunumda ve kullanici
  arayuzunde yorumlanabilirlik (interpretability) icin kullanilir.

Kullanim:
    from src.gradcam import make_gradcam_heatmap, overlay_heatmap
    heatmap = make_gradcam_heatmap(model, image_batch, last_conv_name="conv4b")
    overlay = overlay_heatmap(image, heatmap)
"""

import numpy as np
import tensorflow as tf


def find_last_conv_layer(model) -> str:
    """Modelin son Conv2D katmaninin adini dondur."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Modelde Conv2D katmani bulunamadi.")


def make_gradcam_heatmap(
    model,
    image_batch: np.ndarray,
    last_conv_name: str = None,
    pred_index: int = None,
) -> np.ndarray:
    """Grad-CAM sicaklik haritasi uretir.

    Args:
        model:           Keras modeli.
        image_batch:     (1, H, W, 3) sekilli on islenmis goruntu.
        last_conv_name:  Hedef konvolusyon katmani. None ise otomatik bulunur.
        pred_index:      Hedef sinif. None ise argmax kullanilir.

    Returns:
        (H', W') sekilli float32 heatmap (0-1 araliginda).
    """
    if last_conv_name is None:
        last_conv_name = find_last_conv_layer(model)

    # Konvolusyon aktivasyonlari ve tahmin cikisini donen bir grad model kur
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_name).output, model.output],
    )

    image_batch = tf.cast(image_batch, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient: class score -> feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Global avg pool ile kanal bazli agirliklar
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Feature map'leri agirliklarla carp, kanal bazli topla
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalize [0,1]
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """Heatmap'i orijinal goruntuye overlay eder.

    Args:
        image:   (H, W, 3) float32 [0,1] veya uint8 [0,255] goruntu.
        heatmap: (H', W') float32 [0,1] heatmap.
        alpha:   Overlay saydamligi (0-1).
        colormap: matplotlib colormap adi.

    Returns:
        (H, W, 3) uint8 overlayli goruntu.
    """
    import matplotlib.cm as cm
    from PIL import Image as PILImage

    if image.dtype != np.uint8:
        image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        image_uint8 = image.copy()

    h, w = image_uint8.shape[:2]
    heatmap_img = PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(
        (w, h), PILImage.BILINEAR
    )
    heatmap_resized = np.array(heatmap_img) / 255.0

    cmap = cm.get_cmap(colormap)
    colored = (cmap(heatmap_resized)[..., :3] * 255).astype(np.uint8)

    overlay = (alpha * colored + (1 - alpha) * image_uint8).astype(np.uint8)
    return overlay
