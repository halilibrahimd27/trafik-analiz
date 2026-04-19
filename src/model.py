"""
Trafik isareti tanima — GTSRB icin derin CNN v3.

v2 → v3 Guncellemeler:
  - 4. konvolusyon blogu eklendi (512 filtre) — daha zengin ozellik cikarimi
  - Giris boyutu 48×48 → 64×64 (daha fazla detay)
  - Label smoothing (0.1) — asiri guven + overfit onleme
  - Kademeli SpatialDropout (0.10 → 0.25) — erken katmanlarda daha az duzenlileme
  - L2 weight decay opsiyonu (AdamW)
  - Dense basligi genisletildi (512 birim, BN + Dropout)

Beklenen dogruluk: %98.5-99.3 (GTSRB benchmark)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _conv_block(x, filters, block_idx, spatial_drop):
    """Iki Conv2D + BN + ReLU + MaxPool + SpatialDropout2D blogu."""
    x = layers.Conv2D(
        filters, 3, padding="same",
        kernel_initializer="he_normal",
        name=f"conv{block_idx}a",
    )(x)
    x = layers.BatchNormalization(name=f"bn{block_idx}a")(x)
    x = layers.Activation("relu", name=f"relu{block_idx}a")(x)

    x = layers.Conv2D(
        filters, 3, padding="same",
        kernel_initializer="he_normal",
        name=f"conv{block_idx}b",
    )(x)
    x = layers.BatchNormalization(name=f"bn{block_idx}b")(x)
    x = layers.Activation("relu", name=f"relu{block_idx}b")(x)

    x = layers.MaxPooling2D(2, 2, name=f"pool{block_idx}")(x)
    x = layers.SpatialDropout2D(spatial_drop, name=f"sdrop{block_idx}")(x)
    return x


def build_model(
    num_classes: int,
    img_size,
    learning_rate: float = 1e-3,
    label_smoothing: float = 0.1,
    use_adamw: bool = True,
    weight_decay: float = 1e-4,
):
    """Gelistirilmis VGG-stili 4 blokli CNN v3.

    Mimari (64x64 giris):
        [Conv3x3(64)x2  + BN + MaxPool + SpatialDropout(0.10)]  64→32
        [Conv3x3(128)x2 + BN + MaxPool + SpatialDropout(0.15)]  32→16
        [Conv3x3(256)x2 + BN + MaxPool + SpatialDropout(0.20)]  16→8
        [Conv3x3(512)x2 + BN + MaxPool + SpatialDropout(0.25)]  8→4
        → GlobalAveragePooling
        → Dense(512, relu) + BN + Dropout(0.5)
        → Dense(num_classes, softmax)

    ~4.1M parametre, tamami egitilebilir.

    Args:
        num_classes:      Cikti sinif sayisi.
        img_size:         (yukseklik, genislik) — genellikle (64, 64).
        learning_rate:    Baslangic ogrenme orani.
        label_smoothing:  0.0-0.2 arasi; 0 ise klasik cross-entropy.
        use_adamw:        True ise AdamW (weight decay). False ise Adam.
        weight_decay:     L2 regularizasyon katsayisi (AdamW icin).
    """
    input_shape = (*img_size, 3)
    inp = keras.Input(shape=input_shape, name="input_image")

    x = _conv_block(inp, 64,  1, 0.10)
    x = _conv_block(x,   128, 2, 0.15)
    x = _conv_block(x,   256, 3, 0.20)
    x = _conv_block(x,   512, 4, 0.25)

    # Siniflandirici kafasi
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, kernel_initializer="he_normal", name="dense_512")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Activation("relu", name="relu_head")(x)
    x = layers.Dropout(0.5, name="drop_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inp, outputs, name="trafik_cnn_v3")

    # Optimizer
    if use_adamw and hasattr(keras.optimizers, "AdamW"):
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay,
        )
        opt_name = f"AdamW (wd={weight_decay})"
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        opt_name = "Adam"

    # Label smoothing ile categorical crossentropy
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )

    print(
        f"[Model] CNN v3 olusturuldu — {model.count_params():,} parametre | "
        f"Optimizer: {opt_name} | Label smoothing: {label_smoothing}"
    )
    return model
