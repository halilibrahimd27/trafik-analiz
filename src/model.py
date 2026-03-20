"""
Trafik isareti tanima — GTSRB'ye ozel derin CNN v2.

Guncellemeler (v2):
  - he_normal kernel initializer (daha stabil egitim)
  - SpatialDropout2D (feature map bazli regularizasyon)
  - 3 blok + kademeli filtre artisi

Beklenen dogruluk: %97-99 (GTSRB benchmark)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    num_classes,
    img_size,
    learning_rate: float = 1e-3,
):
    """Gelistirilmis VGG-stili ozel CNN v2.

    Mimari (48x48 giris):
        [Conv3x3(64)x2 + BN + MaxPool + SpatialDropout(0.15)]
        [Conv3x3(128)x2 + BN + MaxPool + SpatialDropout(0.2)]
        [Conv3x3(256)x2 + BN + MaxPool + SpatialDropout(0.25)]
        → GlobalAveragePooling
        → Dense(512, relu) + BN + Dropout(0.5)
        → Dense(num_classes, softmax)

    ~1.5M parametre, tamami egitilebilir.
    """
    input_shape = (*img_size, 3)
    inp = keras.Input(shape=input_shape, name="input_image")

    # Blok 1: 48→24
    x = layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal", name="conv1a")(inp)
    x = layers.BatchNormalization(name="bn1a")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", kernel_initializer="he_normal", name="conv1b")(x)
    x = layers.BatchNormalization(name="bn1b")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2, name="pool1")(x)
    x = layers.SpatialDropout2D(0.15, name="sdrop1")(x)

    # Blok 2: 24→12
    x = layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal", name="conv2a")(x)
    x = layers.BatchNormalization(name="bn2a")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same", kernel_initializer="he_normal", name="conv2b")(x)
    x = layers.BatchNormalization(name="bn2b")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2, name="pool2")(x)
    x = layers.SpatialDropout2D(0.2, name="sdrop2")(x)

    # Blok 3: 12→6
    x = layers.Conv2D(256, 3, padding="same", kernel_initializer="he_normal", name="conv3a")(x)
    x = layers.BatchNormalization(name="bn3a")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same", kernel_initializer="he_normal", name="conv3b")(x)
    x = layers.BatchNormalization(name="bn3b")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2, name="pool3")(x)
    x = layers.SpatialDropout2D(0.25, name="sdrop3")(x)

    # Siniflandirici
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, kernel_initializer="he_normal", name="dense_512")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5, name="drop_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inp, outputs, name="trafik_cnn_v2")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"[Model] CNN v2 olusturuldu — {model.count_params():,} parametre")
    return model
