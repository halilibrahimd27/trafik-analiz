"""
Trafik işareti tanıma — GTSRB'ye özel derin CNN.

Sıfırdan eğitilen bu model transfer learning'e göre çok daha
iyi sonuç verir çünkü:
  1) Trafik işaretleri ImageNet'ten tamamen farklı bir domain'dir.
  2) 26K eğitim örneği özel bir CNN için yeterlidir.
  3) Küçük geometrik özellikler (şekil, renk, numara) için
     özel mimari daha avantajlıdır.

Beklenen doğruluk: %95–99 (GTSRB benchmark)
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
) -> keras.Model:
    """VGG-stili özel CNN — GTSRB için optimize edilmiştir.

    Mimari (48×48 giriş için):
        [Conv3×3(64)×2 + BN + MaxPool + Dropout(0.25)] × 3
        → GlobalAveragePooling
        → Dense(512, relu) + BN + Dropout(0.5)
        → Dense(num_classes, softmax)

    Toplam ~1.2M parametre, eğitilebilir.
    """
    input_shape = (*img_size, 3)

    inp = keras.Input(shape=input_shape, name="input_image")

    # Blok 1  ─────────────────────────────────────────────────────────────────
    x = layers.Conv2D(64, 3, padding="same", name="conv1a")(inp)
    x = layers.BatchNormalization(name="bn1a")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same", name="conv1b")(x)
    x = layers.BatchNormalization(name="bn1b")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2, name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # Blok 2  ─────────────────────────────────────────────────────────────────
    x = layers.Conv2D(128, 3, padding="same", name="conv2a")(x)
    x = layers.BatchNormalization(name="bn2a")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same", name="conv2b")(x)
    x = layers.BatchNormalization(name="bn2b")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2, name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Blok 3  ─────────────────────────────────────────────────────────────────
    x = layers.Conv2D(256, 3, padding="same", name="conv3a")(x)
    x = layers.BatchNormalization(name="bn3a")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same", name="conv3b")(x)
    x = layers.BatchNormalization(name="bn3b")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2, name="pool3")(x)
    x = layers.Dropout(0.25, name="drop3")(x)

    # Sınıflandırıcı  ─────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, name="dense_512")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5, name="drop_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inp, outputs, name="trafik_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"[Model] Özel CNN oluşturuldu — {model.count_params():,} parametre (tamamı eğitilebilir)")
    return model


def unfreeze_top_layers(model, num_layers: int = 20) -> None:
    """Özel CNN için kullanılmaz — tüm katmanlar zaten açık."""
    print("[Model] Özel CNN: tüm katmanlar zaten eğitilebilir, unfreeze gerekmez.")
