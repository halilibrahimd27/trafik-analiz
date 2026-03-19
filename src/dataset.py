"""
GTSRB veri seti yükleme ve ön işleme.
tensorflow-datasets üzerinden otomatik indirir.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np


def get_augmentation_layer() -> tf.keras.Sequential:
    """
    Trafik işaretleri için hafif augmentation.
    - Horizontal flip YOK (yön işaretleri değişir)
    - Hafif döndürme ve zoom (gerçek dünya koşulları)
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),        # ±36 derece
        tf.keras.layers.RandomZoom(0.1),            # %±10 zoom
        tf.keras.layers.RandomBrightness(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augmentation")


def _preprocess(image, label, img_size, num_classes):
    """Görüntüyü yeniden boyutlandır, normalize et ve etiketi one-hot'a çevir."""
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, num_classes)
    return image, label


def load_gtsrb_tfds(img_size=(64, 64), batch_size=64, num_classes=43, data_dir=None):
    """
    tensorflow-datasets üzerinden GTSRB'yi yükler.
    İlk çağrıda (~400 MB) otomatik indirir.

    Returns:
        train_ds, val_ds, test_ds  — tf.data.Dataset nesneleri
    """
    import tensorflow_datasets as tfds

    print("[Veri] GTSRB tensorflow-datasets üzerinden yükleniyor...")
    print("[Veri] İlk çalıştırmada ~400 MB indirilecek, lütfen bekleyin...")

    kwargs = dict(
        name="gtsrb",
        as_supervised=True,
        shuffle_files=True,
    )
    if data_dir:
        kwargs["data_dir"] = data_dir

    train_raw, test_raw = tfds.load(split=["train", "test"], **kwargs)

    total = tf.data.experimental.cardinality(train_raw).numpy()
    val_size = int(total * 0.2)

    train_raw = train_raw.shuffle(10000, seed=42)
    val_raw   = train_raw.take(val_size)
    train_raw = train_raw.skip(val_size)

    def preprocess(img, lbl):
        return _preprocess(img, lbl, img_size, num_classes)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_raw
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(5000)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_raw
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test_ds = (
        test_raw
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    print(f"[Veri] Eğitim: ~{total - val_size} | Doğrulama: ~{val_size} örnek")

    return train_ds, val_ds, test_ds


def load_test_dataset(data_dir, img_size=(64, 64), batch_size=64, num_classes=43):
    """
    data/Test/ klasöründen test verisini yükler (Test.csv etiketleri ile).
    Test.csv yoksa None döner.
    """
    import pandas as pd
    from PIL import Image as PILImage

    csv_path = os.path.join(data_dir, "Test.csv")
    test_dir = os.path.join(data_dir, "Test")

    if not os.path.isfile(csv_path) or not os.path.isdir(test_dir):
        print("[Veri] Test.csv veya Test/ bulunamadı, test seti atlanıyor.")
        return None

    df = pd.read_csv(csv_path)
    # Sütun adlarını esnek tut
    path_col  = "Path"   if "Path"    in df.columns else df.columns[0]
    label_col = "ClassId" if "ClassId" in df.columns else df.columns[-1]

    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, str(row[path_col]))
        if not os.path.isfile(img_path):
            continue
        img = PILImage.open(img_path).convert("RGB").resize((img_size[1], img_size[0]))
        images.append(np.array(img, dtype=np.float32) / 255.0)
        labels.append(int(row[label_col]))

    if not images:
        return None

    images = np.stack(images, axis=0)
    labels = np.array(labels, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((
        images,
        tf.one_hot(labels, num_classes)
    ))
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    print(f"[Veri] Test seti yüklendi: {len(images)} örnek")
    return dataset


def load_gtsrb_from_directory(data_dir, img_size=(64, 64), batch_size=64):
    """
    data/Train/ klasöründen veri yükler (Kaggle formatı).
    80/20 train/val bölünmesi yapar.
    """
    train_dir = os.path.join(data_dir, "Train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"{train_dir} bulunamadı. "
            "Kaggle'dan GTSRB indirip data/ altına çıkartın."
        )

    AUTOTUNE = tf.data.AUTOTUNE
    # Sınıf isimlerini sayısal sırada ver (0,1,2,...,42)
    # image_dataset_from_directory alfabetik sıralar (0,1,10,...), bu yanlış!
    class_names = [str(i) for i in range(43)]

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=class_names,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=class_names,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    normalization = tf.keras.layers.Rescaling(1.0 / 255)
    # NOT: train_ds'e .cache() EKLEME — augmentation her epoch'ta farklı
    # uygulanabilsin diye. Val seti değişmeyeceği için cache'lenebilir.
    train_ds = train_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds   = val_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds
