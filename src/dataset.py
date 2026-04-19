"""
GTSRB veri seti yukleme, augmentation ve on isleme.

Ozellikler:
  - Online augmentation (her epoch farkli donusumler)
  - Sinif dengeleme (oversampling + class weights)
  - CLAHE on isleme (dusuk kontrast iyilestirme)
  - Test-Time Augmentation (TTA) destegi
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np


# -- Augmentation katmani -----------------------------------------------------

def get_augmentation_layer():
    """
    Trafik isaretleri icin gercekci augmentation.
    - Horizontal flip YOK (yon isaretleri anlami degisir)
    - Hafif donme, zoom, parlaklik, kontrast, oteleme
    - Sonunda [0,1] araliginda kirpilir
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom((-0.10, 0.10)),
        tf.keras.layers.RandomBrightness(0.18, value_range=(0.0, 1.0)),
        tf.keras.layers.RandomContrast(0.18),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0)),
    ], name="augmentation")


def get_tta_augmentations():
    """
    Test-Time Augmentation icin hafif donusum listesi.
    Her bir fonksiyon (H, W, 3) tensor -> (H, W, 3) tensor alir.

    TTA: ayni goruntuyu bir kac varyantla tahmin edip olasiliklari ortalayarak
    daha kararli bir sonuc elde etme teknigi.
    """
    def identity(img):      return img
    def rot_p5(img):        return tf.image.rot90(img, k=0)  # placeholder
    def zoom_out(img):
        # Hafif pad + merkeze kirp (zoom-out etkisi)
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        pad = 4
        padded = tf.image.resize_with_crop_or_pad(img, h + 2 * pad, w + 2 * pad)
        return tf.image.resize(padded, (h, w))
    def zoom_in(img):
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        crop = 4
        cropped = tf.image.resize_with_crop_or_pad(img, h - 2 * crop, w - 2 * crop)
        return tf.image.resize(cropped, (h, w))
    def bright_up(img):     return tf.clip_by_value(img * 1.10, 0.0, 1.0)
    def bright_dn(img):     return tf.clip_by_value(img * 0.90, 0.0, 1.0)

    return [identity, zoom_in, zoom_out, bright_up, bright_dn]


def tta_predict(model, image_np, n_augments: int = 5):
    """Tek bir goruntu uzerinde TTA uygular, olasiliklari ortalar.

    Args:
        model:       Keras modeli.
        image_np:    (H, W, 3) float32 [0,1] numpy dizisi.
        n_augments:  Kullanilacak TTA varyant sayisi (en fazla 5).

    Returns:
        (num_classes,) ortalama olasilik vektoru.
    """
    tfs = get_tta_augmentations()[:n_augments]
    variants = []
    img = tf.convert_to_tensor(image_np, dtype=tf.float32)
    for fn in tfs:
        aug = fn(img).numpy()
        variants.append(aug)
    batch = np.stack(variants, axis=0)
    probs = model.predict(batch, verbose=0)
    return probs.mean(axis=0)


# -- CLAHE on isleme ----------------------------------------------------------

def apply_clahe_batch(images_np):
    """
    Goruntu dizisine CLAHE (Contrast Limited Adaptive Histogram Equalization) uygular.
    Dusuk kontrastli / karanlik fotograflarda detaylari one cikarir.
    """
    try:
        import cv2
    except ImportError:
        return images_np

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    result = []
    for img in images_np:
        img_uint8 = (img * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        result.append(enhanced.astype(np.float32) / 255.0)
    return np.stack(result)


# -- Sinif agirliklari hesaplama ---------------------------------------------

def compute_class_weights(data_dir, num_classes=43):
    """Her siniftaki ornek sayisina gore ters orantili agirlik hesaplar."""
    train_dir = os.path.join(data_dir, "Train")
    counts = np.zeros(num_classes, dtype=np.float64)
    for cls_idx in range(num_classes):
        cls_dir = os.path.join(train_dir, str(cls_idx))
        if os.path.isdir(cls_dir):
            counts[cls_idx] = len(os.listdir(cls_dir))

    total = counts.sum()
    weights = {}
    for i in range(num_classes):
        if counts[i] > 0:
            weights[i] = total / (num_classes * counts[i])
        else:
            weights[i] = 1.0
    return weights


def get_class_counts(data_dir, num_classes=43):
    """data/Train/<id>/ altindaki dosya sayilarini dondur."""
    train_dir = os.path.join(data_dir, "Train")
    counts = np.zeros(num_classes, dtype=np.int64)
    for cls_idx in range(num_classes):
        cls_dir = os.path.join(train_dir, str(cls_idx))
        if os.path.isdir(cls_dir):
            counts[cls_idx] = len([
                f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm'))
            ])
    return counts


# -- Oversampling -------------------------------------------------------------

def _oversample_directory(data_dir, img_size, target_per_class=1200):
    """
    Az ornekli siniflari offline augmente ederek hedef sayiya cikarir.
    Tum siniflari numpy array olarak dondurur.
    """
    from PIL import Image as PILImage

    train_dir = os.path.join(data_dir, "Train")
    all_images = []
    all_labels = []

    aug_layer = get_augmentation_layer()

    for cls_idx in range(43):
        cls_dir = os.path.join(train_dir, str(cls_idx))
        if not os.path.isdir(cls_dir):
            continue

        cls_images = []
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm'))]
        for fname in files:
            img = PILImage.open(os.path.join(cls_dir, fname)).convert("RGB").resize((img_size[1], img_size[0]))
            cls_images.append(np.array(img, dtype=np.float32) / 255.0)

        if not cls_images:
            continue

        cls_arr = np.stack(cls_images)
        n_original = len(cls_arr)

        if n_original < target_per_class:
            n_needed = target_per_class - n_original
            aug_images = []
            for _ in range(n_needed):
                idx = np.random.randint(0, n_original)
                src = cls_arr[idx:idx+1]
                augmented = aug_layer(src, training=True).numpy()
                augmented = np.clip(augmented, 0.0, 1.0)
                aug_images.append(augmented[0])
            if aug_images:
                cls_arr = np.concatenate([cls_arr, np.stack(aug_images)], axis=0)

        all_images.append(cls_arr)
        all_labels.extend([cls_idx] * len(cls_arr))

        if cls_idx % 10 == 0:
            print(f"  [Oversampling] Sinif {cls_idx}: {n_original} -> {len(cls_arr)} ornek")

    images = np.concatenate(all_images, axis=0)
    labels = np.array(all_labels, dtype=np.int32)
    print(f"[Veri] Oversampling sonrasi toplam: {len(labels):,} ornek")
    return images, labels


# -- Ana veri yukleme fonksiyonlari ------------------------------------------

def load_gtsrb_from_directory(data_dir, img_size=(64, 64), batch_size=64, augment=False):
    """
    data/Train/ klasorunden veri yukler.
    80/20 train/val bolunmesi yapar.
    """
    train_dir = os.path.join(data_dir, "Train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"{train_dir} bulunamadi. "
            "Kaggle'dan GTSRB indirip data/ altina cikartin."
        )

    AUTOTUNE = tf.data.AUTOTUNE
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

    if augment:
        aug_layer = get_augmentation_layer()
        def train_preprocess(x, y):
            x = normalization(x)
            x = aug_layer(x, training=True)
            return x, y
        train_ds = train_ds.map(train_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    else:
        train_ds = train_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    val_ds = val_ds.map(lambda x, y: (normalization(x), y), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds


def load_balanced_dataset(data_dir, img_size=(64, 64), batch_size=64, target_per_class=1200,
                           apply_clahe=True, online_augment=True):
    """
    Oversampled + balanced veri seti yukler.
    Az ornekli siniflari augmentation ile cogaltir, opsiyonel CLAHE uygular.
    80/20 train/val split yapar.
    """
    print("[Veri] Dengelenmis veri seti hazirlaniyor...")
    images, labels = _oversample_directory(data_dir, img_size, target_per_class)

    if apply_clahe:
        print("[Veri] CLAHE kontrast iyilestirme uygulaniyor...")
        images = apply_clahe_batch(images)

    # Shuffle & split
    n = len(labels)
    indices = np.random.RandomState(42).permutation(n)
    images = images[indices]
    labels = labels[indices]

    split = int(n * 0.8)
    x_train, x_val = images[:split], images[split:]
    y_train, y_val = labels[:split], labels[split:]

    print(f"[Veri] Egitim: {len(y_train):,} | Dogrulama: {len(y_val):,}")

    AUTOTUNE = tf.data.AUTOTUNE
    num_classes = 43

    train_ds = tf.data.Dataset.from_tensor_slices((
        x_train, tf.one_hot(y_train, num_classes)
    )).shuffle(10000)

    if online_augment:
        aug_layer = get_augmentation_layer()
        def aug_fn(x, y):
            # Tek ornek -> ekle batch boyutu -> augmente -> kaldir
            x = aug_layer(tf.expand_dims(x, 0), training=True)[0]
            return x, y
        train_ds = train_ds.map(aug_fn, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((
        x_val, tf.one_hot(y_val, num_classes)
    )).batch(batch_size).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds


def load_test_dataset(data_dir, img_size=(64, 64), batch_size=64, num_classes=43,
                       apply_clahe=True):
    """
    data/Test/ klasorunden test verisini yukler (Test.csv etiketleri ile).
    """
    import pandas as pd
    from PIL import Image as PILImage

    csv_path = os.path.join(data_dir, "Test.csv")
    test_dir = os.path.join(data_dir, "Test")

    if not os.path.isfile(csv_path) or not os.path.isdir(test_dir):
        print("[Veri] Test.csv veya Test/ bulunamadi, test seti atlaniyor.")
        return None

    df = pd.read_csv(csv_path)
    path_col  = "Path"    if "Path"    in df.columns else df.columns[0]
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

    if apply_clahe:
        images = apply_clahe_batch(images)

    dataset = tf.data.Dataset.from_tensor_slices((
        images, tf.one_hot(labels, num_classes)
    ))
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    print(f"[Veri] Test seti yuklendi: {len(images)} ornek{' (CLAHE)' if apply_clahe else ''}")
    return dataset


def load_test_arrays(data_dir, img_size=(64, 64), apply_clahe=True):
    """Test verisini numpy array olarak dondur (evaluate / TTA icin)."""
    import pandas as pd
    from PIL import Image as PILImage

    csv_path = os.path.join(data_dir, "Test.csv")
    test_dir = os.path.join(data_dir, "Test")
    if not os.path.isfile(csv_path) or not os.path.isdir(test_dir):
        return None

    df = pd.read_csv(csv_path)
    path_col  = "Path"    if "Path"    in df.columns else df.columns[0]
    label_col = "ClassId" if "ClassId" in df.columns else df.columns[-1]

    images, labels = [], []
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

    if apply_clahe:
        images = apply_clahe_batch(images)

    return images, labels
