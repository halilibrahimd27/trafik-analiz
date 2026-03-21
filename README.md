# 🚦 Trafik İşareti Tanıma Sistemi

Derin öğrenme tabanlı trafik işareti tanıma projesi.
Sıfırdan eğitilmiş **VGG-stili CNN v2** modeli ile **43 farklı** Alman trafik işaretini **%97.97** test doğruluğuyla tanır.
GTSRB (German Traffic Sign Recognition Benchmark) veri seti ile eğitilmiştir.

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50-red)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-97.97%25-brightgreen)

---

## Sonuçlar

| Metrik | Değer |
|---|---|
| **Test Doğruluğu** | **%97.97** |
| Validation Doğruluğu | %100.00 |
| Test Kaybı | 0.0814 |
| Test Örnek Sayısı | 12,630 |
| Eğitim Süresi | ~40 epoch (~8 dk, M4 GPU) |

### Denenen Yaklaşımlar

| Model | Test Acc | Not |
|---|---|---|
| MobileNetV2 (transfer learning) | ~%50 | ImageNet domain uyumsuzluğu |
| EfficientNetB0 (transfer learning) | ~%6 | Başarısız |
| **CNN v1** (3 blok, sıfırdan) | %97.55 | İlk başarılı model |
| **CNN v2** (3 blok, iyileştirilmiş) | **%97.97** | he_normal + SpatialDropout + Cosine LR |
| CNN v2 + Augmentation | %97.91 | Daha düşük loss ama test acc biraz düşük |

---

## Proje Yapısı

```
trafik-analiz/
├── src/
│   ├── config.py         # Ayarlar ve Türkçe sınıf isimleri (43 sınıf)
│   ├── model.py          # CNN v2 model mimarisi
│   ├── dataset.py        # Veri yükleme, augmentation, CLAHE, oversampling
│   ├── train.py          # Eğitim pipeline (cosine LR, class weights)
│   ├── evaluate.py       # Model değerlendirme ve raporlama
│   ├── predict.py        # Tek görüntü tahmini (CLI)
│   ├── app.py            # Streamlit web arayüzü
│   ├── visualize.py      # Grafik ve görselleştirme fonksiyonları
│   ├── prepare_data.py   # Veri seti indirme (torchvision)
│   └── download_data.py  # Manuel indirme talimatları
├── data/                 # GTSRB veri seti (ayrıca indirilmeli)
├── models/               # Eğitilmiş model dosyaları
├── results/              # Grafikler ve raporlar
├── requirements.txt
└── README.md
```

---

## Kurulum

```bash
# Depoyu klonla
git clone https://github.com/halilibrahimd27/trafik-analiz.git
cd trafik-analiz

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### Veri Seti İndirme

**GTSRB** veri setini Kaggle'dan indir ve `data/` klasörüne çıkart:

```bash
pip install kaggle
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
unzip gtsrb-german-traffic-sign.zip -d data/
```

Ya da [Kaggle sayfasından](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) manuel indir.

Beklenen klasör yapısı:
```
data/
├── Train/
│   ├── 0/   ← 43 sınıf (0–42), toplam 26,640 görüntü
│   ├── 1/
│   └── ...
├── Test/
│   └── *.png  ← 12,630 test görüntüsü
└── Test.csv
```

---

## Kullanım

### 🌐 Web Arayüzü (önerilen)

```bash
python3 -m streamlit run src/app.py
```

Tarayıcıda `http://localhost:8501` adresini aç.

Özellikler:
- Fotoğraf sürükle-bırak veya dosya seç
- Sol panelde 43 levhaya tıklayarak örnek yükleme
- 18 hazır test örneği
- Top-5 tahmin bar grafiği
- Güven göstergesi (gauge)
- Model bilgisi sekmesi

### 🏋️ Model Eğitimi

```bash
# Standart eğitim (40 epoch, cosine LR)
python src/train.py

# Augmentation ile eğitim
python src/train.py --augment --epochs 50

# Dengelenmiş veri seti ile (oversampling)
python src/train.py --balanced --target-per-class 1200

# Tüm seçenekler
python src/train.py --epochs 40 --batch-size 64 --lr 1e-3 --augment --class-weights
```

### 🔍 Tek Görüntü Tahmini (CLI)

```bash
python src/predict.py --image /yol/trafik_isareti.jpg
```

### 📊 Model Değerlendirme

```bash
python src/evaluate.py
```

Üretilen dosyalar:
- `results/confusion_matrix.png` — Karışıklık matrisi
- `results/classification_report.txt` — Sınıf bazlı rapor
- `results/training_history.png` — Eğitim grafikleri
- `results/sample_predictions.png` — Örnek tahminler

---

## Model Mimarisi

**VGG-stili CNN v2** — sıfırdan eğitildi, ~1.5M parametre:

```
Giriş (48×48×3)
    │
[Conv2D(64)×2 + BN + ReLU + MaxPool + SpatialDropout(0.15)]   ← Blok 1
    │
[Conv2D(128)×2 + BN + ReLU + MaxPool + SpatialDropout(0.20)]  ← Blok 2
    │
[Conv2D(256)×2 + BN + ReLU + MaxPool + SpatialDropout(0.25)]  ← Blok 3
    │
GlobalAveragePooling2D
    │
Dense(512) + BN + ReLU + Dropout(0.5)
    │
Dense(43, Softmax)  ← 43 trafik işareti sınıfı
```

### v1 → v2 İyileştirmeler

| Özellik | v1 | v2 |
|---|---|---|
| Kernel init | varsayılan | `he_normal` |
| Dropout tipi | `Dropout` | `SpatialDropout2D` |
| LR schedule | `ReduceLROnPlateau` | Cosine annealing + warmup |
| Test accuracy | %97.55 | **%97.97** |

### Neden Transfer Learning Değil?

Transfer learning (MobileNetV2, EfficientNetB0) denendi ve başarısız oldu:
- Trafik işaretleri ImageNet'ten tamamen farklı bir domain
- Küçük geometrik şekiller (daire, üçgen, ok) için özel CNN daha uygun
- 26K eğitim örneği sıfırdan CNN eğitimi için yeterli

---

## Veri Genişletme (Augmentation)

Eğitim sırasında kullanılabilecek teknikler:

| Teknik | Açıklama |
|---|---|
| RandomRotation | ±22° döndürme (gerçek dünya kamera açısı) |
| RandomZoom | ±8% yakınlaştırma/uzaklaştırma |
| RandomBrightness | ±15% parlaklık değişimi |
| RandomContrast | ±15% kontrast değişimi |
| RandomTranslation | ±6% öteleme |
| CLAHE | Kontrast iyileştirme (düşük ışık koşulları) |
| Oversampling | Az örnekli sınıfları hedef sayıya çıkarma |
| Class Weights | Dengesiz sınıflara daha yüksek ağırlık |

> Not: Horizontal flip kullanılmaz — yön işaretlerinin anlamını değiştirir.

---

## Desteklenen Trafik İşaretleri (43 Sınıf)

| Kategori | İşaretler |
|---|---|
| 🚗 **Hız Limitleri** | 20, 30, 50, 60, 70, 80, 80-sonu, 100, 120 km/s |
| 🚫 **Yasaklayıcı** | Geçiş Yasağı, 3.5T Araç Yasağı, Araç Giremez, 3.5T Giremez, Girilmez |
| 🛑 **Dur / Yol Ver** | Dur, Yol Ver |
| 💛 **Öncelik** | Öncelikli Yol Kavşağı, Öncelikli Yol |
| ⚠️ **Tehlike Uyarıları** | Genel Tehlike, Virajlar, Kaygan Yol, Daralan Yol, Yol Çalışması, Trafik Işıkları, Yaya/Çocuk/Bisiklet Geçidi, Buzlanma, Hayvan Geçidi, Engebeli Yol |
| ↗️ **Yön & Zorunlu** | Sağa/Sola Dönünüz, İleri Gidiniz, İleri/Sağa, İleri/Sola, Sağdan/Soldan Geçiniz, Dönel Kavşak |
| ✅ **Yasakların Sonu** | Tüm Yasakların Sonu, Geçiş Yasağı Sonu, 3.5T Yasak Sonu |

---

## Teknolojiler

- **Python 3.9**
- **TensorFlow 2.16 / Keras** (tensorflow-macos + tensorflow-metal)
- **Streamlit** — interaktif web arayüzü
- **OpenCV** — CLAHE kontrast iyileştirme
- **scikit-learn** — metrik hesaplama
- **Pandas / NumPy / Pillow** — veri işleme
- **Matplotlib / Seaborn** — görselleştirme
- **Apple M4 Metal GPU** — GPU hızlandırmalı eğitim
