# 🚦 Trafik İşareti Tanıma Sistemi

Sıfırdan eğitilmiş derin CNN (VGG-stili) kullanan trafik işareti tanıma projesi.
**43 farklı** Alman trafik işaretini **%97.55** test doğruluğuyla tanır.
GTSRB (German Traffic Sign Recognition Benchmark) veri seti ile eğitilmiştir.

---

## Sonuçlar

| Metrik | Değer |
|---|---|
| Test Doğruluğu | **%97.55** |
| Validation Doğruluğu | **%99.98** |
| Precision (weighted) | %97.69 |
| Recall (weighted) | %97.55 |
| F1 Skoru | %97.51 |
| Test Örnek Sayısı | 12,630 |

---

## Proje Yapısı

```
trafik-analiz/
├── src/
│   ├── config.py         # Ayarlar ve Türkçe sınıf isimleri
│   ├── model.py          # VGG-stili özel CNN mimarisi
│   ├── dataset.py        # Veri yükleme & ön işleme
│   ├── train.py          # Eğitim pipeline
│   ├── evaluate.py       # Model değerlendirme
│   ├── predict.py        # Tek görüntü tahmini (CLI)
│   └── app.py            # Streamlit web arayüzü
├── data/                 # GTSRB veri seti (ayrıca indirilmeli)
│   ├── Train/
│   ├── Test/
│   └── Test.csv
├── models/               # Eğitilmiş model dosyaları
├── results/              # Grafikler ve raporlar
├── requirements.txt
└── README.md
```

---

## Kurulum

```bash
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

Alternatif: [Kaggle sayfasından](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) manuel indir.

Beklenen klasör yapısı:
```
data/
├── Train/
│   ├── 0/   ← 43 sınıf (0–42)
│   ├── 1/
│   └── ...
├── Test/
│   └── *.png
└── Test.csv
```

---

## Kullanım

### Web Arayüzü (önerilen)

```bash
python3 -m streamlit run src/app.py
```

Tarayıcıda `http://localhost:8501` adresini aç.
Fotoğraf yükle veya sol panelden bir levhaya tıkla — model anında tanısın.

### Model Eğitimi

```bash
python src/train.py
python src/train.py --epochs 50 --batch-size 64
```

### Tek Görüntü Tahmini (CLI)

```bash
python src/predict.py --image /yol/trafik_isareti.jpg
```

Örnek çıktı:
```
╔══════════════════════════════════════════════════════════════╗
║                       TAHMİN SONUÇLARI                       ║
╠══════════════════════════════════════════════════════════════╣
║  Görüntü                   trafik.jpg                        ║
║  🥇 Hız Limiti (30 km/s)   %97.10                            ║
║  🥈 Hız Limiti (20 km/s)   %2.70                             ║
║  🥉 Hız Limiti (50 km/s)   %0.10                             ║
╚══════════════════════════════════════════════════════════════╝
```

### Model Değerlendirme

```bash
python src/evaluate.py
```

---

## Model Mimarisi

**VGG-stili Özel CNN** — sıfırdan eğitildi, ~1.3M parametre:

```
Giriş (48×48×3)
    │
[Conv2D(64)×2 + BatchNorm + ReLU + MaxPool + Dropout(0.25)]   ← Blok 1
    │
[Conv2D(128)×2 + BatchNorm + ReLU + MaxPool + Dropout(0.25)]  ← Blok 2
    │
[Conv2D(256)×2 + BatchNorm + ReLU + MaxPool + Dropout(0.25)]  ← Blok 3
    │
GlobalAveragePooling2D
    │
Dense(512, ReLU) + BatchNorm + Dropout(0.5)
    │
Dense(43, Softmax)  ← 43 trafik işareti sınıfı
```

Transfer learning (MobileNetV2, EfficientNetB0) yerine sıfırdan CNN tercih edildi:
trafik işaretleri ImageNet'ten farklı bir domain, özel CNN bu görev için çok daha iyi sonuç veriyor.

---

## Desteklenen Trafik İşaretleri (43 Sınıf)

| Kategori | İşaretler |
|---|---|
| 🚗 Hız Limitleri | 20, 30, 50, 60, 70, 80, 80-sonu, 100, 120 km/s |
| 🚫 Yasaklayıcı | Geçiş Yasağı, Araç Yasağı, Araç Giremez, Girilmez |
| 🛑 Dur / Yol Ver | Dur, Yol Ver |
| 💛 Öncelik | Öncelikli Yol Kavşağı, Öncelikli Yol |
| ⚠️ Tehlike Uyarıları | Genel Tehlike, Viraj, Kaygan Yol, Yol Çalışması, Yaya/Çocuk/Bisiklet Geçidi, Buzlanma, Hayvan Geçidi ... |
| ↗️ Yön & Zorunlu | Sağa/Sola Dönünüz, İleri Gidiniz, Sağdan/Soldan Geçiniz, Dönel Kavşak ... |
| ✅ Yasakların Sonu | Tüm Yasakların Sonu, Geçiş Yasağı Sonu, Ton Yasağı Sonu |

---

## Teknolojiler

- **Python 3.9**
- **TensorFlow / Keras** (tensorflow-macos + tensorflow-metal)
- **Streamlit** — web arayüzü
- **scikit-learn** — metrik hesaplama
- **Pandas / NumPy / Pillow** — veri işleme
- **Matplotlib / Seaborn** — görselleştirme
- **Apple M4 Metal GPU** — eğitim hızlandırma
