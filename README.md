# 🚦 Trafik İşareti Tanıma Sistemi

> **Derin Sinir Ağları dersi sunum projesi** — Sıfırdan eğitilmiş 4 bloklu CNN ile GTSRB veri setinde 43 farklı trafik işaretinin otomatik sınıflandırılması.

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red)
![Model](https://img.shields.io/badge/Model-CNN%20v3%20(4.98M%20params)-brightgreen)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.19%25-brightgreen)
![GPU](https://img.shields.io/badge/GPU-Apple%20M4%20Metal-black)

---

## 📑 İçindekiler

1. [Özet](#-özet)
2. [Motivasyon](#-motivasyon-ve-problem-tanımı)
3. [Veri Seti](#-veri-seti-gtsrb)
4. [Yöntem](#-yöntem)
5. [Deneyler ve Sonuçlar](#-deneyler-ve-sonuçlar)
6. [Web Arayüzü](#-web-arayüzü)
7. [Kurulum](#-kurulum)
8. [Proje Yapısı](#-proje-yapısı)
9. [Referanslar](#-referanslar)

---

## 🎯 Özet

Bu projede, German Traffic Sign Recognition Benchmark (GTSRB) veri seti üzerinde **43 sınıflı trafik işareti sınıflandırma problemi** ele alınmıştır. VGG-stili, 4 konvolüsyon bloklu, sıfırdan eğitilen özel bir CNN mimarisi (**4.98M parametre**) tasarlanmış; **CLAHE önişlemesi**, **oversampling ile sınıf dengeleme**, **label smoothing**, **cosine annealing öğrenme oranı** ve **AdamW** gibi modern derin öğrenme teknikleri ile desteklenmiştir. 12,630 örneklik GTSRB test seti üzerinde **%99.19 doğruluk** ve **%99.71 top-3 doğruluk** elde edilmiştir. **Grad-CAM** ile modelin yorumlanabilirliği analiz edilmiş, **TTA** (Test-Time Augmentation) ile çıkarım aşamasında ek güvence sağlanmış, **güven eşiği + entropi** tabanlı basit bir OOD (out-of-distribution) tespit mekanizması eklenmiştir.

### Özet Sonuçlar

| Metrik | Değer |
|---|---|
| **Test Doğruluğu (Accuracy)** | **%99.19** |
| Weighted Precision | %99.21 |
| Weighted Recall | %99.19 |
| Weighted F1 | %99.16 |
| Top-3 Doğruluk | %99.71 |
| En iyi Validation Doğruluğu | %99.92 |
| Test Örnek Sayısı | 12,630 |
| Eğitim süresi | 108.9 dk (30 epoch, Apple M4 Metal) |
| Eğitim boyutu (oversampled) | 53,160 örnek |

## 🌟 Motivasyon ve Problem Tanımı

Trafik işareti tanıma, **otonom sürüş** ve **sürücü destek sistemlerinin** temel bileşenlerinden biridir. Aynı levhanın farklı aydınlatma, açı, bulanıklık ve kısmi örtülme altında güvenilir biçimde tanınması gerekir. Veri setindeki sınıflar görsel olarak benzer şekillere (üçgen uyarı, daire yasak, dikdörtgen yön) sahip olduğundan; **sınıflar arası ince farkları ayıran güçlü öznitelik çıkarımına** ihtiyaç vardır.

**Problem:** Bir RGB görüntüyü 43 trafik işareti sınıfından birine atama — çok sınıflı görüntü sınıflandırması.

**Matematiksel ifade:** Giriş `x ∈ ℝ^(64×64×3)` için, model `f_θ(x): ℝ^(64×64×3) → Δ^42` fonksiyonu ile 43 sınıf üzerindeki olasılık dağılımını üretir. Hedef, çapraz entropi kaybını minimize etmektir:

$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{43} y_{i,c}^{LS} \log f_\theta(x_i)_c$$

Burada `y^LS` label smoothing ile yumuşatılmış hedef dağılımdır.

---

## 📊 Veri Seti (GTSRB)

**German Traffic Sign Recognition Benchmark** — Stallkamp et al. (2012).

| Özellik | Değer |
|---|---|
| Eğitim örneği | 26,640 |
| Test örneği | 12,630 |
| Sınıf sayısı | 43 |
| Görüntü boyutu | Değişken (15×15 – 250×250) |
| Model giriş boyutu | 64×64 (yeniden boyutlandırma) |
| Renk kanalı | RGB |

**Sınıflar:** Hız limitleri (9), yasaklayıcı (5), uyarı üçgeni (15), yön/zorunlu (8), yol ver / dur (2), öncelik (2), yasak sonu (3), vb.

**Zorluk:** Ciddi **sınıf dengesizliği** — en büyük sınıf 2,010 örnek, en küçük sınıf 210 örnek. Dengesizlik oranı ≈ **9.6×**. Bu, model eğitiminde azınlık sınıfların ihmal edilmesine yol açabilir.

> `results/class_distribution.png` — her sınıftaki örnek sayısını gösterir (eğitim başında otomatik üretilir).

---

## 🧪 Yöntem

Sistemin uçtan uca işleyişi aşağıdaki boru hattıyla özetlenebilir:

```
Ham görüntü → CLAHE → Normalizasyon → Augmentation → 4 Bloklu CNN v3 → Softmax → Sınıf
                                                                              ↓
                                                                          Grad-CAM
```

### 5.1 Önişleme — CLAHE

**Contrast Limited Adaptive Histogram Equalization**, LAB renk uzayında L (aydınlık) kanalına uygulanır:

```python
lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
lab[:,:,0] = clahe.apply(lab[:,:,0])   # sadece parlaklık kanalı
enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
```

- **Neden?** Düşük kontrastlı, gölgeli veya gece fotoğraflarında sembol detayları kaybolur. CLAHE, **yerel histogramları** (4×4 bölge) klipli bir şekilde eşitleyerek detayları öne çıkarır.
- **Avantaj:** Renk dengesini bozmaz (sadece L kanalına uygulanır), küresel histogram eşitlemenin tersine yerel adaptiftir.

### 5.2 Sınıf Dengeleme

Dengesizlik sorununa karşı iki mekanizma:

1. **Oversampling (hedef 1200 örnek/sınıf):**
   Az örnekli sınıflar, [rastgele augmentation] ile hedef sayıya kadar çoğaltılır. Sonuç: yaklaşık **51,600 dengelenmiş eğitim örneği**.

2. **Sınıf ağırlıkları (opsiyonel):**
   $$w_c = \frac{N_\text{total}}{K \cdot N_c}$$
   Her minibatch içindeki kaybın sınıfa göre ölçeklenmesi.

Varsayılan modda oversampling açık, class weights kapalıdır (oversampling sonrası veri zaten dengeli).

### 5.3 Augmentation (Online)

Her epoch'ta farklı rastgele dönüşümler:

| Teknik | Aralık | Gerekçe |
|---|---|---|
| Random Rotation | ±8% (≈ ±29°) | Gerçek kamera açıları |
| Random Zoom | ±10% | Farklı mesafelerden çekim |
| Random Brightness | ±18% | Aydınlatma koşulları |
| Random Contrast | ±18% | Hava koşulları |
| Random Translation | ±8% | Merkezde olmayan levhalar |

> **Horizontal flip YOK.** "Sağa dönünüz" ↔ "Sola dönünüz" anlamsal olarak farklı olduğundan flip işareti bozar.

### 5.4 Model Mimarisi — CNN v3

```
Giriş (64×64×3)
    │
[Conv3x3(64)×2  + BN + ReLU + MaxPool + SpatialDropout(0.10)]   64→32   Blok 1
[Conv3x3(128)×2 + BN + ReLU + MaxPool + SpatialDropout(0.15)]   32→16   Blok 2
[Conv3x3(256)×2 + BN + ReLU + MaxPool + SpatialDropout(0.20)]   16→8    Blok 3
[Conv3x3(512)×2 + BN + ReLU + MaxPool + SpatialDropout(0.25)]    8→4    Blok 4
    │
GlobalAveragePooling2D  (4×4×512 → 512)
    │
Dense(512) + BN + ReLU + Dropout(0.5)
    │
Dense(43, Softmax)
```

**Toplam parametre:** ≈ **4.98M** (tamamı eğitilebilir).

**Tasarım kararları:**
- **4 blok, kademeli filtre artışı** — hiyerarşik öznitelik çıkarımı: kenar → şekil → desen → sınıfa özgü parça.
- **He Normal başlatma** — ReLU aktivasyonları için Xavier'den daha iyi gradyan akışı sağlar.
- **SpatialDropout2D** — klasik dropout piksel bazlı rastgele düşürürken; SpatialDropout **tüm bir feature map kanalını düşürür**. Conv katmanlarda komşu pikseller arası korelasyon yüksek olduğu için bu yöntem daha etkilidir (Tompson et al., 2015).
- **Kademeli dropout (0.10 → 0.25)** — erken katmanlarda düşük (düşük seviye özellikler kritiktir), geç katmanlarda yüksek (aşırı uzmanlaşmayı önler).
- **Global Average Pooling** — fully connected yerine kullanılarak parametre sayısı düşürülür ve overfit azaltılır (Lin et al., 2014).
- **Batch Normalization** — iç kovaryans kaymasını azaltarak daha büyük öğrenme oranlarına izin verir.

### 5.5 Eğitim Stratejisi

**Optimizer:** AdamW (Loshchilov & Hutter, 2019)
- Adam'ın aksine weight decay'i gradyandan ayırır, L2 regularizasyonunu daha doğru uygular.
- `weight_decay = 1e-4`

**Kayıp fonksiyonu:** Label-smoothed categorical cross-entropy
$$y^{LS}_{i,c} = (1-\epsilon) \cdot y_{i,c} + \frac{\epsilon}{K}, \quad \epsilon=0.1$$
- Modelin aşırı güvenden (0.99'un üstü) kaçınmasını sağlar.
- Kalibre edilmiş (güven ≈ doğruluk) olasılıklar üretir.

**LR Schedule:** Cosine Annealing + Linear Warmup
```
Epoch 1-3: lr = 0 → 1e-3 (linear warmup)
Epoch 4+:  lr = lr_min + 0.5·(lr_max - lr_min)·(1 + cos(π · t/T))
```
Warmup, BatchNorm istatistiklerinin stabilize olmasına izin verir; cosine annealing ise yerel minimumlardan kaçmak için yumuşak bir azalma sağlar.

**Erken durdurma:** `val_accuracy` metriğinde 15 epoch patience.

**Checkpointing:** En iyi `val_accuracy` skoruna sahip model otomatik kaydedilir.

### 5.6 Test-Time Augmentation (TTA)

Çıkarım sırasında tek bir tahmin yerine, aynı görüntünün 5 varyantı (orijinal, zoom-in, zoom-out, parlak, karanlık) üretilip olasılıklar ortalanır:

$$p_\text{TTA}(c|x) = \frac{1}{N}\sum_{i=1}^{N} p(c|T_i(x))$$

Daha kararlı ve genellikle daha yüksek doğruluk sağlar, maliyeti 5× inference.

### 5.7 Yorumlanabilirlik — Grad-CAM

**Gradient-weighted Class Activation Mapping** (Selvaraju et al., 2017), son konvolüsyon katmanının aktivasyonlarını sınıf skorunun gradyanlarıyla ağırlıklandırır:

$$\alpha_k^c = \frac{1}{Z}\sum_{i,j} \frac{\partial y^c}{\partial A_{ij}^k} \qquad L^c_\text{Grad-CAM} = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

Sonuç: modelin **nereye baktığını** gösteren sıcaklık haritası. Projenin web arayüzünde **Grad-CAM Analizi** sekmesinde canlı olarak izlenebilir.

### 5.8 Out-of-Distribution (OOD) / Bilinmeyen Levha Tespiti

43-sınıflı bir sınıflandırıcı, eğitim setinde olmayan bir levhayla karşılaştığında **zorunlu olarak en benzer tanıdık sınıfı** atar — bu da yanıltıcı "yüksek güven" üretir. Bu riski azaltmak için arayüze basit bir çoklu-sinyal OOD tespiti eklendi:

1. **Güven eşiği:** Top-1 olasılık < kullanıcı ayarlı eşik (varsayılan %80) → `OUT-OF-DISTRIBUTION` uyarısı.
2. **Margin kontrolü:** Top-1 ile Top-2 arasındaki fark < %15 → `Belirsiz tahmin` uyarısı.
3. **Entropi:** $H(p) = -\sum p_c \log p_c$ değeri, kullanıcıya karar vermesi için metrik olarak gösterilir (0 = emin, 1 ≈ tamamen belirsiz).

> **Sınırlama:** Bu basit yöntem **confident misclassification** (model emin ama yanlış) durumunu yakalayamaz; gerçek OOD tespiti için ODIN, Mahalanobis distance veya Energy-based scoring gibi daha gelişmiş teknikler gerekir. Arayüz, kullanıcıyı ayrıca **Grad-CAM'e başvurmaya yönlendirir**.

---

## 📈 Deneyler ve Sonuçlar

### Ablation (Denenen yaklaşımlar)

| # | Model | Giriş | Param | Test Acc | Not |
|---|---|---|---|---|---|
| 1 | MobileNetV2 (transfer) | 96×96 | 2.3M | ≈%50 | ImageNet domain uyumsuzluğu |
| 2 | EfficientNetB0 (transfer) | 64×64 | 4.0M | ≈%6 | Başarısız |
| 3 | CNN v1 (3 blok) | 48×48 | 1.1M | %97.55 | İlk başarılı |
| 4 | CNN v2 (3 blok + SpatialDrop) | 48×48 | 1.5M | %97.97 | He_normal + Cosine LR |
| 5 | **CNN v3 (4 blok + AdamW + LS)** | **64×64** | **4.98M** | **%99.19** | **Bu proje — final** |

> Transfer learning denemelerinin başarısız olması çok öğreticidir: **domain farkı** (ImageNet doğal fotoğraflar vs. GTSRB küçük trafik işaretleri) çok büyük olduğunda, büyük önceden eğitilmiş modellerin düşük seviyeli özellikleri uyumsuz kalmakta. GTSRB için **sıfırdan eğitim daha iyi sonuç vermektedir**.

### Sınıf Bazlı Performans (Test Seti)

Tam rapor → [`results/classification_report.txt`](results/classification_report.txt)

**En iyi sınıflar (%100):** Dur, Yol Ver, 120 km/s, Çocuk Geçidi, Bisiklet Geçidi, Sağa/Sola Dönünüz, Tüm Yasakların Sonu ve 15'ten fazla sınıf.

**En zayıf sınıf:** Yaya Geçidi — %50 (30/60). Bu sınıf GTSRB benchmark'ında da tarihsel olarak en zor sınıftır:
- Çocuk Geçidi ve Trafik Işıkları işaretlerine **görsel olarak çok benzer** (aynı üçgen, benzer koyu figür).
- Test setinde yalnızca 60 örnek bulunduğu için tek bir yanlış sınıflandırma %1.67'lik doğruluk düşüşüne karşılık gelir.

**Orta zorluk sınıfları (%93-97):** Hız Limiti Sonu (80), Engebeli Yol, Girilmez, Öncelikli Yol, Hız Limiti 60.

### Üretilen Görseller

- **`results/training_history.png`** — Epoch başına accuracy & loss eğrileri
- **`results/confusion_matrix.png`** — 43×43 normalize karışıklık matrisi
- **`results/class_distribution.png`** — Eğitim seti sınıf dağılımı
- **`results/sample_predictions.png`** — Doğru/yanlış tahmin örnekleri
- **`results/classification_report.txt`** — Sınıf bazlı precision/recall/f1
- **`results/training_history.json`** — Tüm eğitim metrikleri (arayüzde kullanılır)

---

## 🌐 Web Arayüzü

Streamlit tabanlı interaktif arayüz **5 sekmeye** sahiptir:

| Sekme | İçerik |
|---|---|
| 📸 **Tahmin** | Fotoğraf yükle, top-5 + confidence gauge, **TTA toggle**, **güven eşiği slider'ı**, OOD/Belirsiz uyarı kartları, Top-1 / Margin / Entropi metrik kutuları |
| 🔍 **Grad-CAM Analizi** | Heatmap + overlay, hedef sınıf seçimi, son konvolüsyon katmanı bilgisi |
| 📊 **Eğitim & İstatistik** | Training curves (accuracy & loss), sınıf dağılımı, metric cards (en iyi val acc, epoch, süre) |
| 🧠 **Model Mimarisi** | Katman listesi, parametre sayısı, çıkış şekilleri (4.98M param, 64×64×3 giriş) |
| 🎓 **Yöntem & Sunum** | 10 maddelik yöntem kartları (CLAHE, AdamW, Label Smoothing, Grad-CAM vb.), referanslar |

**Çalıştırma:**
```bash
python3 -m streamlit run src/app.py
# Tarayıcıda http://localhost:8501
```

**Özellikler:**
- Sol panelde 43 levha kategorize edilmiş (tıklayınca örnek yüklenir)
- 18 hazır test örneği grid'i
- TTA toggle (CLAHE her zaman aktif)
- Dark theme + gradient tipografi

---

## 🛠️ Kurulum

```bash
# 1) Depoyu klonla
git clone https://github.com/halilibrahimd27/trafik-analiz.git
cd trafik-analiz

# 2) Bağımlılıkları yükle
pip install -r requirements.txt

# 3) Veri setini indir (Kaggle)
pip install kaggle
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
unzip gtsrb-german-traffic-sign.zip -d data/
```

Beklenen klasör yapısı:
```
data/
├── Train/
│   ├── 0/      ← 43 sınıf, toplam 26,640 görüntü
│   ├── 1/
│   └── ...
├── Test/
│   └── *.png   ← 12,630 test görüntüsü
└── Test.csv
```

### Kullanım

```bash
# Model eğitimi (~90-120 dk, M4 Metal GPU; 30 epoch, 64×64, batch 64)
python src/train.py --epochs 30

# Değerlendirme + confusion matrix
python src/evaluate.py

# CLI tahmini
python src/predict.py --image yol/trafik.jpg

# Web arayüzü
python3 -m streamlit run src/app.py
```

### Eğitim parametreleri

```bash
python src/train.py \
    --epochs 30 \
    --batch-size 64 \
    --lr 1e-3 \
    --target-per-class 1200     # oversampling hedefi
# Ek bayraklar:
#   --no-balanced    : oversampling kapat
#   --augment        : online augmentation (standart mod için)
#   --class-weights  : sınıf ağırlıkları (dengesiz mod için)
```

---

## 📁 Proje Yapısı

```
trafik-analiz/
├── src/
│   ├── config.py         # Ayarlar + 43 Türkçe sınıf adı
│   ├── model.py          # CNN v3 mimarisi (4 blok, AdamW, label smoothing)
│   ├── dataset.py        # GTSRB loader + CLAHE + oversampling + TTA
│   ├── gradcam.py        # Grad-CAM yorumlanabilirliği (YENİ)
│   ├── train.py          # Eğitim pipeline + cosine LR + history JSON
│   ├── evaluate.py       # Test metrikleri + confusion matrix + rapor
│   ├── predict.py        # CLI tek görüntü tahmini
│   ├── app.py            # Streamlit arayüzü (5 sekme)
│   ├── visualize.py      # Grafik üretimi
│   ├── prepare_data.py   # torchvision ile veri indirme
│   └── download_data.py  # Manuel indirme talimatları
├── data/                 # GTSRB (ayrıca indirilir)
├── models/               # Eğitilmiş .keras modelleri
│   └── trafik_model.keras  ← Aktif (CNN v3)
├── results/              # Grafikler + raporlar + history.json
├── requirements.txt      # Python bağımlılıkları
└── README.md             # Bu dosya
```

---

## 🎓 Ders Sunumu Hızlı Navigasyon

Sunum esnasında bu dosyadaki aşağıdaki bölümleri sırayla göster:

| Slayt | Dosya / Bölüm | İçerik |
|---|---|---|
| Problem & Motivasyon | README §2 | Trafik tanıma neden zor? |
| Veri seti | README §3 + `class_distribution.png` | GTSRB detayları + dengesizlik |
| Mimari | README §5.4 + app → **🧠 Model Mimarisi** sekmesi | Katman katman |
| Eğitim stratejisi | README §5.5 | AdamW + cosine + label smoothing teorisi |
| Deneyler | README §6 ablation tablosu | v1 → v2 → v3 gelişim |
| Sonuçlar | `training_history.png` + `confusion_matrix.png` | Eğriler + karışıklıklar |
| **Demo** | app → **📸 Tahmin** + **🔍 Grad-CAM** | Canlı tahmin + modelin baktığı yer |
| Yorum | app → **🎓 Yöntem & Sunum** sekmesi | 10 maddelik özet kartlar |

---

## 🔬 Teknoloji Yığını

- **Python 3.9** + **TensorFlow 2.16 / Keras**
- **tensorflow-macos** + **tensorflow-metal** (Apple M4 GPU hızlandırması)
- **OpenCV** — CLAHE kontrast iyileştirme
- **scikit-learn** — precision / recall / F1 / confusion matrix
- **Streamlit** — interaktif web arayüzü
- **Matplotlib / Seaborn** — grafik üretimi
- **Pandas / NumPy / Pillow** — veri işleme

---

## 📚 Referanslar

1. **Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012).** The German Traffic Sign Recognition Benchmark. *IJCNN*.
2. **Simonyan, K., & Zisserman, A. (2014).** Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG). *arXiv:1409.1556*.
3. **Ioffe, S., & Szegedy, C. (2015).** Batch Normalization. *ICML*.
4. **He, K., Zhang, X., Ren, S., & Sun, J. (2015).** Delving Deep into Rectifiers (He init). *ICCV*.
5. **Tompson, J., et al. (2015).** Efficient Object Localization Using Convolutional Networks (SpatialDropout). *CVPR*.
6. **Lin, M., Chen, Q., & Yan, S. (2014).** Network In Network (GAP). *ICLR*.
7. **Szegedy, C., et al. (2016).** Rethinking the Inception Architecture (Label Smoothing). *CVPR*.
8. **Loshchilov, I., & Hutter, F. (2017).** SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR*.
9. **Loshchilov, I., & Hutter, F. (2019).** Decoupled Weight Decay Regularization (AdamW). *ICLR*.
10. **Selvaraju, R. R., et al. (2017).** Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.

---

## 📄 Lisans

MIT — Akademik kullanım ve öğrenme amaçlı serbesttir. Veri seti GTSRB kendi lisansına tabidir.

---

<p align="center">
  <sub>🎓 <b>Derin Sinir Ağları</b> dönem sunum projesi · GTSRB · CNN v3 · Grad-CAM · TTA · Apple M4 Metal</sub>
</p>
