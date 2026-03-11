# 🔐 NSL-KDD ile Saldırı Tespiti

Derin öğrenme yöntemleri kullanılarak NSL-KDD veri seti üzerinde ağ saldırısı tespiti.

Bu projede ağ trafiği üzerinde saldırı tespiti yapmak için üç farklı derin öğrenme yaklaşımı uygulanmıştır.

- CNN tabanlı sınıflandırma modeli
- Autoencoder + CNN hibrit modeli (AE-CNN)
- Variational Autoencoder (VAE) tabanlı anomali tespiti

Amaç, bu modellerin performanslarını karşılaştırarak ağ güvenliği için en uygun yaklaşımı analiz etmektir.

---

## 📁 Proje Yapısı

```
├── odev1_cnn_aecnn.py      # CNN ve AE-CNN modelleri
├── odev2_vae_anomaly.py    # VAE anomali tespiti
├── KDDTrain+.txt           # Eğitim veri seti
├── KDDTest+.txt            # Test veri seti
└── README.md               # Proje açıklaması
```

---

## 📦 Kurulum

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

---

## 📥 Veri Seti

NSL-KDD veri setini aşağıdaki linkten indirip proje klasörüne koy:

🔗 [https://www.unb.ca/cic/datasets/nsl.html](https://github.com/jmnwong/NSL-KDD-Dataset.git)

İndirilecek dosyalar: `KDDTrain+.txt` ve `KDDTest+.txt`

---

## 🚀 Çalıştırma

**Ödev 1 – CNN & AE-CNN:**
```bash
python odev1_cnn_aecnn.py
```

**Ödev 2 – VAE Anomali Tespiti:**
```bash
python odev2_vae_anomaly.py
```

---

## 📊 Sonuçlar

### Ödev 1 – CNN vs AE-CNN

| Metrik    | CNN    | AE-CNN |
|-----------|--------|--------|
| Accuracy  | 0.804  | 0.778  |
| Precision | 0.948  | 0.969  |
| Recall    | 0.694  | 0.631  |
| F1-Score  | 0.802  | 0.764  |

### Ödev 2 – VAE Anomali Tespiti

| Metrik    | Değer  |
|-----------|--------|
| ROC-AUC   | 0.9103 |
| Precision | 0.922  |
| Recall    | 0.679  |
| F1-Score  | 0.782  |
| FPR       | 0.076  |

---

## 🧠 Yöntemler

- **CNN**: 1D evrişimli sinir ağı ile doğrudan sınıflandırma
- **AE-CNN**: Autoencoder encoder kısmı + CNN sınıflandırıcı (iki aşamalı)
- **VAE**: Variational Autoencoder ile denetimsiz anomali tespiti

---

📚 Kaynaklar

- NSL-KDD Dataset  
  https://www.unb.ca/cic/datasets/nsl.html

- Kingma & Welling (2014)  
  Auto-Encoding Variational Bayes

- LeCun et al. (2015)  
  Deep Learning – Nature
