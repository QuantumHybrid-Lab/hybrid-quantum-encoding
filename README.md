# Obezite Düzeyi Tahmininde Klasik ve Kuantum Tabanlı Yaklaşımlar: Hibrit Kodlama Analizi

YGA 2026 · Grup 14 tarafından yürütülen bu araştırma projesinde, klasik sağlık verilerinin (UCI Obesity Dataset) kuantum devrelere aktarımında yaşanan verimlilik kayıplarını en aza indirmek için **adaptif hibrit kodlama** stratejileri araştırılmaktadır. 

Kategorik ve sürekli (mixed-type) özelliklerin bir arada bulunduğu veri setlerinde tek tip kuantum kodlamasının (encoding) sebep olduğu doğruluk ve donanım kısıtlamalarını aşmak amacıyla VQC (Variational Quantum Circuit) tabanlı yeni bir mimari geliştirilmiş ve literatüre bir katkı olarak sunulmuştur.

## Özgün Katkı
Bu proje, karma yapıya sahip veriler için kuantum yapay zeka alanında şu yeniliği sunar:
- **Kategorik Değişkenler:** Angle Encoding
- **Sürekli Değişkenler:** Amplitude Encoding

Bu iki yöntemin eşzamanlı olarak aynı VQC (Değişken Kuantum Devresi) içerisinde kullanılabilmesine olanak tanıyan **Adaptif Hibrit Kodlama** stratejisi geliştirilmiştir. Geleneksel tek tip kuantum kodlama veya klasik SVM yaklaşımlarına (baseline) karşılaştırmalı olarak devrenin qubit sayısı, devre derinliği (circuit depth) ve sınıflandırma doğruluğu (accuracy) test edilmiştir.

## Proje Yapısı

```text
├── data/           # UCI Obesity veri seti (Raw & İşlenmiş veriler)
├── notebooks/      # Analiz, görselleştirme ve keşifsel veri analizi Notebook'ları
├── results/        # Deney çıktıları, figürler ve performans metrikleri
├── src/            # Kaynak kod:
│   ├── ablation.py     # Ablasyon (Ablation) testleri ve model incelemesi
│   ├── circuit.py      # Kuantum devre tanımlamaları (Angle, Amp, Basis, Hibrit)
│   ├── preprocessing.py# Veri temizleme ve hazırlık pipeline'ı
│   └── train.py        # PennyLane + PyTorch ile VQC eğitim scripti
├── presentation.md # Proje sunum dosyası (Marp formatında)
└── README.md       # Proje ana dökümantasyonu
```

## Hedef ve Çıktılar
Bu araştırma TR Dizin standartlarına uygun, hakemli dergilerde yayımlanmak üzere hazırlanmış akademik bir çalışmadır.
> **Benchmark Karşılaştırmaları:**
> - Hibrit VQC (PennyLane + PyTorch)
> - Baseline VQC (Basis Encoding)
> - Klasik SVM (scikit-learn rbf-kernel)

## Takım & İş Bölümü (YGA 2026 Grup 14)
- **Atakan Yılmaz:** Simülasyon / Kod (PennyLane VQC, Eğitim optimizasyonu)
- **Emine Gülmez:** Veri Sorumlusu (Veri temizleme, Feature Importance)
- **Enes Furkan Kaya:** Veri Sorumlusu (Veri pipeline, Kaynak matrisi, Literatür karşılaştırma)
- **Tevfik Metin:** Algoritma Analiz Sorumlusu (Makale yazımı, Koordinasyon, Submission)

## Kurulum
Projenin bağımlılıklarını kurmak için aşağıdaki komutu çalıştırabilirsiniz:

```bash
pip install pennylane numpy pandas scikit-learn matplotlib torch
```

## Kullanım

Eğitim sürecini başlatmak için `src/train.py` dosyasını çalıştırabilirsiniz:
```bash
python src/train.py
```
