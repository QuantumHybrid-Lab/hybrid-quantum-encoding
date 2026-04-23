# Hybrid Quantum Encoding Project

Bu proje, UCI Obesity veri seti üzerinde hibrit kuantum kodlama yaklaşımını test eder.

- Kategorik değişkenler için: Angle Encoding
- Sürekli değişkenler için: Amplitude Encoding
- Model: PennyLane + PyTorch tabanlı hibrit kuantum-klasik sınıflandırıcı

Ana hedef, klasik taban çizgileriyle karşılaştırmalı olarak hibrit encoding yaklaşımının performansını incelemektir.
Kalp hastalığı ve kanser veri setleri de hibrit encoding deneyleri için kullanılacaktır.

## Hızlı Başlangıç

```bash
pip install pennylane numpy pandas scikit-learn matplotlib seaborn torch xgboost python-docx
```

Ana eğitimi çalıştırma:

```bash
python src/train.py
```

Ablation çalışması:

```bash
python src/ablation.py
```

## Proje Yapısı

```
hybrid-quantum-encoding/
├── data/
│   ├── obesity/        # UCI Obesity veri seti (ham + temiz)
│   ├── cancer/         # WDBC Breast Cancer veri seti
│   └── heart/          # Cleveland Heart Disease veri seti
├── src/
│   ├── preprocessing.py
│   ├── circuit.py
│   ├── train.py
│   └── ablation.py
├── scripts/            # Klasik baseline analizleri
│   ├── veri_analizi.py
│   ├── kalp_analizi.py
│   ├── kanser_analizi.py
│   └── create_ieee_tables.py
├── results/            # Eğitim çıktıları, grafikler
├── presentation/       # Sunum dosyaları (Marp + PDF)
└── notebooks/
```

## Kaynak Dosyalar

### src/

- `src/preprocessing.py` — Obezite verisini hibrit encoding için hazırlar (angle + amplitude ayrımı, L2 norm, train/test split)
- `src/circuit.py` — Angle, Amplitude ve Hybrid kuantum devrelerini tanımlar; kaynak ve derinlik ölçümü
- `src/train.py` — Ana hibrit model eğitimi (PennyLane devresi + klasik çıkış katmanı, loss/confusion matrix görselleştirme)
- `src/ablation.py` — Tek çıkışlı vs. çok çıkışlı devre mimarilerini karşılaştırır

### scripts/

- `scripts/veri_analizi.py` — Obezite verisi klasik SVM baseline (top-3 özellik, L2 norm)
- `scripts/kalp_analizi.py` — Cleveland heart disease klasik baseline (XGBoost özellik seçimi + SVM)
- `scripts/kanser_analizi.py` — WDBC cancer klasik baseline (RandomForest özellik seçimi + SVM)
- `scripts/create_ieee_tables.py` — IEEE formatında tablo çıktısı üretir (`results/` altına `.docx`)

### data/

- `data/obesity/ObesityDataSet_raw_and_data_sinthetic.csv` — Ham obezite verisi
- `data/obesity/temiz_obezite_verisi.csv` — Ön işlenmiş obezite verisi
- `data/cancer/wdbc.data` — WDBC breast cancer verisi
- `data/heart/processed.cleveland.data` — Cleveland heart disease verisi

### results/

- `results/improved_training_curves.png` — Eğitim metrik eğrileri
- `results/improved_confusion_matrix.png` — Karışıklık matrisi

## Çalışma Akışı

1. `src/preprocessing.py` ile veri hazırlama mantığını doğrula
2. `src/train.py` ile ana hibrit modeli eğit
3. `src/ablation.py` ile tasarım tercihlerini kıyasla
4. `scripts/create_ieee_tables.py` ile rapor tablolarını üret
5. Sonuçları `results/` ve `presentation/` altına aktar
