---
marp: true
theme: uncover
paginate: true
style: |
  @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600;700&display=swap');

  * { box-sizing: border-box; }

  section {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Fira Code', 'Courier New', monospace;
    font-size: 15px;
    text-align: left;
    padding: 40px 50px;
  }

  h1 {
    color: #ffffff;
    font-size: 20px;
    border-bottom: 1px solid #30363d;
    padding-bottom: 6px;
    margin-bottom: 14px;
    letter-spacing: 2px;
    text-transform: uppercase;
  }

  h2 {
    color: #58a6ff;
    font-size: 17px;
    margin-bottom: 10px;
    margin-top: 16px;
  }

  h3 {
    color: #79c0ff;
    font-size: 15px;
    margin: 8px 0;
  }

  blockquote {
    background: #161b22;
    border-left: 4px solid #3fb950;
    color: #3fb950;
    padding: 10px 16px;
    margin: 10px 0;
    font-size: 14px;
  }

  code {
    background: #161b22;
    color: #00ff41;
    font-family: 'Fira Code', monospace;
    font-size: 12px;
    padding: 1px 4px;
    border-radius: 3px;
  }

  pre {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    overflow: hidden;
  }

  pre code {
    background: transparent;
    color: #c9d1d9;
    font-size: 12px;
    padding: 0;
    line-height: 1.6;
  }

  .diff-del { color: #ff7b72; }
  .diff-add { color: #3fb950; }

  strong { color: #e3b341; }
  em     { color: #79c0ff; }

  section::after {
    font-family: 'Fira Code', monospace;
    font-size: 11px;
    color: #484f58;
    content: '[' attr(data-marpit-pagination) '/' attr(data-marpit-pagination-total) ']';
  }

  footer {
    font-family: 'Fira Code', monospace;
    font-size: 11px;
    color: #484f58;
  }

  section.title-slide {
    padding: 30px 50px;
  }

  section.title-slide h1 {
    font-size: 16px;
    color: #484f58;
    border: none;
    letter-spacing: 3px;
  }

  section.title-slide h2 {
    color: #ffffff;
    font-size: 22px;
    line-height: 1.4;
    margin-top: 6px;
  }

  section.title-slide h3 {
    color: #58a6ff;
    font-size: 15px;
    margin-top: 4px;
  }
---

<!-- _class: title-slide -->
<!-- _footer: '' -->
<!-- _paginate: false -->

# // KUANTUM KODLAMA ARAŞTIRMASI · YGA 2026 · GRUP 14

## Obezite Düzeyi Tahmininde Klasik ve Kuantum Tabanlı Yaklaşımlar

### Hibrit Kodlama Analizi

```
╔══════════════════════════════════════════════════════════════════════╗
║  ~  encoding_strategy.diff                            [git diff]    ║
╠══════════════════════════════════════════════════════════════════════╣
║  @@ feature_encoding/strategy.py @@                                 ║
║                                                                      ║
║  - KLASİK VERİ  │ UCI Obesity · 2111 satır · 16 özellik            ║
║  - KLASİK VERİ  │ Tek tip encoding  →  verimlilik kaybı            ║
║  - KLASİK VERİ  │ SVM / Klasik ML pipeline                         ║
║                                                                      ║
║  + KUANTUM KOD  │ Hibrit VQC  (Angle + Amplitude Encoding)         ║
║  + KUANTUM KOD  │ Adaptif encoding  →  karma yapı desteği          ║
║  + KUANTUM KOD  │ PennyLane + PyTorch pipeline                     ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

<!-- _footer: 'YGA 2026 · Grup 14 · Hibrit Kuantum Kodlama' -->

# // SLIDE 01 — BAŞLIK SAYFASI

## Obezite Düzeyi Tahmininde Klasik ve Kuantum Tabanlı Yaklaşımlar: Hibrit Kodlama Analizi

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  ~  encoding_strategy.diff                         [  diff  ]   │
  ├──────────────────────────────────────────────────────────────────┤
  │  @@ -1,3 +1,3 @@ feature_encoding/strategy.py                   │
  │                                                                  │
  │  - KLASİK VERİ  │ UCI Obesity (2111 satır, 16 özellik)         │
  │  - KLASİK VERİ  │ Tek tip encoding → verimlilik kaybı          │
  │  - KLASİK VERİ  │ SVM / Klasik ML pipeline                     │
  │                                                                  │
  │  + KUANTUM KOD  │ Hibrit VQC (Angle + Amplitude Encoding)      │
  │  + KUANTUM KOD  │ Adaptif encoding → karma yapı desteği        │
  │  + KUANTUM KOD  │ PennyLane + PyTorch pipeline                 │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

> **YGA 2026 — Grup 14 · Hibrit Kuantum Kodlama Analizi**

---

<!-- _footer: 'YGA 2026 · Grup 14 · Hibrit Kuantum Kodlama' -->

# // SLIDE 02 — PROBLEMİN TANIMI VE ARAŞTIRMA SORUSU

## Problem

Klasik sağlık verisinin kuantum devrelere aktarımında (encoding) **veri tipine göre verimlilik kaybı** yaşanmaktadır.

```
  Klasik Veri                    Kuantum Devre
  ┌──────────────────┐          ┌────────────────────────────┐
  │ Kategorik   [8]  │──── ? ──►│                            │
  │ (FNVC,CAEC...)   │          │  |ψ⟩ = Σᵢ αᵢ |i⟩          │
  │ Sürekli     [8]  │──── ? ──►│                            │
  │ (Weight,Age...)  │          └────────────────────────────┘
  └──────────────────┘
           ↑
      VERİMLİLİK KAYBI
      (tek encoding, tüm veri tiplerine)
```

## Araştırma Sorusu

> Karma yapılarda **adaptif kodlama seçiminin** sınıflandırma doğruluğu ve devre kaynakları (qubit / derinlik) üzerindeki etkisi nedir?

```
  Doğruluk (Accuracy)  ←────  Encoding Seçimi
  Qubit Sayısı         ←────  Adaptif Strateji
  Devre Derinliği      ←────  Karma Yapı
```

---

<!-- _footer: 'YGA 2026 · Grup 14 · Hibrit Kuantum Kodlama' -->

# // SLIDE 03 — YÖNTEM: ADAPTİF KODLAMA STRATEJİSİ

## Hibrit Encoding Mimarisi

```python
# feature_encoding/hybrid_strategy.py

# ── KATEGORİK (8 özellik) ─────────────────────────────────────────
# FNVC, CAEC, CALC, SMOKE, SCC, MTRANS, family_history, Gender
def angle_encoding(x_cat):
    for i, val in enumerate(x_cat):
        qml.RY(np.pi * val, wires=i)       # θ = π × kategorik_değer
    # |ψ⟩ = Rᵧ(θ)|0⟩  ←  tek qubit rotasyonu

# ── SÜREKLİ (8 özellik) ───────────────────────────────────────────
# Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE
def amplitude_encoding(x_cont):
    x_norm = x_cont / np.linalg.norm(x_cont)   # normalize
    qml.AmplitudeEmbedding(x_norm, wires=range(8, 11), normalize=True)
    # |ψ⟩ = Σᵢ αᵢ|i⟩  ←  tam süperpozisyon

# ── BASELINE (Referans) ────────────────────────────────────────────
def basis_encoding(x_bin):
    qml.BasisEmbedding(x_bin, wires=range(N))   # |x⟩ binary mapping
```

> **Özgün nokta:** Kategorik → Angle · Sürekli → Amplitude · aynı VQC içinde.

---

<!-- _footer: 'YGA 2026 · Grup 14 · Hibrit Kuantum Kodlama' -->

# // SLIDE 04 — TEKNİK İŞ AKIŞI VE MİMARİ

## VQC (Variational Quantum Circuit) — Terminal Çıktısı

```
$ pennylane.draw(hybrid_vqc)(X_sample)

q0 (cat): ──[Rᵧ(θ₁)]──●──────────[Rᵧ(φ₁)]──[M]─
q1 (cat): ──[Rᵧ(θ₂)]──┼──●───────[Rᵧ(φ₂)]──[M]─
q2 (cat): ──[Rᵧ(θ₃)]──┼──┼──●────[Rᵧ(φ₃)]──[M]─
q3 (cat): ──[Rᵧ(θ₄)]──┼──┼──┼──●─[Rᵧ(φ₄)]──[M]─
           ─ ─ ─ ─ ─ ─ ┼──┼──┼──┼ ─ ─ ─ ─ ─ ─ ─
q4 (con): ──[Amp]───────●──┼──┼──┼─[Rᵧ(φ₅)]──[M]─
q5 (con): ──[Amp]──────────●──┼──┼─[Rᵧ(φ₆)]──[M]─
q6 (con): ──[Amp]─────────────●──┼─[Rᵧ(φ₇)]──[M]─
q7 (con): ──[Amp]────────────────●─[Rᵧ(φ₈)]──[M]─

◄── Angle Encoding ──►◄─── Variational Layer (trainable) ───►
```

```
  ÖLÇÜM METRİKLERİ
  ├── circuit.depth()   →  devre derinliği analizi
  ├── len(dev.wires)    →  toplam qubit sayısı
  └── accuracy_score()  →  sınıflandırma doğruluğu
```

---

<!-- _footer: 'YGA 2026 · Grup 14 · Hibrit Kuantum Kodlama' -->

# // SLIDE 05 — VERİ SETİ VE HEDEF

## UCI Obesity Levels Dataset

```
  $ dataset.info()
  ┌─────────────────────────────────────────────────────┐
  │  Kaynak  : UCI Machine Learning Repository          │
  │  Satır   : 2,111   ████████████████████████████    │
  │  Özellik : 16      ████████████████                 │
  │  Sınıf   : 7       ███████                          │
  └─────────────────────────────────────────────────────┘

  $ dataset['NObeyesdad'].unique()
  ├── Insufficient_Weight    ├── Overweight_Level_II
  ├── Normal_Weight          ├── Obesity_Type_I
  ├── Overweight_Level_I     ├── Obesity_Type_II
  └── Obesity_Type_III
```

## Benchmark Karşılaştırması

```
  ┌──────────────────────┬──────────────────────────────────────┐
  │  Model               │  Araç / Kütüphane                    │
  ├──────────────────────┼──────────────────────────────────────┤
  │  Hibrit VQC          │  PennyLane + PyTorch (Adam, lr=0.01) │
  │  Baseline VQC        │  PennyLane (Basis Encoding)          │
  │  Klasik SVM          │  scikit-learn (kernel=rbf)           │
  └──────────────────────┴──────────────────────────────────────┘
```

> **Yayın Hedefi:** TR Dizin hakemli dergi · Akademik makale formatı

---

<!-- _footer: 'YGA 2026 · Grup 14 · Hibrit Kuantum Kodlama' -->

# // SLIDE 06 — LİTERATÜR VE ÖZGÜN DEĞER

## Mevcut Literatür vs. Bu Çalışma

```
  ~  literature.diff                                    [git diff]
  ┌─────────────────────────────────────────────────────────────┐
  │  @@ EPJ Quantum Technology, 2024 @@                         │
  │                                                             │
  │  - Tek tip veri (ya tam kategorik ya tam sürekli)          │
  │  - Tek encoding stratejisi tüm özelliklere uygulanır       │
  │  - Mixed-type yapılar görmezden gelinir                    │
  │                                                             │
  │  + Mixed-type (karma) özellik yapısı desteklenir           │
  │  + Veri tipine göre dinamik encoding seçimi (adaptif)      │
  │  + Kategorik ↔ Sürekli ayrımı VQC içinde birleştirilir    │
  └─────────────────────────────────────────────────────────────┘
```

## Özgün Katkı

```
  ╔═══════════════════════════════════════════════════════════════╗
  ║  ★  ÖZGÜN NOKTA                                               ║
  ║                                                               ║
  ║  Karma (mixed-type) özellik yapısına sahip gerçek dünya       ║
  ║  sağlık verisinde, özellik tipine göre dinamik encoding       ║
  ║  seçimi yapan hibrit bir VQC mimarisi → literatürde ilk kez. ║
  ║                                                               ║
  ║  Rakip çalışmalar bu ayrımı görmezden gelir:                 ║
  ║  tüm özellikler aynı encoding → verimlilik kaybı.            ║
  ╚═══════════════════════════════════════════════════════════════╝
```

---

<!-- _footer: 'YGA 2026 · Grup 14 · Hibrit Kuantum Kodlama' -->

# // SLIDE 07 — GÖREV DAĞILIMI VE ROL MATRİSİ

## Takım Yapısı

```
  $ cat team/roles.yaml

  - name:  Atakan Yılmaz
    role:  Simülasyon / Kod
    tasks:
      - PennyLane VQC implementasyonu
      - Encoding katmanları (Angle + Amplitude)
      - Model eğitimi ve optimizasyon

  - name:  Emine Gülmez
    role:  Veri Sorumlusu
    tasks:
      - Veri temizleme (preprocessing)
      - Feature Importance analizi
      - Train/test split ve cross-validation

  - name:  Enes Furkan Kaya
    role:  Veri Sorumlusu
    tasks:
      - Veri ön işleme pipeline
      - Kaynak matrisi oluşturma
      - Related work karşılaştırma tablosu

  - name:  Tevfik Metin
    role:  Algoritma Analiz Sorumlusu
    tasks:
      - Makale yazımı (TR Dizin formatı)
      - Koordinasyon ve milestone takibi
      - Submission süreci yönetimi
```

---

<!-- _footer: 'YGA 2026 · Grup 14 · Hibrit Kuantum Kodlama' -->

# // SLIDE 08 — ZAMAN ÇİZELGESİ: 8 HAFTALIK PLAN

## Proje Takvimi

```
  $ git log --oneline --graph  (planlanan)

  HAFTA   GÖREV                                        DURUM
  ──────────────────────────────────────────────────────────────────
  Hf 1  ▸ Ortam kurulumu (PennyLane, PyTorch)         [ KURULUM  ]
         ▸ Veri seti indirme ve ilk inceleme
         ▸ Literatür taraması başlangıcı

  Hf 2  ▸ Preprocessing pipeline                      [ HAZIRLIK ]
         ▸ Feature importance analizi
         ▸ Encoding strateji kararları

  Hf 3  ▸ Angle Encoding implementasyonu              [SIMÜLASYON]
         ▸ Amplitude Encoding implementasyonu
         ▸ Basis Encoding (baseline) kurulumu

  Hf 4  ▸ VQC mimarisi tasarımı ve test              [ DENEYLER ]
         ▸ Hibrit encoding entegrasyonu

  Hf 5  ▸ SVM benchmark karşılaştırması              [   ANALİZ ]
         ▸ Devre derinliği / qubit analizi

  Hf 6  ▸ Makale taslağı (Abstract + Intro)          [RAPORLAMA ]
  Hf 7  ▸ Makale tamamlama (Results + Discussion)    [  MAKALE  ]
  Hf 8  ▸ Final revizyon + Submission                [SUBMISSION ]
  ──────────────────────────────────────────────────────────────────
  ◄── KURULUM ──►◄──── SİMÜLASYON & ANALİZ ────►◄── MAKALE ──►
       Hf 1-2             Hf 3-4-5                   Hf 6-7-8
```

---

<!-- _paginate: false -->
<!-- _footer: '' -->

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   $ python hybrid_vqc.py --mode=final --dataset=obesity             ║
║                                                                      ║
║   [■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■] 100%  Training complete         ║
║                                                                      ║
║   Results:                                                           ║
║     Hybrid VQC Accuracy  :  [TBD]%                                  ║
║     Classical SVM        :  [TBD]%                                  ║
║     Circuit Depth        :  [TBD]                                   ║
║     Qubit Count          :  [TBD]                                   ║
║                                                                      ║
║   |ψ_final⟩ = Σᵢ αᵢ·AngleEnc(xᵢ) ⊗ Σⱼ βⱼ·AmpEnc(xⱼ)            ║
║                                                                      ║
║   [SUCCESS] Hibrit Kuantum Kodlama Araştırması · YGA 2026 Grup 14  ║
╚══════════════════════════════════════════════════════════════════════╝
```

> Sorular ve tartışma için teşekkürler.

---

<!-- _footer: 'YGA 2026 · Grup 14 · Hibrit Kuantum Kodlama' -->

# // TEŞEKKÜRLER

```text
            =============
          /               \
         |  QUANTUM CORE   |
          \_______________/
            |  |  |  |  |
           _|__|__|__|__|_
          |===============|
          |  o    o    o  |
          |  o    o    o  |
           \_____________/
             |  |   |  |
            _|__|___|__|_
           |=============|
           |   o     o   |
            \___________/
              | | | | |
             _|=======|_
            | o   o   o |
             \_________/
               |  |  |
              _|=====|_
             |    o    |
              \_______/
                 | |
                _|_|_
               \_____/
                  |
                 ---
```
