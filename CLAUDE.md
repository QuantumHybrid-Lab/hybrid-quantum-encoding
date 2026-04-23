# CLAUDE.md

## Project Overview
This repository studies hybrid quantum encoding for multiclass obesity classification.
Primary workflow is built around the UCI Obesity dataset and PennyLane + PyTorch experiments.

## Main Goal
Improve classification performance by combining:
- Angle encoding for categorical features
- Amplitude encoding for continuous features

## Key Entry Points
- `src/train.py`: Main hybrid model training script
- `src/ablation.py`: Ablation experiments for architecture choices
- `src/preprocessing.py`: Data preparation pipeline
- `src/circuit.py`: Quantum circuit definitions and resource checks

## Data Sources
Primary dataset:
- `data/obesity/ObesityDataSet_raw_and_data_sinthetic.csv`
- `data/obesity/temiz_obezite_verisi.csv`

Side-analysis datasets (to be used in hybrid encoding experiments):
- `data/cancer/wdbc.data` (breast cancer)
- `data/cancer/breast+cancer+wisconsin+diagnostic.zip`
- `data/heart/processed.cleveland.data` (heart disease)
- `data/heart/heart+disease.zip`

## Environment
Recommended Python packages:
- pennylane
- torch
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- xgboost
- python-docx

## Commands
Install dependencies:
```bash
pip install pennylane numpy pandas scikit-learn matplotlib seaborn torch xgboost python-docx
```

Run main training:
```bash
python src/train.py
```

Run ablation:
```bash
python src/ablation.py
```

## Notes for Contributors
- Keep obesity workflow as the main experimental track.
- Cancer and heart datasets will be used for hybrid encoding experiments (not just classical baselines).
- `scripts/` contains classical baseline analyses; main quantum experiments live in `src/`.
- When reporting feature counts, do not mix target labels with input features.
- Feature count must be compatible with amplitude encoding (2^n): use 3 features → 2^2=4 padded, or 8 features → 2^3=8.
- Store generated outputs in `results/`.
