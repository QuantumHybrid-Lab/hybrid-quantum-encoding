"""
Improved Hybrid Quantum-Classical Classifier
=============================================
Obesity dataset (7-class classification)

Key fixes vs. previous implementation:
  1. Circuit returns 6 PauliZ expectations (not 1) → proper multi-class output
  2. Classical head: Linear(6 → 7) + CrossEntropyLoss (includes softmax)
  3. Mini-batch training (batch_size=16) with PyTorch Adam
  4. StronglyEntanglingLayers instead of BasicEntanglerLayers
  5. ReduceLROnPlateau + early stopping

Circuit architecture (6 qubits):
  Qubits 0-2 : AmplitudeEmbedding  ← 8 continuous features  (2^3 = 8 ✓)
  Qubits 3-5 : AngleEmbedding      ← 3 categorical features
  Variational : StronglyEntanglingLayers (n_layers × 6 × 3 params)
  Output      : expval(PauliZ) on all 6 qubits → Linear(6, 7)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pennylane as qml
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────
N_QUBITS   = 6      # 3 amplitude + 3 angle
N_LAYERS   = 2      # StronglyEntanglingLayers depth
BATCH_SIZE = 16
EPOCHS     = 100
LR         = 0.003
PATIENCE   = 20     # early-stopping patience
RESULTS_DIR = "results"

# ─────────────────────────────────────────────────────────────────
# 1.  Data loading & preprocessing
# ─────────────────────────────────────────────────────────────────
def load_data(path: str):
    df = pd.read_csv(path)

    cat_features  = ["Gender", "family_history_with_overweight",
                     "FAVC", "SMOKE", "SCC", "CAEC", "CALC", "MTRANS"]
    cont_features = ["Age", "Height", "Weight", "FCVC",
                     "NCP", "CH2O", "FAF", "TUE"]

    le = LabelEncoder()
    for col in cat_features:
        df[col] = le.fit_transform(df[col])

    # Categorical → [0, π]  (angle encoding)
    scaler_cat = MinMaxScaler(feature_range=(0, np.pi))
    X_cat = scaler_cat.fit_transform(df[cat_features]).astype(np.float32)

    # Continuous → L2-normalised  (amplitude encoding needs unit vector)
    scaler_cont = MinMaxScaler()
    X_cont_raw  = scaler_cont.fit_transform(df[cont_features]).astype(np.float32)
    X_cont      = normalize(X_cont_raw, norm="l2")

    le_target = LabelEncoder()
    y = le_target.fit_transform(df["NObeyesdad"]).astype(np.int64)

    splits = train_test_split(
        X_cat, X_cont, y,
        test_size=0.2, random_state=42, stratify=y
    )
    X_cat_tr, X_cat_te, X_cont_tr, X_cont_te, y_tr, y_te = splits
    return X_cat_tr, X_cat_te, X_cont_tr, X_cont_te, y_tr, y_te, le_target.classes_


# ─────────────────────────────────────────────────────────────────
# 2.  Quantum circuit
# ─────────────────────────────────────────────────────────────────
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def hybrid_circuit(inputs, weights):
    """
    inputs : tensor of shape (11,) — concat([x_cont(8), x_cat3(3)])
    weights: tensor of shape (N_LAYERS, N_QUBITS, 3)
    returns: list of 6 PauliZ expectations
    """
    x_cont = inputs[:, :8]   # 8 continuous features → AmplitudeEmbedding
    x_cat3 = inputs[:, 8:]   # 3 categorical features → AngleEmbedding

    qml.AmplitudeEmbedding(x_cont, wires=[0, 1, 2], normalize=True)
    qml.AngleEmbedding(x_cat3,  wires=[3, 4, 5], rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ─────────────────────────────────────────────────────────────────
# 3.  Hybrid model (quantum layer + classical head)
# ─────────────────────────────────────────────────────────────────
class HybridQNN(nn.Module):
    def __init__(self, n_classes: int = 7):
        super().__init__()
        weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
        self.qlayer   = qml.qnn.TorchLayer(hybrid_circuit, weight_shapes)
        self.head     = nn.Linear(N_QUBITS, n_classes)

    def forward(self, x_cont, x_cat):
        # Concatenate inputs: [8 cont || 3 cat] = 11-dim vector per sample
        x = torch.cat([x_cont, x_cat[:, :3]], dim=1)   # (B, 11)
        q_out = self.qlayer(x)                           # (B, 6)
        return self.head(q_out)                          # (B, 7)


# ─────────────────────────────────────────────────────────────────
# 4.  Train / evaluate helpers
# ─────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x_cont, x_cat, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x_cont, x_cat), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    for x_cont, x_cat, y in loader:
        preds.extend(model(x_cont, x_cat).argmax(dim=1).tolist())
        labels.extend(y.tolist())
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, f1, preds, labels


# ─────────────────────────────────────────────────────────────────
# 5.  Plotting helpers
# ─────────────────────────────────────────────────────────────────
def plot_curves(train_losses, test_accs, test_f1s, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, "b-o", markersize=4)
    ax1.set_title("İyileştirilmiş Hibrit Model - Train Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")

    ax2.plot(test_accs, "g-o", markersize=4, label="Accuracy")
    ax2.plot(test_f1s,  "r-s", markersize=4, label="Macro F1")
    ax2.set_title("İyileştirilmiş Hibrit Model - Test Metrics")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score"); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "improved_training_curves.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(labels, preds, classes, out_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
    ax.set_title("İyileştirilmiş Hibrit Model - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "improved_confusion_matrix.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Locate dataset
    for candidate in [
        "temiz_obezite_verisi.csv",
        "data/ObesityDataSet_raw_and_data_sinthetic.csv",
        "ObesityDataSet_raw_and_data_sinthetic.csv",
    ]:
        if os.path.exists(candidate):
            DATA_PATH = candidate
            break
    else:
        raise FileNotFoundError("Veri seti bulunamadı. 'data/' klasörüne koyunuz.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────
    X_cat_tr, X_cat_te, X_cont_tr, X_cont_te, y_tr, y_te, classes = load_data(DATA_PATH)
    n_classes = len(classes)
    print(f"Veri yüklendi | Eğitim: {len(y_tr)}, Test: {len(y_te)}, Sınıf: {n_classes}")

    def to_loader(Xc, Xa, y, shuffle=False):
        ds = TensorDataset(
            torch.tensor(Xc), torch.tensor(Xa), torch.tensor(y)
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = to_loader(X_cont_tr, X_cat_tr, y_tr, shuffle=True)
    test_loader  = to_loader(X_cont_te, X_cat_te, y_te)

    # ── Model, optimiser, scheduler ───────────────────────────
    model     = HybridQNN(n_classes=n_classes)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    print(f"\nDevre    : {N_QUBITS} kübit, {N_LAYERS} katman StronglyEntanglingLayers")
    print(f"Model    : HybridQNN (kuantum + Linear({N_QUBITS}→{n_classes}))")
    print(f"Optimizer: Adam lr={LR}, batch={BATCH_SIZE}, epoch={EPOCHS}")
    print("─" * 60)

    train_losses, test_accs, test_f1s = [], [], []
    history = []
    best_acc, best_state = 0.0, None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        loss        = train_epoch(model, train_loader, optimizer, criterion)
        acc, f1, _, _ = evaluate(model, test_loader)
        scheduler.step(loss)

        train_losses.append(loss)
        test_accs.append(acc)
        test_f1s.append(f1)
        history.append({
            "epoch": epoch,
            "train_loss": round(loss, 4),
            "test_acc": round(acc, 4),
            "test_f1": round(f1, 4)
        })

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {loss:.4f} "
                  f"| Acc: {acc:.4f} | F1: {f1:.4f}")

        if no_improve >= PATIENCE:
            print(f"\nErken durdurma: {PATIENCE} epoch boyunca iyileşme yok.")
            break

    print("─" * 60)
    print(f"En iyi test accuracy : {best_acc:.4f}")

    # ── Final evaluation with best weights ─────────────────────
    model.load_state_dict(best_state)
    best_acc, best_f1, best_preds, best_labels = evaluate(model, test_loader)
    print(f"En iyi model Accuracy: {best_acc:.4f}  |  Macro F1: {best_f1:.4f}")

    # ── Save plots and academic reports ────────────────────────
    plot_curves(train_losses, test_accs, test_f1s, RESULTS_DIR)
    plot_confusion_matrix(best_labels, best_preds, classes, RESULTS_DIR)

    # Detaylı metrikleri CSV dosyasına kaydet
    pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, "training_logs_per_epoch.csv"), index=False)

    # Genel sonuç özetini TXT olarak makale yazımı için kaydet
    with open(os.path.join(RESULTS_DIR, "academic_report_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== HİBRİT KUANTUM MODELİ EĞİTİM RAPORU ===\n")
        f.write(f"Model: HybridQNN ({N_QUBITS} Kübit, {N_LAYERS} Katman StronglyEntanglingLayers)\n")
        f.write(f"Hiperparametreler: BATCH={BATCH_SIZE}, LR={LR}, MAX_EPOCHS={EPOCHS}, PATIENCE={PATIENCE}\n")
        f.write("-" * 45 + "\n")
        f.write(f"En İyi Test Doğruluğu (Accuracy) : {best_acc:.4f}\n")
        f.write(f"En İyi Test Macro F1 Skoru       : {best_f1:.4f}\n")
        f.write(f"Toplam Tamamlanan Epoch Sayısı   : {epoch}\n")
        f.write("-" * 45 + "\n")
        f.write("Not: Her bir epoch için detaylı test sonuçları 'training_logs_per_epoch.csv' dosyasındadır.\n")

    from sklearn.metrics import classification_report
    report_text = classification_report(best_labels, best_preds, target_names=classes)
    print("\n" + report_text)
    
    with open(os.path.join(RESULTS_DIR, "academic_report_summary.txt"), "a", encoding="utf-8") as f:
        f.write("\n=== SINIF BAZINDA SONUÇLAR (CLASSIFICATION REPORT) ===\n")
        f.write(report_text)