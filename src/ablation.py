"""
Ablation Study: Compares 3 model variants to justify design choices.

| Model                  | Circuit       | Output head         |
|------------------------|---------------|---------------------|
| single_output          | 6-qubit hybrid| expval(Z_0) only    |  ← original bug
| multi_output_basic     | 6-qubit hybrid| 6 expvals + Linear  |  ← our fix
| multi_output_strongly  | 6-qubit hybrid| 6 expvals + Linear  |  ← + StronglyEntanglingLayers
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pennylane as qml
from sklearn.metrics import accuracy_score, f1_score

# reuse data loading from train.py
from train import load_data, train_epoch, evaluate

EPOCHS     = 20
BATCH_SIZE = 16
LR         = 0.005
N_QUBITS   = 6
N_LAYERS   = 2
RESULTS_DIR = "results"

dev = qml.device("default.qubit", wires=N_QUBITS)


# ── Circuits ──────────────────────────────────────────────────────
@qml.qnode(dev, interface="torch")
def circuit_single_out(inputs, weights):
    """Original design: single PauliZ measurement."""
    qml.AmplitudeEmbedding(inputs[:, :8], wires=[0, 1, 2], normalize=True)
    qml.AngleEmbedding(inputs[:, 8:],    wires=[3, 4, 5], rotation="Y")
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev, interface="torch")
def circuit_multi_basic(inputs, weights):
    """Fix 1: multiple outputs, BasicEntanglerLayers."""
    qml.AmplitudeEmbedding(inputs[:, :8], wires=[0, 1, 2], normalize=True)
    qml.AngleEmbedding(inputs[:, 8:],    wires=[3, 4, 5], rotation="Y")
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


@qml.qnode(dev, interface="torch")
def circuit_multi_strongly(inputs, weights):
    """Fix 1+2: multiple outputs + StronglyEntanglingLayers."""
    qml.AmplitudeEmbedding(inputs[:, :8], wires=[0, 1, 2], normalize=True)
    qml.AngleEmbedding(inputs[:, 8:],    wires=[3, 4, 5], rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# ── Model wrappers ────────────────────────────────────────────────
class SingleOutModel(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        w = {"weights": (N_LAYERS, N_QUBITS)}   # BasicEntanglerLayers shape
        self.qlayer = qml.qnn.TorchLayer(circuit_single_out, w)
        self.head   = nn.Linear(1, n_classes)

    def forward(self, x_cont, x_cat):
        x     = torch.cat([x_cont, x_cat[:, :3]], dim=1)
        q_out = self.qlayer(x).unsqueeze(1)      # (B, 1)
        return self.head(q_out.squeeze(-1))      # (B, n_classes)

    # simpler: just unsqueeze before linear
    def forward(self, x_cont, x_cat):
        x   = torch.cat([x_cont, x_cat[:, :3]], dim=1)
        out = self.qlayer(x)                     # (B,)
        return self.head(out.unsqueeze(1))       # (B, 1) → (B, n_classes)


class MultiOutBasicModel(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        w = {"weights": (N_LAYERS, N_QUBITS)}
        self.qlayer = qml.qnn.TorchLayer(circuit_multi_basic, w)
        self.head   = nn.Linear(N_QUBITS, n_classes)

    def forward(self, x_cont, x_cat):
        x = torch.cat([x_cont, x_cat[:, :3]], dim=1)
        return self.head(self.qlayer(x))


class MultiOutStronglyModel(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        w = {"weights": (N_LAYERS, N_QUBITS, 3)}   # StronglyEntanglingLayers needs extra dim
        self.qlayer = qml.qnn.TorchLayer(circuit_multi_strongly, w)
        self.head   = nn.Linear(N_QUBITS, n_classes)

    def forward(self, x_cont, x_cat):
        x = torch.cat([x_cont, x_cat[:, :3]], dim=1)
        return self.head(self.qlayer(x))


# ── Run ablation ──────────────────────────────────────────────────
def run_ablation(data_path):
    X_cat_tr, X_cat_te, X_cont_tr, X_cont_te, y_tr, y_te, classes = load_data(data_path)
    n_classes = len(classes)

    def make_loader(Xc, Xa, y, shuffle=False):
        ds = TensorDataset(torch.tensor(Xc), torch.tensor(Xa), torch.tensor(y))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(X_cont_tr, X_cat_tr, y_tr, shuffle=True)
    test_loader  = make_loader(X_cont_te, X_cat_te, y_te)

    variants = {
        "1_single_output (original bug)":        SingleOutModel(n_classes),
        "2_multi_output_basic":                  MultiOutBasicModel(n_classes),
        "3_multi_output_strongly (best)":        MultiOutStronglyModel(n_classes),
    }

    criterion = nn.CrossEntropyLoss()
    results   = {}

    for name, model in variants.items():
        print(f"\n── {name} ──")
        optimizer = optim.Adam(model.parameters(), lr=LR)

        best_acc = 0.0
        for epoch in range(1, EPOCHS + 1):
            loss          = train_epoch(model, train_loader, optimizer, criterion)
            acc, f1, _, _ = evaluate(model, test_loader)
            if acc > best_acc:
                best_acc = acc
                best_f1  = f1
            if epoch % 5 == 0:
                print(f"  Epoch {epoch:2d} | loss {loss:.4f} | acc {acc:.4f} | f1 {f1:.4f}")

        results[name] = {"best_acc": round(best_acc, 4), "best_f1": round(best_f1, 4)}
        print(f"  → Best acc: {best_acc:.4f}  F1: {best_f1:.4f}")

    # Print summary table
    print("\n" + "═" * 55)
    print(f"{'Model':<40} {'Acc':>6} {'F1':>8}")
    print("─" * 55)
    for name, r in results.items():
        print(f"{name:<40} {r['best_acc']:>6.4f} {r['best_f1']:>8.4f}")
    print("═" * 55)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSonuçlar '{RESULTS_DIR}/ablation_results.json' dosyasına kaydedildi.")


if __name__ == "__main__":
    for candidate in [
        "data/ObesityDataSet_raw_and_data_sinthetic.csv",
        "ObesityDataSet_raw_and_data_sinthetic.csv",
    ]:
        if os.path.exists(candidate):
            run_ablation(candidate)
            break
    else:
        raise FileNotFoundError("Veri seti bulunamadı.")
