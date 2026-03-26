import pennylane as qml
import numpy as np

# ── Angle Encoding ──
dev_angle = qml.device('default.qubit', wires=8)

@qml.qnode(dev_angle)
def angle_circuit(x, weights):
    qml.AngleEmbedding(x, wires=range(8), rotation='Y')
    qml.BasicEntanglerLayers(weights, wires=range(8))
    return qml.expval(qml.PauliZ(0))

# ── Amplitude Encoding ──
dev_amp = qml.device('default.qubit', wires=3)

@qml.qnode(dev_amp)
def amplitude_circuit(x, weights):
    qml.AmplitudeEmbedding(x, wires=range(3), normalize=True)
    qml.BasicEntanglerLayers(weights, wires=range(3))
    return qml.expval(qml.PauliZ(0))

# ── Hibrit Encoding ──
dev_hybrid = qml.device('default.qubit', wires=11)

@qml.qnode(dev_hybrid)
def hybrid_circuit(x_cat, x_cont, weights):
    qml.AngleEmbedding(x_cat, wires=range(8), rotation='Y')
    qml.AmplitudeEmbedding(x_cont, wires=range(8, 11), normalize=True)
    qml.BasicEntanglerLayers(weights, wires=range(11))
    return qml.expval(qml.PauliZ(0))

# ── Devre çizimlerini test et ──
if __name__ == "__main__":
    x_cat  = np.random.uniform(0, np.pi, 8)
    x_cont = np.random.rand(8)
    x_cont = x_cont / np.linalg.norm(x_cont)

    w_angle  = np.random.randn(3, 8)
    w_amp    = np.random.randn(3, 3)
    w_hybrid = np.random.randn(3, 11)

    print("── Angle Encoding Devresi ──")
    print(qml.draw(angle_circuit)(x_cat, w_angle))

    print("\n── Amplitude Encoding Devresi ──")
    print(qml.draw(amplitude_circuit)(x_cont, w_amp))

    print("\n── Hibrit Encoding Devresi ──")
    print(qml.draw(hybrid_circuit)(x_cat, x_cont, w_hybrid))

    print("\n── Devre Metrikleri ──")
    angle_specs  = qml.specs(angle_circuit)(x_cat, w_angle)
    amp_specs    = qml.specs(amplitude_circuit)(x_cont, w_amp)
    hybrid_specs = qml.specs(hybrid_circuit)(x_cat, x_cont, w_hybrid)

    print(f"Angle   — derinlik: {angle_specs['resources'].depth}, kübit: {angle_specs['resources'].num_wires}")
    print(f"Amplitude — derinlik: {amp_specs['resources'].depth}, kübit: {amp_specs['resources'].num_wires}")
    print(f"Hibrit  — derinlik: {hybrid_specs['resources'].depth}, kübit: {hybrid_specs['resources'].num_wires}")