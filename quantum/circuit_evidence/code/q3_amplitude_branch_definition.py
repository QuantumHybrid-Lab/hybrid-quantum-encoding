import pennylane as qml

Q3_AMP_QUBITS = 4
Q3_AMP_LAYERS = 2
dev_q3_amp = qml.device("default.qubit", wires=Q3_AMP_QUBITS)

@qml.qnode(dev_q3_amp)
def q3_amplitude_branch(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(Q3_AMP_QUBITS), normalize=True)
    for layer in range(Q3_AMP_LAYERS):
        for i in range(Q3_AMP_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(Q3_AMP_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % Q3_AMP_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(Q3_AMP_QUBITS)]