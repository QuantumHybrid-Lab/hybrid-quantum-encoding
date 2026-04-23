import pennylane as qml

Q2_QUBITS = 10
Q2_LAYERS = 4
dev_q2 = qml.device("default.qubit", wires=Q2_QUBITS)

@qml.qnode(dev_q2)
def q2_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(Q2_QUBITS), rotation='Y')
    for layer in range(Q2_LAYERS):
        for i in range(Q2_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(Q2_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % Q2_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(Q2_QUBITS)]