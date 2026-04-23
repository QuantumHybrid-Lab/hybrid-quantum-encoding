import pennylane as qml

Q1_QUBITS = 10
Q1_LAYERS = 2
dev_q1 = qml.device("default.qubit", wires=Q1_QUBITS)

@qml.qnode(dev_q1)
def q1_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(Q1_QUBITS), rotation='Y')
    for layer in range(Q1_LAYERS):
        for i in range(Q1_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(Q1_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % Q1_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(Q1_QUBITS)]