import pennylane as qml

Q3_ANGLE_QUBITS = 10
Q3_ANGLE_LAYERS = 2
dev_q3_angle = qml.device("default.qubit", wires=Q3_ANGLE_QUBITS)

@qml.qnode(dev_q3_angle)
def q3_angle_branch(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(Q3_ANGLE_QUBITS), rotation='Y')
    for layer in range(Q3_ANGLE_LAYERS):
        for i in range(Q3_ANGLE_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(Q3_ANGLE_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % Q3_ANGLE_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(Q3_ANGLE_QUBITS)]