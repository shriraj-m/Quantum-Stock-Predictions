import pennylane as qp
import torch
from pennylane import numpy as np
from torch.nn import Module


class QuantumLayer(Module):
    def __init__(self, _num_qubits, _num_layers):
        super().__init__()
        self.num_qubits = _num_qubits
        self.num_layers = _num_layers
        self.device = qp.device("default.qubit", wires=_num_qubits)
        weight_shapes = {"weights": (_num_layers, _num_qubits)}
        self.quantumlayer = qp.qnn.TorchLayer(self._circuit, weight_shapes)

    def _circuit(self, inputs, weights):
        for x in range(self.num_qubits):
            qp.RY(inputs[x], wires=x) # SINGLE QUBIT Y ROTATION

        for i in range(self.num_layers):
            for j in range(self.num_qubits):
                qp.RZ(weights[i][j], wires=i) # SINGLE QUBIT Z ROTATION
            for j in range(self.num_qubits - 1):
                qp.CNOT(wires=[i, i+1]) # CNOT OPERATION
            qp.CNOT(wires=[self.num_qubits - 1, 0]) # FULL RING ENTANGLEMENT

        # Returns expectation value.
        return [qp.expval(qp.PauliZ(i)) for i in range(self.num_qubits)]

    def forward(self, x):
        return self.quantumlayer(x)
