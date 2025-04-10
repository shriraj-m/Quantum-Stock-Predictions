import pennylane as qp
import torch
from pennylane import numpy as np
from torch.nn import Module


class QuantumLayer(Module):
    def __init__(self, num_qubits, num_layers):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.device = qp.device("default.qubit", wires=num_qubits)
        weight_shapes = {"weights": (num_layers, num_qubits)}
        
        # Create the QNode from the circuit
        self.qnode = qp.QNode(self._circuit, self.device)
        self.quantumlayer = qp.qnn.TorchLayer(self.qnode, weight_shapes)

    def _circuit(self, inputs, weights):
        for x in range(self.num_qubits):
            qp.RY(inputs[x], wires=x) # SINGLE QUBIT Y ROTATION

        for i in range(self.num_layers):
            for j in range(self.num_qubits):
                qp.RZ(weights[i][j], wires=j) # SINGLE QUBIT Z ROTATION
            for j in range(self.num_qubits - 1):
                qp.CNOT(wires=[j, j+1]) # CNOT OPERATION
            qp.CNOT(wires=[self.num_qubits - 1, 0]) # FULL RING ENTANGLEMENT

        # Returns expectation value.
        return [qp.expval(qp.PauliZ(i)) for i in range(self.num_qubits)]

    def forward(self, x):
        # Process each sample in the batch
        batch_size = x.shape[0]
        results = []
        for i in range(batch_size):
            results.append(self.quantumlayer(x[i]))
        return torch.stack(results)
