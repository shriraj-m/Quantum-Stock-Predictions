import torch
import torch.nn as nn

from quantum_layer import QuantumLayer
# input dim is the number of time series features (ex: past 5 prices at close)
# fc 1 maps inputs to number of qubits
# fc 2 and 3 maps quantum features back to a prediction
# quantum process through the QNN (entangling quantum layer)

class HybridQNN(nn.Module):
    def __init__(self, _input_dim, _hidden_dim, _num_qubits=4, _num_q_layers=1):
        super().__init__()
        
        self.fc1 = nn.Linear(_input_dim, _num_qubits) # Classical Layer -> matches # of qubits
        self.quantum = QuantumLayer(_num_qubits, _num_q_layers)
        self.fc2 = nn.Linear(_num_qubits, _hidden_dim)
        self.fc3 = nn.Linear(_hidden_dim, 1) # output singular value (regression model)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.quantum(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)