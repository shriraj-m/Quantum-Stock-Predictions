import torch
import torch.nn as nn

from quantum_layer import QuantumLayer
# input dim is the number of time series features (ex: past 5 prices at close)
# fc 1 maps inputs to number of qubits
# fc 2 and 3 maps quantum features back to a prediction
# quantum process through the QNN (entangling quantum layer)

class HybridQNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_qubits=4, num_q_layers=1):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, num_qubits) # Classical Layer -> matches # of qubits
        self.quantum = QuantumLayer(num_qubits, num_q_layers)
        self.fc2 = nn.Linear(num_qubits, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) # output singular value (regression model)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.quantum(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)