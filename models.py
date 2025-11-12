import torch
import torch.nn as nn
import torch.nn.functional as F


# Flatten and simple softmax model for softmax regression
class SoftmaxModel(nn.Module):
    def __init__(self, input_dim=28*28, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

# Two-layer MLP used in paper (50 units each layer)
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden=50, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Simple CNN 