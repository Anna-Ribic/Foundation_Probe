import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, device: torch.device):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.name = 'linear_probe'
        self.probe = nn.Conv2d(input_dim, num_classes, kernel_size=1).to(device)

    def update_n_classes(self, n_classes: int):
        self.probe = nn.Conv2d(self.input_dim, n_classes, kernel_size=1).to(self.device)

    def forward(self, x):
        return self.probe(x)
