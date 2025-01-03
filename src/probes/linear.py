import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes: int, device: torch.device):
        super().__init__()
        if type(input_dim) is not int:
            self.input_dim = sum(input_dim)
        else:
            self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device
        self.name = 'linear_probe'
        self.probe = nn.Conv2d(input_dim, num_classes, kernel_size=1).to(device)

    def update_input_dim(self, input_dim):
        if type(input_dim) is not int:
            self.input_dim = sum(input_dim)
        self.probe = nn.Conv2d(self.input_dim, self.num_classes, kernel_size=1).to(self.device)

    def update_n_classes(self, n_classes: int):
        self.num_classes = n_classes
        self.probe = nn.Conv2d(self.input_dim, n_classes, kernel_size=1).to(self.device)

    def forward(self, feats):
        if type(feats) is list:
            feats = torch.cat(feats, dim=1)
        print(self.probe(feats).shape)
        return self.probe(feats)
