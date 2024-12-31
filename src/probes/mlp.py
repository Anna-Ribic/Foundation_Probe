import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes: int, device: torch.device):
        super().__init__()

        if type(input_dim) is not int:
            self.input_dim = sum(input_dim)
        else:
            self.input_dim = input_dim

        self.num_classes = num_classes
        self.device = device
        self.name = 'mlp_probe'

        self.probe = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(True),
            nn.Linear(self.input_dim, num_classes)
        ).to(device)

    def update_input_dim(self, input_dim):
        if type(input_dim) is not int:
            self.input_dim = sum(input_dim)
        else:
            self.input_dim = input_dim

        self.probe = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(True),
            nn.Linear(self.input_dim, self.num_classes)
        ).to(self.device)

    def update_n_classes(self, n_classes: int):
        self.num_classes = n_classes
        self.probe = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(True),
            nn.Linear(self.input_dim, n_classes)
        ).to(self.device)

    def forward(self, feats):
        b, c, h, w = feats.shape
        feats = feats.flatten(-2).transpose(1, 2)  # Flatten spatial dimensions and transpose
        feats = self.probe(feats)  # Pass through MLP
        feats = feats.transpose(1, 2).view(b, -1, h, w)  # Reshape back to original dimensions

        return feats
