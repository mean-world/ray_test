import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.input = nn.Linear(6, 64)
        self.fc = nn.Linear(64, 32)
        self.output = nn.Linear(32, 6)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.input(out)
        out = self.relu(out)
        out = self.fc(out)
        out = self.relu(out)
        out = self.output(out)
        out = torch.sigmoid(out)
        return out[:, -1, :]
    