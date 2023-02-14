import torch
import torch.nn as nn

class GeneratorBase(nn.Module):
    def __init__(self):
        super().__init__()

    def train_step(self):
        raise NotImplementedError()

    def forward(self):
        raise NotImplementedError()

    def save(self, path):
        torch.save(self.state_dict(), path)