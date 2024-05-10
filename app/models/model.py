# Import modules
from torch import nn


# Creat basic Neural Network
class NeuralNet(nn.Module):
    """The implementation of the neural network."""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        outputs = self.stack(x)
        return outputs
