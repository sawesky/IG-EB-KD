import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """
    Small LeNet-style CNN.
    """

    def __init__(self, channels1=32, channels2=64, hidden=128, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(channels1, channels2, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(channels2 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)

        if return_features:
            return logits, h
        return logits


def make_model(model_cfg):
    name = model_cfg["name"]

    if name == "lenet":
        return LeNet(
            channels1=model_cfg["channels1"],
            channels2=model_cfg["channels2"],
            hidden=model_cfg["hidden"],
        )

    raise ValueError(f"Unknown model name: {name}")