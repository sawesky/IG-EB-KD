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
        self.fc1 = nn.Linear(channels2 * 7 * 7, hidden) # hard-coded
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
    
class CifarCNN(nn.Module):
    def __init__(self, channels1=16, channels2=32, channels3=64, hidden=128, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, channels1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels1, channels2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels2, channels3, kernel_size=3, padding=1)

        # CIFAR: 32x32
        # after pool after conv2: 16x16
        # after pool after conv3: 8x8
        self.fc1 = nn.Linear(channels3 * 8 * 8, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)

        if return_features:
            return logits, h

        return logits

# basic residual block for CIFAR ResNet
# standard CIFAR ResNet uses 3x3 convolutions and no bottleneck
class BasicBlock(nn.Module):
     
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class CifarResNet(nn.Module):
    """
    Standard CIFAR ResNet.

    For CIFAR ResNets:
        depth = 6n + 2

    Therefore:
        depth=20 -> n=3 blocks per stage
        depth=56 -> n=9 blocks per stage
    """

    def __init__(self, depth=20, base_channels=16, num_classes=10):
        super().__init__()

        if (depth - 2) % 6 != 0:
            raise ValueError(
                f"CIFAR ResNet depth must satisfy depth = 6n + 2, got depth={depth}"
            )

        blocks_per_stage = (depth - 2) // 6

        self.depth = depth
        self.blocks_per_stage = blocks_per_stage
        self.in_planes = base_channels

        self.conv1 = nn.Conv2d(
            3,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.layer1 = self._make_layer(
            planes=base_channels,
            num_blocks=blocks_per_stage,
            stride=1,
        )
        self.layer2 = self._make_layer(
            planes=base_channels * 2,
            num_blocks=blocks_per_stage,
            stride=2,
        )
        self.layer3 = self._make_layer(
            planes=base_channels * 4,
            num_blocks=blocks_per_stage,
            stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        layers = []

        layers.append(BasicBlock(self.in_planes, planes, stride))
        self.in_planes = planes

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        h = torch.flatten(x, 1)
        logits = self.fc(h)

        if return_features:
            return logits, h

        return logits

class WideBasicBlock(nn.Module):
    """
    Basic WideResNet block for CIFAR.

    WRN uses depth = 6n + 4.
    Example:
        WRN-16-2 -> depth=16, widen_factor=2, n=2
        WRN-40-2 -> depth=40, widen_factor=2, n=6
    """

    def __init__(self, in_planes, out_planes, stride=1, dropout_rate=0.0):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.dropout_rate = dropout_rate

        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = None

    def forward(self, x):
        out = F.relu(self.bn1(x))

        if self.shortcut is not None:
            shortcut = self.shortcut(out)
        else:
            shortcut = x

        out = self.conv1(out)
        out = F.relu(self.bn2(out))

        if self.dropout_rate > 0.0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        out = self.conv2(out)

        return out + shortcut


class CifarWideResNet(nn.Module):
    """
    WideResNet for CIFAR.

    Uses the standard WRN depth rule:
        depth = 6n + 4

    Examples:
        WRN-16-2 -> depth=16, widen_factor=2
        WRN-40-2 -> depth=40, widen_factor=2
    """

    def __init__(self, depth=16, widen_factor=2, dropout_rate=0.0, num_classes=10):
        super().__init__()

        if (depth - 4) % 6 != 0:
            raise ValueError(
                f"WideResNet depth must satisfy depth = 6n + 4, got depth={depth}"
            )

        blocks_per_stage = (depth - 4) // 6

        channels = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor,
        ]

        self.depth = depth
        self.widen_factor = widen_factor
        self.blocks_per_stage = blocks_per_stage

        self.conv1 = nn.Conv2d(
            3,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.layer1 = self._make_layer(
            in_planes=channels[0],
            out_planes=channels[1],
            num_blocks=blocks_per_stage,
            stride=1,
            dropout_rate=dropout_rate,
        )

        self.layer2 = self._make_layer(
            in_planes=channels[1],
            out_planes=channels[2],
            num_blocks=blocks_per_stage,
            stride=2,
            dropout_rate=dropout_rate,
        )

        self.layer3 = self._make_layer(
            in_planes=channels[2],
            out_planes=channels[3],
            num_blocks=blocks_per_stage,
            stride=2,
            dropout_rate=dropout_rate,
        )

        self.bn = nn.BatchNorm2d(channels[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], num_classes)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride, dropout_rate):
        layers = []

        layers.append(
            WideBasicBlock(
                in_planes=in_planes,
                out_planes=out_planes,
                stride=stride,
                dropout_rate=dropout_rate,
            )
        )

        for _ in range(1, num_blocks):
            layers.append(
                WideBasicBlock(
                    in_planes=out_planes,
                    out_planes=out_planes,
                    stride=1,
                    dropout_rate=dropout_rate,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.relu(self.bn(x))
        x = self.avgpool(x)

        h = torch.flatten(x, 1)
        logits = self.fc(h)

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
            num_classes=model_cfg.get("num_classes", 10),
        )

    if name == "cifar_cnn":
        return CifarCNN(
            channels1=model_cfg["channels1"],
            channels2=model_cfg["channels2"],
            channels3=model_cfg["channels3"],
            hidden=model_cfg["hidden"],
            num_classes=model_cfg.get("num_classes", 10),
        )

    if name == "cifar_resnet":
        return CifarResNet(
            depth=model_cfg["depth"],
            base_channels=model_cfg["base_channels"],
            num_classes=model_cfg.get("num_classes", 10),
        )
    
    if name == "cifar_wrn":
        return CifarWideResNet(
            depth=model_cfg["depth"],
            widen_factor=model_cfg["widen_factor"],
            dropout_rate=model_cfg.get("dropout_rate", 0.0),
            num_classes=model_cfg.get("num_classes", 10),
        )
    
    raise ValueError(f"Unknown model name: {name}")