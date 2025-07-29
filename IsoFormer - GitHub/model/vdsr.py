from math import sqrt

import torch
from torch import nn


class ConvReLU3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ConvReLU3D, self).__init__()
        self.conv = nn.Conv3d(channels, channels, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)
        return out


class VDSR3D(nn.Module):
    def __init__(self) -> None:
        super(VDSR3D, self).__init__()
        # Input layer
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False),
            nn.ReLU(True),
        )

        # Features trunk blocks
        trunk = []
        for _ in range(18):
            trunk.append(ConvReLU3D(64))
        self.trunk = nn.Sequential(*trunk)

        # Output layer
        self.conv2 = nn.Conv3d(64, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False)

        # Initialize model weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)

        out = torch.add(out, identity)
        print(out.shape)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                module.weight.data.normal_(0.0, sqrt(2 / (module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2] * module.out_channels)))