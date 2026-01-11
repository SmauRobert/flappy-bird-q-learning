import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], num_actions: int) -> None:
        """
        Deep Q-Network with 3 Convolutional Layers and 2 Linear Layers.
        Args:
            input_shape: (Channels, Height, Width) -> e.g., (4, 84, 84)
            num_actions: Number of possible actions -> e.g., 2
        """
        super(Model, self).__init__()

        # 1. Convolutional Layers (Feature Extraction)
        self.features = nn.Sequential(
            # Conv 1: Low-level features (edges)
            nn.Conv2d(
                in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4
            ),
            nn.ReLU(),
            # Conv 2: Shapes (pipes, bird)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Conv 3: Higher-level relationships (distance gaps)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # 2. Dynamic Size Calculation
        self.fc_input_dim = self._get_conv_out(input_shape)

        # 3. Fully Connected Layers (Decision Making)
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512), nn.ReLU(), nn.Linear(512, num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Convolutions
        x = self.features(x)

        # 2. Flatten: (Batch_Size, Channels, H, W) -> (Batch_Size, Features)
        x = x.view(x.size(0), -1)

        # 3. Linear Layers
        return self.fc(x)

    def _get_conv_out(self, shape: tuple[int, int, int]) -> int:
        """
        Passes a dummy tensor through the conv layers to calculate flattened output size.
        """
        # Create a dummy input with batch_size=1
        o = torch.zeros(1, *shape)
        o = self.features(o)
        # Calculate product of dimensions (channels * height * width)
        return int(np.prod(o.size()))
