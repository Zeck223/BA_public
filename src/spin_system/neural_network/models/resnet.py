"""Spin-grid ResNet18 architecture.

The model uses pre-activation residual blocks for spin-grid regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Residual block with two pre-activated 3x3 convolutions.

    The skip path passes the original input unchanged when ``downsample`` is
    absent. Otherwise, it applies ``downsample`` to the batch-normalized and
    activated input. The two paths are added without a final activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Spatial stride of the first convolution.
        downsample: Optional projection for the skip path.
        padding_mode: Padding mode for the convolutions.

    Returns:
        BasicBlock: A residual block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        padding_mode: str = "zeros",
    ) -> None:
        """Initialize the convolutional and skip paths.

        Args:
            in_channels: Number of channels in the input tensor.
            out_channels: Number of channels produced by each convolution.
            stride: Spatial stride of the first convolution. Defaults to 1.
            downsample: Optional projection applied to the
                pre-activated input so that the skip path matches the output
                shape of the convolutional path. Defaults to None.
            padding_mode: Padding mode for the 3x3 convolutions.
                Defaults to "zeros".

        Returns:
            None: Initializes the block in place.
        """
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            padding_mode=padding_mode,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the convolutional path to the identity or projected skip path.

        Args:
            x: Input batch in NCHW format with shape
                ``(N, in_channels, H, W)``.

        Returns:
            torch.Tensor: Block output with shape
                ``(N, out_channels, ceil(H / stride), ceil(W / stride))``.
        """
        preactivated = self.relu(self.bn1(x))
        identity = x if self.downsample is None else self.downsample(preactivated)

        out = self.conv1(preactivated)
        out = self.conv2(self.relu(self.bn2(out)))
        return out + identity


class SpinResNet18(nn.Module):
    """Pre-activation ResNet18 for spin-grid regression.

    Four stages contain two residual blocks each, with 20, 40, 80, and 160
    output channels. Adaptive average pooling produces a fixed-size feature
    vector for the final linear layer. Inputs may be flattened channel-last
    samples, NHWC batches, or NCHW batches.

    Args:
        num_out_features: Number of values predicted per sample.
        L: Height and width of the square input grid.
        dropout: Dropout probability before the final layer.
        in_channels: Number of channels in each input grid position.

    Returns:
        SpinResNet18: A residual neural network for spin-grid regression.
    """

    _padding_mode = "zeros"
    _base_channels = 20
    _blocks_per_stage = 2

    def __init__(
        self,
        num_out_features: int = 2,
        L: int = 100,
        dropout: float = 0.0,
        in_channels: int = 3,
    ) -> None:
        """Initialize the ResNet18 and its weights.

        Args:
            num_out_features: Number of values predicted per sample.
                Defaults to 2.
            L: Positive height and width of the square input grid.
                Defaults to 100. Spatial convolutions use zero padding.
            dropout: Dropout probability before the final linear layer.
                Defaults to 0.0.
            in_channels: Positive number of channels per grid position.
                Defaults to 3.

        Returns:
            None: The network is initialized in place.

        Raise:
            ValueError: If ``L`` or ``in_channels`` is not positive.
        """
        super().__init__()
        if L <= 0 or in_channels <= 0:
            raise ValueError("L and in_channels must be positive")

        self.L = L
        self.n_input_channels = in_channels
        channels = self._base_channels
        self.in_channels = channels

        self.conv1 = nn.Conv2d(
            in_channels, channels, kernel_size=5, stride=2, padding=2,
            padding_mode=self._padding_mode, bias=False,
        )
        blocks_per_stage = (self._blocks_per_stage,) * 4
        self.layer1 = self._make_layer(channels, blocks_per_stage[0], stride=1)
        self.layer2 = self._make_layer(channels * 2, blocks_per_stage[1], stride=2)
        self.layer3 = self._make_layer(channels * 4, blocks_per_stage[2], stride=2)
        self.layer4 = self._make_layer(channels * 8, blocks_per_stage[3], stride=2)

        self.bn_out = nn.BatchNorm2d(channels * 8)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * 8, num_out_features)

        self._initialize_weights()

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Build a stage of residual blocks and update the current channel count.

        Args:
            out_channels: Number of output channels in each block.
            num_blocks: Number of blocks in the stage; must be positive.
            stride: Spatial stride of the first block.

        Returns:
            nn.Sequential: Residual blocks, with a projected skip path in the
                first block if its input and output dimensions differ.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = [
            BasicBlock(
                self.in_channels, out_channels, stride, downsample,
                padding_mode=self._padding_mode,
            )
        ]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(out_channels, out_channels, padding_mode=self._padding_mode)
            )
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Initialize convolutional, batch-normalization, and linear parameters.

        Convolutions use Kaiming normal initialization; batch-normalization
        scales and biases are set to one and zero. Linear weights use a normal
        distribution with standard deviation 0.01, and their biases are zero.

        Args:
            None: Uses the modules and parameters stored on this network.

        Returns:
            None: Initializes the network parameters in place.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a batch of configured square grids to NCHW format.

        Args:
            x: Flattened channel-last samples of shape
                ``(N, L * L * C)``, an NHWC batch of shape ``(N, L, L, C)``,
                or an NCHW batch of shape ``(N, C, L, L)``, where ``C`` is
                the configured input-channel count.

        Returns:
            torch.Tensor: Batch of shape ``(N, C, L, L)``. An existing NCHW
                batch is returned unchanged.

        Raise:
            ValueError: If the tensor rank or dimensions do not match one of
                the supported shapes.
        """
        if x.ndim == 2:
            expected_features = self.L * self.L * self.n_input_channels
            if x.shape[1] != expected_features:
                raise ValueError(
                    f"Expected {expected_features} flattened input features, "
                    f"got {x.shape[1]}"
                )
            x = x.reshape(
                x.shape[0], self.L, self.L, self.n_input_channels
            ).permute(0, 3, 1, 2)
        elif x.ndim == 4:
            if x.shape[1:] == (
                self.n_input_channels,
                self.L,
                self.L,
            ):
                return x
            if x.shape[1:] == (
                self.L,
                self.L,
                self.n_input_channels,
            ):
                x = x.permute(0, 3, 1, 2)
            else:
                raise ValueError(
                    "Expected an image batch shaped "
                    f"(N, {self.n_input_channels}, {self.L}, {self.L}) "
                    f"or (N, {self.L}, {self.L}, {self.n_input_channels}), "
                    f"got {tuple(x.shape)}"
                )
        else:
            raise ValueError(
                "Expected flattened samples or a four-dimensional image batch, "
                f"got a tensor with {x.ndim} dimensions"
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict output values for a batch of spin-grid representations.

        Args:
            x: Flattened channel-last, NHWC, or NCHW input batch
                matching the configured grid size and channel count.

        Returns:
            torch.Tensor: Unbounded predictions of shape
                ``(N, num_out_features)``.

        """
        x = self._to_nchw(x)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.relu(self.bn_out(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(self.dropout(x))
