# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from torch import nn as nn

from rsl_rl.utils import get_param, resolve_nn_activation


class CNN_1D(nn.Sequential):
    """Convolutional Neural Network (CNN).

    The CNN network is a sequence of convolutional layers, optional normalization layers, optional activation functions,
    and optional pooling. The final output can be flattened.
    """

    def __init__(
        self,
        input_dim: int,
        input_channels: int,
        output_channels: tuple[int] | list[int],
        kernel_size: int | tuple[int] | list[int],
        stride: int | tuple[int] | list[int] = 1,
        dilation: int | tuple[int] | list[int] = 1,
        padding: str = "none",
        norm: str | tuple[str] | list[str] = "none",
        activation: str = "elu",
        max_pool: bool | tuple[bool] | list[bool] = False,
        global_pool: str = "none",
        flatten: bool = True,
    ) -> None:
        """Initialize the CNN.

        Args:
            input_dim: Height and width of the input.
            input_channels: Number of input channels.
            output_channels: List of output channels for each convolutional layer.
            kernel_size: List of kernel sizes for each convolutional layer or a single kernel size for all layers.
            stride: List of strides for each convolutional layer or a single stride for all layers.
            dilation: List of dilations for each convolutional layer or a single dilation for all layers.
            padding: Padding type to use. Either 'none', 'zeros', 'reflect', 'replicate', or 'circular'.
            norm: List of normalization types for each convolutional layer or a single type for all layers. Either
                'none', 'batch', or 'layer'.
            activation: Activation function to use.
            max_pool: List of booleans indicating whether to apply max pooling after each convolutional layer or a
                single boolean for all layers.
            global_pool: Global pooling type to apply at the end. Either 'none', 'max', or 'avg'.
            flatten: Whether to flatten the output tensor.
        """
        super().__init__()

        # Resolve activation function
        activation_function = resolve_nn_activation(activation)

        # Create layers sequentially
        layers = []
        last_channels = input_channels
        last_dim = input_dim
        for idx in range(len(output_channels)):
            # Get parameters for the current layer
            k = get_param(kernel_size, idx)
            s = get_param(stride, idx)
            d = get_param(dilation, idx)
            p = (
                _compute_padding(last_dim, k, s, d)
                if padding in ["zeros", "reflect", "replicate", "circular"]
                else 0
            )

            # Append convolutional layer
            layers.append(
                nn.Conv1d(
                    in_channels=last_channels,
                    out_channels=output_channels[idx],
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    dilation=d,
                    padding_mode=padding if padding in ["zeros", "reflect", "replicate", "circular"] else "zeros",
                )
            )

            # Append normalization layer if specified
            n = get_param(norm, idx)
            if n == "none":
                pass
            elif n == "batch":
                layers.append(nn.BatchNorm1d(output_channels[idx]))
            elif n == "layer":
                norm_input_dim = _compute_output_dim(last_dim, k, s, d, p)
                layers.append(nn.LayerNorm([output_channels[idx], norm_input_dim[0]]))
            else:
                raise ValueError(
                    f"Unsupported normalization type: {n}. Supported types are 'none', 'batch', and 'layer'."
                )

            # Append activation function
            layers.append(activation_function)

            # Apply max pooling if specified
            if get_param(max_pool, idx):
                layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

            # Update last channels and dimensions
            last_channels = output_channels[idx]
            last_dim = _compute_output_dim(last_dim, k, s, d, p, is_max_pool=get_param(max_pool, idx))

        # Apply global pooling if specified
        if global_pool == "none":
            pass
        elif global_pool == "max":
            layers.append(nn.AdaptiveMaxPool1d(1))
            last_dim = 1
        elif global_pool == "avg":
            layers.append(nn.AdaptiveAvgPool1d(1))
            last_dim = 1
        else:
            raise ValueError(
                f"Unsupported global pooling type: {global_pool}. Supported types are 'none', 'max', and 'avg'."
            )

        # Apply flattening if specified
        if flatten:
            layers.append(nn.Flatten(start_dim=1))

        # Store final output dimension
        self._output_channels = last_channels if not flatten else None
        self._output_dim = last_dim if not flatten else last_channels * last_dim

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    @property
    def output_channels(self) -> int | None:
        """Get the number of output channels or None if output is flattened."""
        return self._output_channels

    @property
    def output_dim(self) -> tuple[int, int] | int:
        """Get the output height and width or total output dimension if output is flattened."""
        return self._output_dim

    def init_weights(self) -> None:
        """Initialize the weights of the CNN with Xavier initialization."""
        for idx, module in enumerate(self):
            if isinstance(module, nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN."""
        for layer in self:
            x = layer(x)
        return x


def _compute_padding(input_w: int, kernel: int, stride: int, dilation: int) -> int:
    """Compute the optimal padding for the current layer.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    w = math.ceil((stride * math.floor(input_w / stride) - input_w - stride + dilation * (kernel - 1) + 1) / 2)
    return w


def _compute_output_dim(
    input_w:int,
    kernel: int,
    stride: int,
    dilation: int,
    padding:  int,
    is_max_pool: bool = False,
) -> int:
    """Compute the output height and width of the current layer.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    w = math.floor((input_w + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)

    if is_max_pool:
        w = math.ceil(w / 2)

    return w
