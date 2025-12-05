"""
LoRA (Low-Rank Adaptation) Layer Implementation for SAM3

This module implements LoRA layers that can be injected into transformer models
for efficient fine-tuning.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    LoRA layer that adds low-rank adaptation to a linear transformation.

    This implements the LoRA technique from "LoRA: Low-Rank Adaptation of Large Language Models"
    where a linear layer's weight W is augmented with a low-rank update: W' = W + BA
    where B is (out_features x r) and A is (r x in_features), with r << min(in_features, out_features)

    Args:
        in_features: Size of input features
        out_features: Size of output features
        rank: Rank of the low-rank matrices (r)
        alpha: Scaling factor for LoRA (controls the magnitude of updates)
        dropout: Dropout probability for LoRA path
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA parameters using Kaiming uniform initialization for A and zero for B."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Apply dropout to input
        x_dropout = self.dropout(x)

        # Low-rank adaptation: (x @ A^T) @ B^T
        result = F.linear(x_dropout, self.lora_B @ self.lora_A)

        # Scale by alpha/r
        return result * self.scaling

    def merge_weights(self) -> torch.Tensor:
        """
        Merge LoRA weights into a single weight matrix.

        Returns:
            Merged weight matrix of shape (out_features, in_features)
        """
        return (self.lora_B @ self.lora_A) * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation.

    This wraps an existing nn.Linear layer and adds a LoRA layer in parallel.
    The original linear layer's weights are frozen.

    Args:
        linear: Original linear layer to adapt
        rank: Rank of LoRA matrices
        alpha: LoRA scaling factor
        dropout: Dropout probability for LoRA path
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Store original linear layer and freeze it
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False

        # Store dimensions for compatibility
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Add LoRA adaptation
        self.lora = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    @property
    def weight(self):
        """Expose weight for compatibility with nn.Linear."""
        return self.linear.weight

    @property
    def bias(self):
        """Expose bias for compatibility with nn.Linear."""
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original linear transformation + LoRA adaptation.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Original linear transformation (frozen)
        result = self.linear(x)

        # Add LoRA adaptation
        result = result + self.lora(x)

        return result

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into the original linear layer.

        Returns:
            New linear layer with merged weights
        """
        merged_weight = self.linear.weight.data + self.lora.merge_weights()

        new_linear = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=self.linear.bias is not None,
        )
        new_linear.weight.data = merged_weight
        if self.linear.bias is not None:
            new_linear.bias.data = self.linear.bias.data.clone()

        return new_linear
