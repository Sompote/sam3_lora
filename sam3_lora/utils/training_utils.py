"""Training utility functions."""

import torch
import torch.nn as nn


def print_trainable_parameters(model: nn.Module):
    """
    Print summary of trainable parameters in the model.

    Args:
        model: Model to analyze
    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


def get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
