#!/usr/bin/env python3
"""
Test script for standalone SAM3 LoRA.

This tests that the package works without SAM3 dependencies.
"""

import torch
from sam3_lora import LoRAConfig, inject_lora_into_model
from sam3_lora.model import SimpleTransformer, SimpleSegmentationModel
from sam3_lora.utils import print_trainable_parameters


def test_lora_injection():
    """Test LoRA injection on simple models."""
    print("="*60)
    print("Testing Standalone SAM3 LoRA")
    print("="*60)

    # Test 1: Simple Transformer
    print("\nTest 1: LoRA Injection on Simple Transformer")
    print("-"*60)

    model = SimpleTransformer(d_model=256, nhead=8)
    print("Before LoRA:")
    print_trainable_parameters(model)

    lora_config = LoRAConfig(
        rank=8,
        alpha=16.0,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "linear1", "linear2"]
    )

    model = inject_lora_into_model(model, lora_config, verbose=False)

    print("\nAfter LoRA:")
    print_trainable_parameters(model)

    # Test forward pass
    batch_size = 2
    seq_len = 10
    d_model = 256

    src = torch.randn(batch_size, seq_len, d_model)
    tgt = torch.randn(batch_size, seq_len, d_model)

    with torch.no_grad():
        output = model(src, tgt)

    print(f"\n✓ Forward pass successful!")
    print(f"  Input shape: {src.shape}")
    print(f"  Output shape: {output.shape}")

    # Test 2: Simple Segmentation Model
    print("\n\nTest 2: LoRA Injection on Segmentation Model")
    print("-"*60)

    model2 = SimpleSegmentationModel(d_model=256, nhead=8)
    print("Before LoRA:")
    print_trainable_parameters(model2)

    model2 = inject_lora_into_model(model2, lora_config, verbose=False)

    print("\nAfter LoRA:")
    print_trainable_parameters(model2)

    # Test forward pass
    x = torch.randn(batch_size, seq_len, d_model)

    with torch.no_grad():
        output2 = model2(x)

    print(f"\n✓ Forward pass successful!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output2.shape}")

    # Test 3: Backward pass
    print("\n\nTest 3: Backward Pass (Training)")
    print("-"*60)

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )

    output = model(src, tgt)
    loss = output.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"✓ Backward pass successful!")
    print(f"  Loss: {loss.item():.4f}")

    # Count gradients
    lora_params_with_grad = sum(
        1 for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )
    print(f"  LoRA params with gradients: {lora_params_with_grad}")

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    print("\nThe standalone package works correctly without SAM3!")


if __name__ == "__main__":
    test_lora_injection()
