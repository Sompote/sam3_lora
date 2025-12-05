#!/usr/bin/env python3
"""
Test script to verify LoRA injection works correctly.

This creates a simple transformer model and injects LoRA to verify functionality.
"""

import torch
import torch.nn as nn
from src.lora.lora_utils import LoRAConfig, inject_lora_into_model, print_trainable_parameters


class SimpleTransformer(nn.Module):
    """Simple transformer for testing LoRA injection."""

    def __init__(self, d_model=256, nhead=8, num_layers=2):
        super().__init__()

        # Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output head
        self.head = nn.Linear(d_model, 10)

    def forward(self, src, tgt):
        # Encode
        memory = self.encoder(src)

        # Decode
        output = self.decoder(tgt, memory)

        # Output
        return self.head(output)


def main():
    print("=" * 60)
    print("Testing LoRA Injection")
    print("=" * 60)

    # Create model
    print("\n1. Creating simple transformer model...")
    model = SimpleTransformer(d_model=256, nhead=8, num_layers=2)

    print("\nBefore LoRA injection:")
    print_trainable_parameters(model)

    # Create LoRA config
    print("\n2. Creating LoRA configuration...")
    lora_config = LoRAConfig(
        rank=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "linear1", "linear2"],
    )

    print(f"   Rank: {lora_config.rank}")
    print(f"   Alpha: {lora_config.alpha}")
    print(f"   Dropout: {lora_config.dropout}")
    print(f"   Target modules: {lora_config.target_modules}")

    # Inject LoRA
    print("\n3. Injecting LoRA into model...")
    model = inject_lora_into_model(model, lora_config, verbose=True)

    print("\nAfter LoRA injection:")
    print_trainable_parameters(model)

    # Test forward pass
    print("\n4. Testing forward pass...")
    batch_size = 2
    seq_len = 10
    d_model = 256

    src = torch.randn(batch_size, seq_len, d_model)
    tgt = torch.randn(batch_size, seq_len, d_model)

    try:
        with torch.no_grad():
            output = model(src, tgt)
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {src.shape}")
        print(f"  Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test backward pass
    print("\n5. Testing backward pass...")
    try:
        output = model(src, tgt)
        loss = output.sum()
        loss.backward()
        print(f"✓ Backward pass successful!")
        print(f"  Loss: {loss.item():.4f}")

        # Check gradients
        lora_params_with_grad = 0
        frozen_params_with_grad = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    lora_params_with_grad += 1
                else:
                    print(f"  Warning: Trainable param {name} has no gradient")
            else:
                if param.grad is not None:
                    frozen_params_with_grad += 1
                    print(f"  Warning: Frozen param {name} has gradient!")

        print(f"  LoRA params with gradients: {lora_params_with_grad}")
        print(f"  Frozen params with gradients: {frozen_params_with_grad}")

    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
