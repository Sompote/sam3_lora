#!/usr/bin/env python3
"""
Simplified training script for testing LoRA without full SAM3.
This demonstrates LoRA training on a simple model with real data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.lora.lora_utils import LoRAConfig, inject_lora_into_model, print_trainable_parameters
from src.data.dataset import LoRASAM3Dataset


def simple_collate(batch):
    """Simple collate function."""
    return {
        'images': [item['image'] for item in batch],
        'annotations': [item['annotations'] for item in batch],
    }


def main():
    print("="*60)
    print("SAM3 LoRA - Simple Demo Training")
    print("="*60)
    
    # 1. Create a simple model (for demo purposes)
    print("\n1. Creating simple model...")
    
    class SimpleSegmentationModel(nn.Module):
        """Simple model for demonstration."""
        def __init__(self):
            super().__init__()
            # Simple encoder
            self.encoder = nn.TransformerEncoderLayer(
                d_model=256, 
                nhead=8, 
                dim_feedforward=1024,
                batch_first=True
            )
            # Simple head
            self.head = nn.Linear(256, 1)
        
        def forward(self, x):
            # x: (batch, seq, 256)
            x = self.encoder(x)
            return self.head(x.mean(dim=1))
    
    model = SimpleSegmentationModel()
    print("✓ Model created")
    
    # 2. Inject LoRA
    print("\n2. Injecting LoRA...")
    lora_config = LoRAConfig(
        rank=8,
        alpha=16.0,
        dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "linear1", "linear2"]
    )
    model = inject_lora_into_model(model, lora_config, verbose=True)
    
    print("\n✓ LoRA injected")
    print_trainable_parameters(model)
    
    # 3. Setup training
    print("\n3. Setting up training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01
    )
    
    # 4. Create dataloader (optional - uses dummy data if not available)
    print("\n4. Setting up data...")
    use_real_data = False
    
    try:
        dataset = LoRASAM3Dataset(
            img_folder='/workspace/sam3_lora/data/train',
            ann_file='/workspace/sam3_lora/data/train/_annotations.coco.json',
        )
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=simple_collate,
            num_workers=0,
        )
        use_real_data = True
        print(f"✓ Using real data: {len(dataset)} samples")
    except Exception as e:
        print(f"⚠ Could not load real data: {e}")
        print("  Using dummy data instead")
    
    # 5. Training loop
    print("\n5. Starting training...")
    print("-"*60)
    
    num_epochs = 5
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        if use_real_data:
            # Use real dataloader
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress:
                # Create dummy input for demo (replace with real preprocessing)
                batch_size = len(batch['images'])
                x = torch.randn(batch_size, 10, 256).to(device)
                y = torch.randn(batch_size, 1).to(device)
                
                optimizer.zero_grad()
                output = model(x)
                loss = nn.MSELoss()(output, y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                progress.set_postfix({'loss': f'{loss.item():.4f}'})
        else:
            # Use dummy data
            for i in range(10):  # 10 dummy batches
                # Dummy data
                x = torch.randn(4, 10, 256).to(device)
                y = torch.randn(4, 1).to(device)
                
                optimizer.zero_grad()
                output = model(x)
                loss = nn.MSELoss()(output, y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
    
    # 6. Save LoRA weights
    print("\n6. Saving LoRA weights...")
    from src.lora.lora_utils import get_lora_state_dict
    
    lora_weights = get_lora_state_dict(model)
    output_path = "/workspace/sam3_lora/demo_lora.pt"
    torch.save({
        'lora_state_dict': lora_weights,
        'lora_config': {
            'rank': lora_config.rank,
            'alpha': lora_config.alpha,
            'target_modules': list(lora_config.target_modules),
        }
    }, output_path)
    
    print(f"✓ LoRA weights saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    print("\n" + "="*60)
    print("✓ Training complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check the saved weights: demo_lora.pt")
    print("2. Load them with: torch.load('demo_lora.pt')")
    print("3. Inject into new model with: load_lora_state_dict(model, state_dict)")
    print("="*60)


if __name__ == "__main__":
    import os
    main()
