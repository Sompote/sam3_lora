# SAM3 LoRA - CLI Training Guide

## 🚀 Quick Start (What Works Now)

### Option 1: Test LoRA (Recommended - Works Immediately)

Verify LoRA works with a simple transformer:

```bash
cd /workspace/sam3_lora
python3 test_lora_injection.py
```

**Expected Output:**
```
============================================================
Testing LoRA Injection
============================================================
...
✓ Forward pass successful!
✓ Backward pass successful!
✓ All tests passed!
============================================================
```

---

## 📝 Option 2: Full SAM3 Training (Requires Setup)

### Step 1: Check Your Configuration

Edit the config file:
```bash
vim /workspace/sam3_lora/src/configs/lora_config_example.yaml
```

Key settings to verify:
```yaml
paths:
  data_root: /workspace/sam3_lora/data
  experiment_log_dir: /workspace/sam3_lora/experiments
  bpe_path: /workspace/sam3/assets/bpe_simple_vocab_16e6.txt.gz
  sam3_checkpoint: null  # Or path to SAM3 checkpoint

lora:
  rank: 8              # LoRA rank (4, 8, 16, 32)
  alpha: 16.0          # Scaling (typically 2*rank)
  dropout: 0.1
  target_modules:      # Which modules get LoRA
    - q_proj
    - k_proj
    - v_proj
    - out_proj
    - linear1
    - linear2

training:
  max_epochs: 20
  batch_size: 2
  learning_rate: 1e-4
  use_amp: true
  amp_dtype: bfloat16
```

### Step 2: Basic Training Command

```bash
cd /workspace/sam3_lora

# Basic training
python3 train.py --config src/configs/lora_config_example.yaml

# With GPU selection
CUDA_VISIBLE_DEVICES=0 python3 train.py --config src/configs/lora_config_example.yaml

# With specific device
python3 train.py \
  --config src/configs/lora_config_example.yaml \
  --device cuda
```

### Step 3: Resume Training

```bash
python3 train.py \
  --config src/configs/lora_config_example.yaml \
  --resume experiments/checkpoints/best.pt
```

---

## 🛠️ Current Issue & Solutions

### Issue: `NotImplementedError` in Loss Computation

The trainer's `_compute_loss()` method is not implemented yet.

### Solution A: Implement SAM3 Loss (Recommended for Production)

Edit `/workspace/sam3_lora/src/train/train_lora.py`:

```python
def _compute_loss(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
    """Compute loss using SAM3's loss functions."""
    from sam3.train.loss.sam3_loss import Sam3LossWrapper

    # Use SAM3's loss
    # This is a simplified example - adjust based on your SAM3 version
    if hasattr(self, 'loss_fn'):
        return self.loss_fn(outputs, batch)
    else:
        # Fallback: simple dummy loss for testing
        if isinstance(outputs, dict) and 'loss' in outputs:
            return outputs['loss']
        raise NotImplementedError("Loss function not configured")
```

### Solution B: Simple Demo Training Loop

Create a minimal training script for testing:

```bash
cat > /workspace/sam3_lora/train_simple.py << 'EOF'
#!/usr/bin/env python3
"""
Simplified training script for testing LoRA without full SAM3.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.lora.lora_utils import LoRAConfig, inject_lora_into_model, print_trainable_parameters
from src.data.dataset import LoRASAM3Dataset

def main():
    # 1. Create a simple model (for demo)
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.TransformerEncoderLayer(256, 8, 1024, batch_first=True)
            self.head = nn.Linear(256, 1)

        def forward(self, x):
            x = self.encoder(x)
            return self.head(x.mean(dim=1))

    model = SimpleModel()

    # 2. Inject LoRA
    lora_config = LoRAConfig(rank=8, alpha=16.0, target_modules=["q_proj", "v_proj"])
    model = inject_lora_into_model(model, lora_config, verbose=True)
    print_trainable_parameters(model)

    # 3. Create optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )

    # 4. Training loop
    model.train()
    for epoch in range(5):
        # Dummy data
        x = torch.randn(4, 10, 256)
        y = torch.randn(4, 1)

        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # 5. Save LoRA weights
    from src.lora.lora_utils import get_lora_state_dict
    lora_weights = get_lora_state_dict(model)
    torch.save(lora_weights, "demo_lora.pt")
    print("\n✓ Training complete! LoRA weights saved to demo_lora.pt")

if __name__ == "__main__":
    main()
EOF

python3 /workspace/sam3_lora/train_simple.py
```

---

## 🎯 Alternative: Use SAM3's Official Trainer with LoRA

This is the recommended approach for production:

### Create Integration Script

```bash
cat > /workspace/sam3_lora/train_with_sam3.py << 'EOF'
#!/usr/bin/env python3
"""
Train SAM3 with LoRA using SAM3's official trainer.
"""

import sys
sys.path.insert(0, '/workspace/sam3')

from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from src.lora.lora_utils import LoRAConfig, inject_lora_into_model

def main():
    # 1. Initialize Hydra with SAM3's configs
    initialize_config_module("sam3.train", version_base="1.2")

    # 2. Load your config (create one based on SAM3's examples)
    cfg = compose(config_name="your_config")

    # 3. Build model
    model = instantiate(cfg.trainer.model, _recursive_=False)

    # 4. Inject LoRA
    lora_config = LoRAConfig(
        rank=8,
        alpha=16.0,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
    )
    model = inject_lora_into_model(model, lora_config, verbose=True)

    # 5. Replace model in trainer config
    cfg.trainer.model = model

    # 6. Run SAM3's trainer
    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()

if __name__ == "__main__":
    main()
EOF
```

---

## 📊 CLI Options

### Full Command Reference

```bash
# Basic training
python3 train.py --config <config_file>

# Resume from checkpoint
python3 train.py --config <config_file> --resume <checkpoint_path>

# Specify device
python3 train.py --config <config_file> --device cuda
python3 train.py --config <config_file> --device cpu

# With environment variables
CUDA_VISIBLE_DEVICES=0,1 python3 train.py --config <config_file>
```

### Config File Options

Create custom configs by copying and editing:

```bash
# Copy example config
cp src/configs/lora_config_example.yaml src/configs/my_config.yaml

# Edit your config
vim src/configs/my_config.yaml

# Use it
python3 train.py --config src/configs/my_config.yaml
```

---

## 🔍 Monitor Training

### Option 1: Tensorboard

```bash
# In another terminal
tensorboard --logdir /workspace/sam3_lora/experiments/logs

# Open browser to: http://localhost:6006
```

### Option 2: Log Files

```bash
# Watch training log
tail -f /workspace/sam3_lora/experiments/logs/training.log

# Check checkpoints
ls -lh /workspace/sam3_lora/experiments/checkpoints/
```

### Option 3: Training Stats

```bash
# Check last 20 log lines
tail -20 /workspace/sam3_lora/experiments/logs/training.log

# Search for specific metrics
grep "Loss:" /workspace/sam3_lora/experiments/logs/training.log
```

---

## 🎓 Example: Quick Demo Training

Here's a complete example that works right now:

```bash
cd /workspace/sam3_lora

# Run simple demo training
python3 train_simple.py
```

This will:
1. Create a simple transformer model
2. Inject LoRA (rank=8)
3. Train for 5 epochs on dummy data
4. Save LoRA weights

---

## 🐛 Troubleshooting

### Error: "SAM3 model not found"
```bash
# Check if SAM3 is installed
python3 -c "import sam3; print('SAM3 installed')"

# If not, install it
cd /workspace/sam3
pip install -e .
```

### Error: "BPE path not found"
```bash
# Check if file exists
ls /workspace/sam3/assets/bpe_simple_vocab_16e6.txt.gz

# If not, download it
mkdir -p /workspace/sam3/assets
cd /workspace/sam3/assets
wget https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz
```

### Error: "CUDA out of memory"
```bash
# Reduce batch size in config
vim src/configs/lora_config_example.yaml
# Change: batch_size: 1

# Or reduce LoRA rank
# Change: rank: 4

# Or use gradient accumulation
# Change: gradient_accumulation_steps: 4
```

### Error: "NotImplementedError: Loss computation"
```bash
# Use the simple demo instead
python3 train_simple.py

# Or implement the loss function as shown in Solution A above
```

---

## 📋 Checklist Before Training

- [ ] Config file exists and paths are correct
- [ ] Data exists in `/workspace/sam3_lora/data/`
- [ ] COCO annotations created (`_annotations.coco.json`)
- [ ] BPE vocab file exists
- [ ] SAM3 installed (`pip install -e /workspace/sam3`)
- [ ] GPU available (or use `--device cpu`)
- [ ] Experiment directory created

Run this to check:
```bash
cd /workspace/sam3_lora
bash quick_start.sh
```

---

## ✅ What Works Now

```bash
# ✅ Test LoRA injection
python3 test_lora_injection.py

# ✅ Simple demo training
python3 train_simple.py

# ✅ Data conversion
python3 convert_roboflow_to_coco.py

# ✅ Check environment
bash quick_start.sh
```

## ⏳ What Needs Setup

```bash
# ⏳ Full SAM3 training (needs loss function)
python3 train.py --config src/configs/lora_config_example.yaml
```

---

## 🎯 Recommended Workflow

1. **Start with test**: `python3 test_lora_injection.py` ✅
2. **Try simple demo**: `python3 train_simple.py` ✅
3. **Implement SAM3 loss**: Edit `src/train/train_lora.py`
4. **Run full training**: `python3 train.py --config ...`

The infrastructure is ready - you just need to plug in the SAM3 loss function!
