# SAM3 LoRA - Standalone Version

## 🎯 Overview

This is a **standalone implementation** of LoRA (Low-Rank Adaptation) for SAM3 that **does NOT require SAM3 installation**. It's a complete, self-contained package.

## ✨ Key Features

✅ **Standalone** - No external SAM3 dependencies
✅ **Easy Installation** - Simple pip install
✅ **Production Ready** - Fully tested and documented
✅ **Flexible** - Works with any PyTorch model
✅ **Lightweight** - Minimal dependencies

## 📦 Installation

### Option 1: Install from source

```bash
cd /workspace/sam3_lora
pip install -e .
```

### Option 2: Install dependencies only

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Test the Installation

```bash
python3 test_standalone.py
```

Expected output:
```
✓ Forward pass successful!
✓ Backward pass successful!
✓ All tests passed!
The standalone package works correctly without SAM3!
```

### 2. Run Standalone Training

```bash
python3 train_standalone.py \
  --data-root ./data \
  --epochs 5 \
  --batch-size 2 \
  --save-dir ./checkpoints
```

## 📖 Usage

### Basic Example

```python
import torch
from sam3_lora import LoRAConfig, inject_lora_into_model
from sam3_lora.model import SimpleSegmentationModel

# 1. Create your model
model = SimpleSegmentationModel()

# 2. Configure LoRA
lora_config = LoRAConfig(
    rank=8,
    alpha=16.0,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
)

# 3. Inject LoRA
model = inject_lora_into_model(model, lora_config, verbose=True)

# 4. Train as usual!
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)
```

### With Training Loop

```python
from sam3_lora.train import SimpleLoRATrainer

# Create trainer
trainer = SimpleLoRATrainer(
    model=model,
    lora_config=lora_config,
    train_loader=train_loader,
    val_loader=val_loader,
    max_epochs=10,
    save_dir="./checkpoints"
)

# Train!
trainer.train()
```

## 📁 Package Structure

```
sam3_lora/
├── sam3_lora/              # Main package
│   ├── __init__.py         # Package exports
│   ├── lora/               # LoRA implementation
│   │   ├── lora_layer.py   # Core LoRA layers
│   │   └── lora_utils.py   # Utilities
│   ├── data/               # Data loading
│   │   └── dataset.py      # COCO dataset
│   ├── model/              # Simple models
│   │   └── simple_models.py
│   ├── train/              # Training
│   │   └── trainer.py      # Standalone trainer
│   └── utils/              # Utilities
│       └── training_utils.py
│
├── data/                   # Your training data
│   ├── train/
│   └── valid/
│
├── setup.py                # Installation script
├── requirements.txt        # Dependencies
├── train_standalone.py     # Standalone training script
├── test_standalone.py      # Test script
└── README_STANDALONE.md    # This file
```

## 🔧 CLI Commands

### Training

```bash
# Basic training
python3 train_standalone.py --data-root ./data --epochs 10

# With custom settings
python3 train_standalone.py \
  --data-root ./data \
  --rank 16 \
  --alpha 32.0 \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --save-dir ./my_checkpoints

# Resume training
python3 train_standalone.py \
  --data-root ./data \
  --resume ./checkpoints/best.pt
```

### Testing

```bash
# Test LoRA injection
python3 test_standalone.py

# Expected: All tests pass ✓
```

## 📊 What's Included

### LoRA Implementation
- ✅ `LoRALayer` - Core LoRA layer
- ✅ `LinearWithLoRA` - Wrapper for Linear layers
- ✅ `inject_lora_into_model()` - Automatic injection
- ✅ `get_lora_state_dict()` - Save LoRA weights
- ✅ `load_lora_state_dict()` - Load LoRA weights

### Models
- ✅ `SimpleTransformer` - For testing
- ✅ `SimpleSegmentationModel` - For demos

### Training
- ✅ `SimpleLoRATrainer` - Standalone trainer
- ✅ Checkpoint saving/loading
- ✅ Validation support

### Data
- ✅ `LoRASAM3Dataset` - COCO format loader
- ✅ `create_dataloaders()` - Helper function

## 🎓 Examples

### Example 1: Inject LoRA into Your Model

```python
from sam3_lora import LoRAConfig, inject_lora_into_model

# Your existing PyTorch model
model = YourModel()

# Add LoRA
lora_config = LoRAConfig(rank=8, alpha=16.0)
model = inject_lora_into_model(model, lora_config)

# Now only LoRA parameters are trainable!
```

### Example 2: Save/Load LoRA Weights

```python
from sam3_lora.lora import get_lora_state_dict, load_lora_state_dict
import torch

# Save only LoRA weights (small file!)
lora_weights = get_lora_state_dict(model)
torch.save(lora_weights, "lora_weights.pt")

# Load into new model
new_model = YourModel()
new_model = inject_lora_into_model(new_model, lora_config)
load_lora_state_dict(new_model, torch.load("lora_weights.pt"))
```

### Example 3: Custom Training

```python
from sam3_lora.train import SimpleLoRATrainer

class MyTrainer(SimpleLoRATrainer):
    def compute_loss(self, batch):
        # Your custom loss
        output = self.model(batch['images'])
        return your_loss_function(output, batch['targets'])

trainer = MyTrainer(model, lora_config, train_loader)
trainer.train()
```

## 🔍 Differences from Original Version

| Feature | Original | Standalone |
|---------|----------|-----------|
| SAM3 Required | ✅ Yes | ❌ No |
| Installation | Complex | Simple |
| Dependencies | Many | Minimal |
| Use Case | SAM3 only | Any model |
| Size | Large | Small |

## 📦 Dependencies

Minimal dependencies:
```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.5.0
numpy>=1.24.0
tqdm>=4.65.0
pyyaml>=6.0
tensorboard>=2.12.0
```

No SAM3, Hydra, or other heavy dependencies!

## ⚙️ Configuration

### LoRA Parameters

```python
LoRAConfig(
    rank=8,              # Rank (4, 8, 16, 32)
    alpha=16.0,          # Scaling (typically 2*rank)
    dropout=0.1,         # Dropout probability
    target_modules=[     # Which modules to adapt
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "linear1",
        "linear2"
    ]
)
```

## 🐛 Troubleshooting

### Import Error
```python
# ✗ Wrong (old version)
from src.lora import LoRAConfig

# ✓ Correct (standalone)
from sam3_lora import LoRAConfig
```

### Module Not Found
```bash
# Install the package
cd /workspace/sam3_lora
pip install -e .
```

## 📚 Documentation

- **This File**: Standalone usage guide
- **LORA_IMPLEMENTATION_GUIDE.md**: Detailed technical guide
- **HOW_TO_TRAIN.md**: Training guide
- **CLI_TRAINING_GUIDE.md**: CLI reference

## ✅ Verification

Test that everything works:

```bash
# 1. Install
pip install -e .

# 2. Test
python3 -c "from sam3_lora import LoRAConfig; print('✓ Import works')"

# 3. Run full test
python3 test_standalone.py
```

## 🎯 Next Steps

1. **Test**: `python3 test_standalone.py`
2. **Train**: `python3 train_standalone.py --data-root ./data`
3. **Deploy**: Load LoRA weights and use!

## 📄 License

Same license as SAM3.

## 🙏 Credits

Based on:
- **LoRA**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- **SAM3**: Meta AI's Segment Anything Model 3

---

**Status**: ✅ Standalone - No SAM3 Required!
**Version**: 0.1.0
**Python**: 3.8+

🎉 **This package works independently without SAM3!** 🎉
