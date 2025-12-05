# SAM3 LoRA Implementation Guide

This repository provides LoRA (Low-Rank Adaptation) fine-tuning for SAM3 (Segment Anything Model 3) for efficient adaptation to custom segmentation tasks.

## Features

- **LoRA Implementation**: Efficient fine-tuning using low-rank adaptation
- **Flexible Configuration**: YAML-based configuration for easy experimentation
- **Configurable Target Modules**: Apply LoRA to specific transformer components:
  - Query/Key/Value projections in attention
  - Output projections in attention
  - Feed-forward network layers
- **Compatible with SAM3 Training Pipeline**: Follows the same training procedure as `sam3/train`
- **Memory Efficient**: Only trains a small fraction of parameters

## Directory Structure

```
sam3_lora/
├── data/                          # Training data in COCO format
│   ├── train/                     # Training images and annotations
│   ├── valid/                     # Validation images and annotations
│   └── test/                      # Test images and annotations
├── src/
│   ├── lora/                      # LoRA implementation
│   │   ├── lora_layer.py         # LoRA layer definitions
│   │   └── lora_utils.py         # Utilities for LoRA injection
│   ├── data/                      # Data loading utilities
│   │   └── dataset.py            # Dataset and DataLoader
│   ├── train/                     # Training logic
│   │   └── train_lora.py         # LoRA trainer
│   └── configs/                   # Configuration files
│       └── lora_config_example.yaml  # Example config
├── train.py                       # Main training script
└── LORA_IMPLEMENTATION_GUIDE.md   # This file
```

## Installation

1. Install SAM3 dependencies:
```bash
cd /workspace/sam3
pip install -e .
```

2. Install additional requirements:
```bash
pip install tensorboard pyyaml
```

## Data Format

The training data should be in COCO format with the following structure:

```
data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── _annotations.coco.json
└── valid/
    ├── image1.jpg
    ├── image2.jpg
    └── _annotations.coco.json
```

The annotation file should follow the COCO JSON format:
```json
{
  "images": [
    {
      "image_id": 1,
      "file_name": "image1.jpg",
      "height": 480,
      "width": 640
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "bbox": [x, y, width, height],
      "area": 12345,
      "segmentation": {...}
    }
  ]
}
```

## Configuration

Edit `src/configs/lora_config_example.yaml` to customize your training:

### Key Configuration Options

#### LoRA Parameters
```yaml
lora:
  rank: 8                    # Rank of LoRA matrices (higher = more capacity)
  alpha: 16.0                # LoRA scaling factor (typically 2*rank)
  dropout: 0.1               # Dropout for LoRA layers

  # Target modules to apply LoRA
  target_modules:
    - q_proj                 # Query projection in attention
    - k_proj                 # Key projection in attention
    - v_proj                 # Value projection in attention
    - out_proj               # Output projection in attention
    - linear1                # First FFN layer
    - linear2                # Second FFN layer
```

**LoRA Configuration Tips:**
- **rank**: Start with 4-8. Higher values (16-32) give more capacity but use more memory
- **alpha**: Typically set to 2*rank or 1*rank. Controls the magnitude of LoRA updates
- **target_modules**:
  - Use `["q_proj", "v_proj"]` for minimal training (fastest, least memory)
  - Use `["q_proj", "k_proj", "v_proj", "out_proj"]` for attention-only
  - Use all modules for maximum adaptation capacity

#### Dataset Configuration
```yaml
dataset:
  train_img_folder: /workspace/sam3_lora/data/train
  train_ann_file: /workspace/sam3_lora/data/train/_annotations.coco.json
  val_img_folder: /workspace/sam3_lora/data/valid
  val_ann_file: /workspace/sam3_lora/data/valid/_annotations.coco.json
  resolution: 1008           # Input resolution
  max_ann_per_img: 200       # Maximum annotations per image
```

#### Training Configuration
```yaml
training:
  max_epochs: 20
  batch_size: 2
  learning_rate: 1e-4
  gradient_accumulation_steps: 1
  use_amp: true              # Use automatic mixed precision
  amp_dtype: bfloat16        # bfloat16 or float16
```

## Usage

### Basic Training

```bash
python train.py --config src/configs/lora_config_example.yaml
```

### Resume Training

```bash
python train.py --config src/configs/lora_config_example.yaml --resume experiments/checkpoints/best.pt
```

### Advanced Usage

#### Custom LoRA Configuration

To apply LoRA only to attention layers:

```yaml
lora:
  rank: 8
  alpha: 16.0
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - out_proj
```

To apply LoRA to all transformer components:

```yaml
lora:
  rank: 16
  alpha: 32.0
  target_modules:
    - all
```

#### Training with Segmentation Masks

```yaml
training:
  enable_segmentation: true  # Enable mask prediction
```

## Expected Results

With LoRA fine-tuning:
- **Trainable Parameters**: ~1-5% of total model parameters
- **Memory Usage**: Significantly reduced compared to full fine-tuning
- **Training Speed**: Faster than full fine-tuning
- **Performance**: Comparable to full fine-tuning on domain-specific tasks

## LoRA Parameters Explanation

### What is LoRA?

LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to frozen pretrained weights:

```
W' = W + B×A
```

Where:
- `W` is the frozen pretrained weight (d_out × d_in)
- `A` is trainable (rank × d_in)
- `B` is trainable (d_out × rank)
- `rank` << min(d_out, d_in)

### Key Hyperparameters

1. **Rank (`r`)**:
   - Determines the bottleneck dimension
   - Lower rank = fewer parameters, less capacity
   - Typical values: 4, 8, 16, 32
   - Start with 8 for most tasks

2. **Alpha (`α`)**:
   - Scaling factor for LoRA updates
   - Controls how much the LoRA adaptation affects the output
   - Common practice: α = 2r or α = r
   - Higher alpha = stronger adaptation

3. **Target Modules**:
   - **Attention Projections**: Most important for adaptation
     - `q_proj`: Query projection
     - `k_proj`: Key projection
     - `v_proj`: Value projection
     - `out_proj`: Output projection
   - **FFN Layers**: Additional capacity
     - `linear1`: First FFN layer
     - `linear2`: Second FFN layer

## Monitoring Training

Training logs and metrics are saved to the experiment directory:

```
experiments/
├── logs/
│   ├── training.log           # Training logs
│   └── events.out.tfevents.*  # Tensorboard events
└── checkpoints/
    ├── best.pt                # Best model checkpoint
    └── epoch_*.pt             # Periodic checkpoints
```

View training progress with Tensorboard:

```bash
tensorboard --logdir experiments/logs
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `rank` in LoRA config
- Enable gradient checkpointing (if supported)
- Use smaller `resolution`

### Slow Training
- Increase `batch_size` (if memory allows)
- Use `gradient_accumulation_steps` for effective larger batch size
- Reduce `num_workers` if CPU is bottleneck

### Poor Performance
- Increase LoRA `rank` for more capacity
- Increase `alpha` for stronger adaptation
- Add more target modules
- Train for more epochs
- Check data quality and annotations

## Citation

If you use this code, please cite:

```bibtex
@article{sam3,
  title={SAM3: Segment Anything Model 3},
  author={...},
  journal={arXiv preprint},
  year={2024}
}

@article{lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```

## License

This project follows the same license as SAM3.
