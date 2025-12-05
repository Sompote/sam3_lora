# SAM3 LoRA File Structure

## Complete File Tree

```
sam3_lora/
├── src/
│   ├── __init__.py                           # Package initialization
│   │
│   ├── lora/                                 # LoRA implementation
│   │   ├── __init__.py                       # LoRA package exports
│   │   ├── lora_layer.py                     # Core LoRA layer classes
│   │   └── lora_utils.py                     # LoRA injection utilities
│   │
│   ├── data/                                 # Data loading
│   │   ├── __init__.py                       # Data package exports  
│   │   └── dataset.py                        # COCO dataset & dataloaders
│   │
│   ├── train/                                # Training logic
│   │   ├── __init__.py                       # Training package exports
│   │   └── train_lora.py                     # LoRA trainer class
│   │
│   └── configs/                              # Configuration files
│       └── lora_config_example.yaml          # Example training config
│
├── data/                                     # Training data (user-provided)
│   ├── train/                                # Training images & annotations
│   │   ├── *.jpg/*.png                       # Training images
│   │   └── _annotations.coco.json            # COCO format annotations
│   ├── valid/                                # Validation images & annotations
│   │   ├── *.jpg/*.png                       # Validation images
│   │   └── _annotations.coco.json            # COCO format annotations
│   └── test/                                 # Test images (optional)
│       ├── *.jpg/*.png                       # Test images
│       └── _annotations.coco.json            # COCO format annotations
│
├── experiments/                              # Training outputs (auto-created)
│   ├── logs/                                 # Training logs
│   │   ├── training.log                      # Text logs
│   │   └── events.out.tfevents.*             # Tensorboard events
│   └── checkpoints/                          # Model checkpoints
│       ├── best.pt                           # Best model (LoRA weights)
│       └── epoch_*.pt                        # Periodic checkpoints
│
├── train.py                                  # Main training script
├── quick_start.sh                            # Quick start helper script
├── LORA_IMPLEMENTATION_GUIDE.md              # Implementation guide
├── IMPLEMENTATION_SUMMARY.md                 # Technical summary
├── FILE_STRUCTURE.md                         # This file
└── README.md                                 # Main README (existing)
```

## File Descriptions

### Core Implementation Files

#### `src/lora/lora_layer.py` (180 lines)
Implements the core LoRA layers:
- `LoRALayer`: Low-rank adaptation layer with A and B matrices
- `LinearWithLoRA`: Wrapper that adds LoRA to nn.Linear layers
- Weight initialization and merging functionality

#### `src/lora/lora_utils.py` (230 lines)
Utilities for LoRA injection and management:
- `LoRAConfig`: Configuration class for LoRA parameters
- `inject_lora_into_model()`: Injects LoRA into SAM3 model
- `get_lora_parameters()`: Extracts trainable LoRA parameters
- `get_lora_state_dict()`: Saves/loads LoRA weights
- `merge_lora_weights()`: Merges LoRA into base model
- `print_trainable_parameters()`: Parameter statistics

#### `src/data/dataset.py` (180 lines)
Data loading for COCO format:
- `LoRASAM3Dataset`: PyTorch dataset for COCO format data
- `collate_fn()`: Batching function
- `create_dataloaders()`: Creates train/val dataloaders

#### `src/train/train_lora.py` (250 lines)
Main training logic:
- `LoRATrainer`: Trainer class following SAM3's training procedure
- Training loop with AMP, gradient accumulation, clipping
- Validation loop
- Checkpoint saving/loading (LoRA weights only)
- Tensorboard logging

#### `src/configs/lora_config_example.yaml` (150 lines)
Complete training configuration:
- Paths configuration (data, experiments, BPE, checkpoints)
- LoRA hyperparameters (rank, alpha, dropout, target modules)
- Dataset configuration (folders, annotations, resolution)
- Training configuration (epochs, batch size, optimizer, scheduler)
- Checkpoint and logging configuration
- Distributed training settings

#### `train.py` (170 lines)
Main entry point:
- Loads YAML configuration
- Builds SAM3 model
- Injects LoRA adapters
- Creates dataloaders
- Initializes trainer and optimizer
- Starts training

### Documentation Files

#### `LORA_IMPLEMENTATION_GUIDE.md`
Comprehensive user guide:
- Installation instructions
- Data format requirements
- Configuration guide with examples
- Usage instructions
- LoRA parameter explanation
- Troubleshooting tips

#### `IMPLEMENTATION_SUMMARY.md`
Technical documentation:
- Architecture overview
- LoRA injection points
- Training workflow
- Integration with SAM3
- Example configurations

#### `FILE_STRUCTURE.md` (this file)
Complete file structure and descriptions

### Helper Scripts

#### `quick_start.sh`
Setup and validation script:
- Checks Python and CUDA availability
- Creates required directories
- Validates data presence
- Shows example training commands

## Usage Flow

1. **Setup**: Run `./quick_start.sh` to verify environment
2. **Configure**: Edit `src/configs/lora_config_example.yaml` with your paths
3. **Train**: Run `python train.py --config src/configs/lora_config_example.yaml`
4. **Monitor**: Use Tensorboard on `experiments/logs/`
5. **Deploy**: Load LoRA weights from `experiments/checkpoints/`

## Key Features by File

### Memory Efficiency (lora_layer.py)
- Low-rank matrices reduce parameters by 100-1000x
- Only train 1-5% of model parameters

### Flexibility (lora_utils.py)
- Configure which modules get LoRA
- Adjust rank for parameter/capacity tradeoff
- Support for weight merging or adapter deployment

### Data Compatibility (dataset.py)
- COCO format support
- Compatible with SAM3's data pipeline
- Handles images and segmentation masks

### Production Ready (train_lora.py)
- Follows SAM3's training structure
- AMP for faster training
- Gradient accumulation for larger effective batch size
- Robust checkpoint saving

## Integration Points with SAM3

The implementation integrates with SAM3 at these points:

1. **Model Building**: Uses `sam3.model_builder.build_sam3_image_model()`
2. **Loss Functions**: Compatible with `sam3.train.loss.*`
3. **Transforms**: Can use `sam3.train.transforms.*`
4. **Training Structure**: Follows `sam3.train.trainer.Trainer` pattern

## Customization Points

Users can customize:

1. **LoRA Configuration** (`lora_config_example.yaml`):
   - rank, alpha, dropout
   - target_modules list
   
2. **Training Parameters** (`lora_config_example.yaml`):
   - Batch size, learning rate
   - Epochs, gradient accumulation
   - AMP settings

3. **Data Loading** (`dataset.py`):
   - Add custom transforms
   - Modify data augmentation
   
4. **Training Logic** (`train_lora.py`):
   - Custom loss computation
   - Additional metrics
   - Custom callbacks

## Total Lines of Code

- Core Implementation: ~840 lines
- Configuration: ~150 lines  
- Main Script: ~170 lines
- Documentation: ~500 lines
- **Total: ~1660 lines**

Compact but complete implementation ready for production use!
