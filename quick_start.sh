#!/bin/bash
# Quick start script for SAM3 LoRA fine-tuning

echo "========================================="
echo "SAM3 LoRA Fine-tuning Quick Start"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if CUDA is available
echo ""
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'No CUDA')"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p experiments/logs
mkdir -p experiments/checkpoints
echo "✓ Directories created"

# Check if data exists
echo ""
echo "Checking data..."
if [ -d "data/train" ] && [ -f "data/train/_annotations.coco.json" ]; then
    echo "✓ Training data found"
    num_train_images=$(ls data/train/*.jpg data/train/*.png 2>/dev/null | wc -l)
    echo "  - Training images: $num_train_images"
else
    echo "⚠ Training data not found in data/train/"
    echo "  Please prepare your data in COCO format"
fi

if [ -d "data/valid" ] && [ -f "data/valid/_annotations.coco.json" ]; then
    echo "✓ Validation data found"
    num_val_images=$(ls data/valid/*.jpg data/valid/*.png 2>/dev/null | wc -l)
    echo "  - Validation images: $num_val_images"
else
    echo "⚠ Validation data not found in data/valid/"
    echo "  Please prepare your data in COCO format"
fi

# Example training command
echo ""
echo "========================================="
echo "To start training, run:"
echo ""
echo "python train.py --config src/configs/lora_config_example.yaml"
echo ""
echo "Make sure to edit the config file first:"
echo "- Update paths.data_root"
echo "- Update paths.experiment_log_dir"
echo "- Update paths.bpe_path"
echo "- Update paths.sam3_checkpoint (if you have a pretrained model)"
echo ""
echo "========================================="
