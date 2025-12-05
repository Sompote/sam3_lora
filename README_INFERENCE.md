# SAM3 + LoRA Inference Guide

## Overview

This project provides two inference scripts for SAM3 with LoRA fine-tuning:

1. **`infer_sam.py`** - New script based on official SAM3 patterns (recommended)
2. **`inference_lora.py`** - Original simple inference script

## `infer_sam.py` (Recommended)

Based on the official SAM3 batched inference patterns with several improvements:

### Features
- ✅ Multiple text prompts support
- ✅ Official SAM3 data structures and transforms
- ✅ Transparent manual post-processing
- ✅ Proper coordinate handling (boxes clamped to image bounds)
- ✅ Color-coded visualization for multiple prompts
- ✅ Auto-detection of best weights

### Usage

#### Single Prompt
```bash
python3 infer_sam.py \
    --config configs/full_lora_config.yaml \
    --image path/to/image.jpg \
    --prompt "crack" \
    --output output.png \
    --threshold 0.3
```

#### Multiple Prompts
```bash
python3 infer_sam.py \
    --config configs/full_lora_config.yaml \
    --image path/to/image.jpg \
    --prompt "crack" "defect" "damage" \
    --output output.png \
    --threshold 0.3
```

### Arguments

- `--config`: Path to training config YAML (required)
- `--weights`: Path to LoRA weights (optional, auto-detected from config)
- `--image`: Path to input image (required)
- `--prompt`: One or more text prompts (default: "object")
- `--output`: Output visualization path (default: "output.png")
- `--threshold`: Detection confidence threshold (default: 0.5)
- `--resolution`: Input resolution (default: 1008)
- `--no-boxes`: Don't show bounding boxes
- `--no-masks`: Don't show segmentation masks

### Examples

**Crack detection with low threshold:**
```bash
python3 infer_sam.py \
    --config configs/full_lora_config.yaml \
    --image data/test/crack_image.jpg \
    --prompt "crack" \
    --threshold 0.3 \
    --output crack_detection.png
```

**Multiple defect types:**
```bash
python3 infer_sam.py \
    --config configs/full_lora_config.yaml \
    --image data/test/defect_image.jpg \
    --prompt "crack" "spalling" "corrosion" \
    --threshold 0.4 \
    --output defect_analysis.png
```

**Mask-only visualization:**
```bash
python3 infer_sam.py \
    --config configs/full_lora_config.yaml \
    --image data/test/image.jpg \
    --prompt "crack" \
    --no-boxes \
    --output mask_only.png
```

## `inference_lora.py` (Legacy)

Simple inference script with basic functionality.

### Usage
```bash
python3 inference_lora.py \
    --config configs/full_lora_config.yaml \
    --weights outputs/sam3_lora_full/best_lora_weights.pt \
    --image path/to/image.jpg \
    --prompt "crack" \
    --output output.png \
    --threshold 0.5
```

## Output Format

Both scripts generate:
1. **Visualization image**: Shows detected objects with:
   - Bounding boxes (colored by prompt)
   - Segmentation masks (semi-transparent overlay)
   - Confidence scores
   - Prompt labels

2. **Console summary**:
   ```
   📊 Summary:
      Prompt 'crack': 1 detections
         Max confidence: 0.320
      Prompt 'damage': 3 detections
         Max confidence: 0.401
   ```

## Tips

1. **Threshold Selection**:
   - Lower threshold (0.3): More detections, may include false positives
   - Higher threshold (0.6): Fewer detections, higher precision
   - Default (0.5): Balanced approach

2. **Prompt Engineering**:
   - Be specific: "concrete crack" vs "crack"
   - Try variations: "defect", "damage", "deterioration"
   - Multiple prompts can catch different aspects

3. **Performance**:
   - First inference is slow due to model compilation
   - Subsequent inferences are much faster
   - Resolution affects both quality and speed

## Troubleshooting

**Issue**: No detections found
- Try lowering the threshold (--threshold 0.3)
- Try different prompt variations
- Check if the LoRA weights are properly loaded

**Issue**: Boxes extending outside image
- This is fixed in `infer_sam.py` (boxes are automatically clamped)
- Update to the latest version if using `inference_lora.py`

**Issue**: Out of memory
- Reduce resolution (--resolution 512)
- Use CPU instead of GPU (modify device in code)

## Model Weights

The script auto-detects the best LoRA weights from the config's output directory:
- Default location: `outputs/sam3_lora_full/best_lora_weights.pt`
- Override with `--weights` argument if needed

## Architecture

`infer_sam.py` follows the official SAM3 inference pipeline:

1. **Image Loading**: PIL Image → SAM3 Datapoint
2. **Transforms**: Resize (1008x1008) → Normalize
3. **Batching**: Collate with official collator
4. **Inference**: Forward pass through SAM3 + LoRA
5. **Post-processing**: Manual (transparent and controllable)
6. **Visualization**: Matplotlib with color-coded overlays
