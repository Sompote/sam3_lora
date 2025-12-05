#!/usr/bin/env python3
"""Test script to check if auxiliary outputs are generated in training mode."""

import torch
from train_sam3_lora_native import SAM3TrainerNative
from sam3.model.model_misc import SAM3Output

print("="*60)
print("Testing Auxiliary Outputs Generation")
print("="*60)

# Initialize trainer
trainer = SAM3TrainerNative('configs/full_lora_config.yaml')

# Create a small batch
from torch.utils.data import DataLoader
from train_sam3_lora_native import SimpleSAM3Dataset
from sam3.train.data.collator import collate_fn_api

train_ds = SimpleSAM3Dataset("data", image_set="train")
train_loader = DataLoader(
    train_ds,
    batch_size=1,
    collate_fn=lambda x: collate_fn_api(x, dict_key="input", with_seg_masks=True),
    shuffle=False,
    num_workers=0,
)

# Get one batch
batch_dict = next(iter(train_loader))
input_batch = batch_dict["input"]

# Move to device
def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            val = getattr(obj, field)
            setattr(obj, field, move_to_device(val, device))
        return obj
    return obj

input_batch = move_to_device(input_batch, trainer.device)

# Test in TRAINING mode
print("\n1. Testing in TRAINING mode (model.train())")
print("-" * 60)
trainer.model.train()
print(f"Model training state: {trainer.model.training}")

with torch.no_grad():
    outputs_list = trainer.model(input_batch)

# Check outputs with iteration mode
with SAM3Output.iteration_mode(
    outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
) as outputs_iter:
    for stage_idx, stage_outputs in enumerate(outputs_iter):
        print(f"\nStage {stage_idx}:")
        for step_idx, outputs in enumerate(stage_outputs):
            print(f"  Step {step_idx}:")
            print(f"    Has pred_logits: {outputs.get('pred_logits') is not None}")
            print(f"    Has pred_boxes: {outputs.get('pred_boxes') is not None}")
            print(f"    Has pred_masks: {outputs.get('pred_masks') is not None}")
            print(f"    Has presence_logit_dec: {outputs.get('presence_logit_dec') is not None}")
            print(f"    Has aux_outputs: {outputs.get('aux_outputs') is not None}")

            if outputs.get('aux_outputs'):
                print(f"    Number of aux_outputs: {len(outputs['aux_outputs'])}")
                if len(outputs['aux_outputs']) > 0:
                    print(f"    Aux output keys: {outputs['aux_outputs'][0].keys()}")

# Test in EVAL mode
print("\n" + "="*60)
print("2. Testing in EVAL mode (model.eval())")
print("-" * 60)
trainer.model.eval()
print(f"Model training state: {trainer.model.training}")

with torch.no_grad():
    outputs_list = trainer.model(input_batch)

# Check outputs with iteration mode
with SAM3Output.iteration_mode(
    outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
) as outputs_iter:
    for stage_idx, stage_outputs in enumerate(outputs_iter):
        print(f"\nStage {stage_idx}:")
        for step_idx, outputs in enumerate(stage_outputs):
            print(f"  Step {step_idx}:")
            print(f"    Has pred_logits: {outputs.get('pred_logits') is not None}")
            print(f"    Has pred_boxes: {outputs.get('pred_boxes') is not None}")
            print(f"    Has pred_masks: {outputs.get('pred_masks') is not None}")
            print(f"    Has presence_logit_dec: {outputs.get('presence_logit_dec') is not None}")
            print(f"    Has aux_outputs: {outputs.get('aux_outputs') is not None}")

            if outputs.get('aux_outputs'):
                print(f"    Number of aux_outputs: {len(outputs['aux_outputs'])}")
                if len(outputs['aux_outputs']) > 0:
                    print(f"    Aux output keys: {outputs['aux_outputs'][0].keys()}")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)
