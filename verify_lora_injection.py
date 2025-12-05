
import torch
import torch.nn as nn
from sam3_lora import LoRAConfig, inject_lora_into_model
from sam3_lora.model.simple_models import SimpleSegmentationModel

def verify_injection():
    print("Verifying LoRA injection on SimpleSegmentationModel...")
    
    # 1. Create model
    model = SimpleSegmentationModel(d_model=256, nhead=8, dim_feedforward=1024)
    
    # 2. Define Config with all targets
    config = LoRAConfig(
        rank=4,
        alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "linear1", "linear2"]
    )
    
    print(f"Target modules: {config.target_modules}")
    
    # 3. Inject
    model = inject_lora_into_model(model, config, verbose=False)
    
    # 4. Inspect
    print("\nChecking modules:")
    injected_modules = []
    missed_linear_modules = []
    
    for name, module in model.named_modules():
        if "lora" in name: # Skip the internal lora modules themselves
            continue
            
        if hasattr(module, "lora"): # It's a wrapper (LinearWithLoRA)
            print(f" [x] Injected: {name}")
            injected_modules.append(name)
        elif isinstance(module, nn.Linear):
            print(f" [ ] MISSED:   {name}")
            missed_linear_modules.append(name)
        elif isinstance(module, nn.MultiheadAttention):
             print(f" [?] MultiheadAttention found at: {name}")
             # Check internal parameter names
             params = dict(module.named_parameters())
             if 'in_proj_weight' in params:
                 print(f"     -> Uses in_proj_weight (fused QKV). LoRA cannot inject into fused QKV with current utility.")
    
    print("\nSummary:")
    print(f"Total injected: {len(injected_modules)}")
    print(f"Total missed Linears: {len(missed_linear_modules)}")
    
    # Check specific targets
    print("\nTarget verification:")
    targets = ["q_proj", "k_proj", "v_proj", "out_proj", "linear1", "linear2"]
    
    # Identify what matches what
    # In SimpleSegmentationModel (TransformerEncoderLayer):
    # encoder.self_attn.out_proj -> out_proj
    # encoder.linear1 -> linear1
    # encoder.linear2 -> linear2
    # encoder.self_attn.in_proj_weight -> Q, K, V (Fused)
    
    # Head is just 'head'
    
    for t in targets:
        found = False
        for name in injected_modules:
            if t in name or name.endswith(t):
                found = True
                break
        print(f"Target '{t}': {'FOUND' if found else 'MISSING (Likely fused or named differently)'}")

if __name__ == "__main__":
    verify_injection()
