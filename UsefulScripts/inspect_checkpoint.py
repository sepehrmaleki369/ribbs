#!/usr/bin/env python3
"""
Script to inspect the contents of a PyTorch Lightning checkpoint
"""

import torch
import sys
import os
from pprint import pprint

def inspect_checkpoint(checkpoint_path):
    """Inspect the contents of a checkpoint file"""
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    print(f"Inspecting checkpoint: {checkpoint_path}")
    print("=" * 60)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Print top-level keys
        print("Top-level keys in checkpoint:")
        for key in sorted(checkpoint.keys()):
            if isinstance(checkpoint[key], dict):
                print(f"  {key}: dict with {len(checkpoint[key])} keys")
            elif isinstance(checkpoint[key], list):
                print(f"  {key}: list with {len(checkpoint[key])} items")
            elif hasattr(checkpoint[key], 'shape'):
                print(f"  {key}: tensor with shape {checkpoint[key].shape}")
            else:
                print(f"  {key}: {type(checkpoint[key]).__name__} = {checkpoint[key]}")
        
        print("\n" + "=" * 60)
        
        # Check training state
        print("Training State:")
        print(f"  Epoch: {checkpoint.get('epoch', 'NOT FOUND')}")
        print(f"  Global Step: {checkpoint.get('global_step', 'NOT FOUND')}")
        print(f"  PyTorch Lightning Version: {checkpoint.get('pytorch-lightning_version', 'NOT FOUND')}")
        
        # Check optimizer state
        print(f"\nOptimizer State:")
        if 'optimizer_states' in checkpoint:
            opt_states = checkpoint['optimizer_states']
            print(f"  Number of optimizers: {len(opt_states)}")
            for i, opt_state in enumerate(opt_states):
                if 'param_groups' in opt_state:
                    param_groups = opt_state['param_groups']
                    print(f"  Optimizer {i}:")
                    for j, group in enumerate(param_groups):
                        print(f"    Param group {j}: lr={group.get('lr', 'NOT FOUND')}")
                else:
                    print(f"  Optimizer {i}: No param_groups found")
        else:
            print("  NO OPTIMIZER STATES FOUND")
        
        # Check learning rate scheduler state
        print(f"\nLR Scheduler State:")
        if 'lr_schedulers' in checkpoint:
            lr_schedulers = checkpoint['lr_schedulers']
            print(f"  Number of LR schedulers: {len(lr_schedulers)}")
            for i, scheduler_state in enumerate(lr_schedulers):
                print(f"  Scheduler {i}:")
                if isinstance(scheduler_state, dict):
                    for key, value in scheduler_state.items():
                        if key == 'state_dict' and isinstance(value, dict):
                            print(f"    {key}:")
                            for skey, svalue in value.items():
                                print(f"      {skey}: {svalue}")
                        else:
                            print(f"    {key}: {value}")
                else:
                    print(f"    Type: {type(scheduler_state)}")
                    print(f"    Value: {scheduler_state}")
        else:
            print("  NO LR SCHEDULER STATES FOUND")
        
        # Check model state dict
        print(f"\nModel State:")
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            model_keys = [k for k in state_dict.keys() if not k.startswith('loss_fn.') and not k.startswith('metrics.')]
            print(f"  Model parameters: {len(model_keys)} tensors")
            if model_keys:
                print(f"  Sample keys: {list(model_keys)[:3]}...")
        else:
            print("  NO MODEL STATE DICT FOUND")
        
        # Check hyperparameters
        print(f"\nHyperparameters:")
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            print(f"  Found {len(hparams)} hyperparameters:")
            for key, value in hparams.items():
                if isinstance(value, dict):
                    print(f"    {key}: dict with {len(value)} keys")
                else:
                    print(f"    {key}: {value}")
        else:
            print("  NO HYPERPARAMETERS FOUND")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_checkpoint.py <checkpoint_path>")
        print("Example: python inspect_checkpoint.py outputs/baseline_unet_massroads/checkpoints/epoch=1559-step=109200.ckpt")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    inspect_checkpoint(checkpoint_path) 
