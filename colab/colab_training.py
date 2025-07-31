# Google Colab Training Script
# Run this cell to train the model

import torch
import os

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Train the model
print("\n=== STARTING TRAINING ===")
!python train_regression.py

print("\n=== TRAINING COMPLETE ===") 