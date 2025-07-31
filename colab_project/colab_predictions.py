# Google Colab Prediction Script
# Run this cell to generate predictions

import torch
import os

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Generate predictions
print("\n=== GENERATING PREDICTIONS ===")
!python show_best_regression_predictions.py

print("\n=== PREDICTIONS COMPLETE ===")
