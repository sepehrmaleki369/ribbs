# Vessel Distance Map Regression - Google Colab Setup

## Quick Start

1. **Upload and Extract**: Run the colab_setup.py script
2. **Train Model**: Run the colab_training.py script  
3. **Generate Predictions**: Run the colab_predictions.py script

## Project Structure

- `configs/` - Configuration files
- `core/` - Core dataset and utilities
- `models/` - Model architectures
- `drive/` - Dataset (images and distance maps)
- `checkpoints_regression/` - Saved model checkpoints
- `train_regression.py` - Main training script
- `create_distance_maps_dataset_fixed.py` - Dataset creation
- `visualize_clean_skeletonization.py` - Visualization script

## GPU Training

This project is optimized for GPU training on Google Colab.
The model will automatically use CUDA if available.

## Expected Results

- Training time: ~10-15 minutes on GPU
- Validation MSE: ~1000-1200
- Validation MAE: ~20-25
- Generated predictions in predictions/ folder
