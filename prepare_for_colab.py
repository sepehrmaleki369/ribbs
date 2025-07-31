import os
import shutil
import zipfile

def prepare_for_colab():
    """Prepare the project for Google Colab"""
    
    print("=== PREPARING PROJECT FOR GOOGLE COLAB ===")
    
    # Create colab directory
    colab_dir = "colab_project"
    if os.path.exists(colab_dir):
        shutil.rmtree(colab_dir)
    os.makedirs(colab_dir)
    
    # Essential directories to copy
    essential_dirs = [
        "configs",
        "core", 
        "models",
        "drive",
        "checkpoints_regression"
    ]
    
    # Essential files to copy
    essential_files = [
        "requirements.txt",
        "train_regression.py",
        "create_distance_maps_dataset_fixed.py",
        "visualize_clean_skeletonization.py"
    ]
    
    print("Copying essential directories...")
    for dir_name in essential_dirs:
        if os.path.exists(dir_name):
            shutil.copytree(dir_name, os.path.join(colab_dir, dir_name))
            print(f"✅ Copied {dir_name}/")
        else:
            print(f"⚠️  {dir_name}/ not found")
    
    print("Copying essential files...")
    for file_name in essential_files:
        if os.path.exists(file_name):
            shutil.copy2(file_name, colab_dir)
            print(f"✅ Copied {file_name}")
        else:
            print(f"⚠️  {file_name} not found")
    
    # Create colab setup script
    colab_setup_script = """# Google Colab Setup Script
# Run this cell to set up the project

import os
import zipfile
from google.colab import files

# Upload the project zip file
print("Please upload the colab_project.zip file...")
uploaded = files.upload()

# Extract the project
for filename in uploaded.keys():
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('.')
    print(f"✅ Extracted {filename}")

# Install requirements
print("Installing requirements...")
!pip install -r requirements.txt

# Verify setup
print("\\n=== PROJECT SETUP COMPLETE ===")
print("Available files:")
!ls -la

print("\\nAvailable directories:")
!ls -la | grep "^d"

print("\\n✅ Project ready for training!")
"""
    
    with open(os.path.join(colab_dir, "colab_setup.py"), "w") as f:
        f.write(colab_setup_script)
    
    # Create training script for colab
    colab_training_script = """# Google Colab Training Script
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
print("\\n=== STARTING TRAINING ===")
!python train_regression.py

print("\\n=== TRAINING COMPLETE ===")
"""
    
    with open(os.path.join(colab_dir, "colab_training.py"), "w") as f:
        f.write(colab_training_script)
    
    # Create prediction script for colab
    colab_prediction_script = """# Google Colab Prediction Script
# Run this cell to generate predictions

import torch
import os

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Generate predictions
print("\\n=== GENERATING PREDICTIONS ===")
!python show_best_regression_predictions.py

print("\\n=== PREDICTIONS COMPLETE ===")
"""
    
    with open(os.path.join(colab_dir, "colab_predictions.py"), "w") as f:
        f.write(colab_prediction_script)
    
    # Create README for colab
    readme_content = """# Vessel Distance Map Regression - Google Colab Setup

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
"""
    
    with open(os.path.join(colab_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    # Create zip file
    zip_filename = "colab_project.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(colab_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, colab_dir)
                zipf.write(file_path, arcname)
    
    print(f"\n✅ Project prepared for Colab!")
    print(f"📦 Zip file created: {zip_filename}")
    print(f"📁 Project directory: {colab_dir}/")
    
    # Print instructions
    print(f"\n=== GOOGLE COLAB SETUP INSTRUCTIONS ===")
    print(f"1. Download {zip_filename} from your local machine")
    print(f"2. Upload it to Google Colab")
    print(f"3. Run the colab_setup.py script")
    print(f"4. Run the colab_training.py script for GPU training")
    print(f"5. Run the colab_predictions.py script for predictions")
    
    return zip_filename

if __name__ == "__main__":
    prepare_for_colab() 