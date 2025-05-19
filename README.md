 
# SegLab: Modular PyTorch-Lightning Segmentation Framework

SegLab is a flexible, modular framework for image segmentation experiments built with PyTorch Lightning. The framework is designed to be easily extensible, allowing researchers and practitioners to implement custom models, loss functions, and metrics with minimal effort.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Using Models](#using-models)
  - [Built-in Models](#built-in-models)
  - [Creating Custom Models](#creating-custom-models)
- [Loss Functions](#loss-functions)
  - [Built-in Losses](#built-in-losses)
  - [Creating Custom Losses](#creating-custom-losses)
  - [Mixed Losses](#mixed-losses)
- [Metrics](#metrics)
  - [Built-in Metrics](#built-in-metrics)
  - [Creating Custom Metrics](#creating-custom-metrics)
- [Configuration System](#configuration-system)
- [Training and Evaluation](#training-and-evaluation)
- [Monitoring with TensorBoard](#monitoring-with-tensorboard)
  - [Sharing TensorBoard with Cloudflare](#sharing-tensorboard-with-cloudflare)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## Introduction

SegLab provides a unified, configuration-driven interface for image segmentation tasks. Key features include:

- **Modular architecture**: Easily swap between models, losses, and metrics
- **Configuration-driven**: All parameters defined in YAML files with no hardcoded values
- **Full-size image processing**: Process images of any size during validation and testing
- **Comprehensive callbacks**: Best checkpoints saved for each metric, visualization, code archiving
- **Detailed logging**: Color-coded console output, file logging, TensorBoard integration
- **Checkpoint management**: Resume training from any checkpoint
- **Mixed precision training**: Speed up training with automatic mixed precision

## Project Structure

```
seglab/
├── configs/                   # Configuration files
│   ├── dataset/              # Dataset configurations
│   ├── model/                # Model configurations
│   ├── loss/                 # Loss configurations
│   ├── metrics/              # Metrics configurations
│   ├── inference/            # Inference configurations
│   └── main.yaml             # Main configuration
├── core/                     # Core framework components
│   ├── model_loader.py       # Dynamic model loading
│   ├── loss_loader.py        # Dynamic loss loading
│   ├── mix_loss.py           # Mixed loss implementation
│   ├── metric_loader.py      # Dynamic metric loading
│   ├── dataloader.py         # Dataset wrapper
│   ├── validator.py          # Chunked inference implementation
│   ├── callbacks.py          # Training callbacks
│   ├── logger.py             # Logging setup
│   ├── checkpoint.py         # Checkpoint management
│   └── utils.py              # Utility functions
├── models/                   # Model implementations
│   ├── base_models.py        # Built-in model implementations
│   ├── custom_model.py       # Example custom model
│   └── ...                   # Your custom models
├── losses/                   # Loss implementations
│   ├── custom_loss.py        # Example custom loss
│   └── ...                   # Your custom losses
├── metrics/                  # Metric implementations
│   ├── apls.py               # Average Path Length Similarity metric
│   ├── connected_components.py # Connected Components Quality metric
│   └── ...                   # Your custom metrics
├── seglit_module.py          # PyTorch Lightning module
├── train.py                  # Training script
└── requirements.txt          # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/seglab.git
cd seglab
```

2. Create a virtual environment and install dependencies:
```bash
# Using conda
conda create -n seglab python=3.8
conda activate seglab
pip install -r requirements.txt

# Using venv
python -m venv seglab-env
source seglab-env/bin/activate  # On Windows, use: seglab-env\Scripts\activate
pip install -r requirements.txt
```

3. Prepare your dataset:
```bash
# Dataset structure should follow:
# data/
# ├── mydataset/
# │   ├── train/
# │   │   ├── image/      # Input images
# │   │   └── label/      # Ground truth masks
# │   ├── valid/
# │   │   ├── image/
# │   │   └── label/
# │   └── test/
# │       ├── image/
# │       └── label/
```

## Getting Started

1. Update configuration files according to your dataset:

```bash
# Configure dataset
nano configs/dataset/mydataset.yaml

# Configure main file to reference your dataset
nano configs/main.yaml  # Update dataset_config: "mydataset.yaml"
```

2. Run training:

```bash
python train.py --config configs/main.yaml
```

## Using Models

### Built-in Models

SegLab comes with several built-in models in `models/base_models.py`, including:

1. **UNet**: Basic UNet architecture for semantic segmentation
2. **UNetReg**: UNet variant for regression tasks
3. **UNetBin**: UNet variant for binary segmentation

To use a built-in model, create a model configuration file:

```yaml
# configs/model/unet_base.yaml
# Path and class
path: "models.base_models"  # Path to base_models.py
class: "UNet"               # Model class name

# Model parameters
params:
  in_channels: 3            # Number of input channels (RGB)
  m_channels: 64            # Base number of channels
  out_channels: 1           # Number of output channels (1 for binary)
  n_convs: 2                # Number of convolutions per block
  n_levels: 4               # Depth of the UNet
  dropout: 0.2              # Dropout rate
  batch_norm: true          # Use batch normalization
  upsampling: "deconv"      # Type of upsampling ("deconv", "nearest", "bilinear")
  pooling: "max"            # Type of pooling ("max", "avg")
  three_dimensional: false  # 2D or 3D model
```

Then update your main configuration to use this model:

```yaml
# configs/main.yaml
model_config: "unet_base.yaml"
```

### Creating Custom Models

You can create your own custom models by:

1. Creating a new Python file in the `models/` directory:

```python
# models/my_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomSegModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(features, features*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, features//2, kernel_size=2, stride=2),
            nn.Conv2d(features//2, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)  # For binary segmentation
```

2. Creating a configuration file for your model:

```yaml
# configs/model/my_custom_model.yaml
path: "models.my_model"          # Path to your model file
class: "MyCustomSegModel"        # Your model class name

params:
  in_channels: 3
  out_channels: 1
  features: 64
```

3. Updating the main configuration to use your model:

```yaml
# configs/main.yaml
model_config: "my_custom_model.yaml"
```

## Loss Functions

### Built-in Losses

SegLab supports all PyTorch built-in losses like BCE, MSE, etc., which can be directly referenced in the loss configuration:

```yaml
# configs/loss/bce_loss.yaml
# Using built-in PyTorch loss
primary_loss:
  class: "BCELoss"  # PyTorch's built-in Binary Cross Entropy
  params: {}

secondary_loss: null
alpha: 0.0
start_epoch: 0
```

```yaml
# configs/loss/weighted_bce_loss.yaml
# Using built-in PyTorch loss with weights
primary_loss:
  class: "BCELoss"
  params:
    weight: [0.1, 0.9]  # Class weights for imbalanced datasets
    reduction: "mean"

secondary_loss: null
alpha: 0.0
start_epoch: 0
```

```yaml
# configs/loss/mse_loss.yaml
# Using built-in PyTorch MSE loss
primary_loss:
  class: "MSELoss"  # PyTorch's built-in Mean Squared Error
  params: {}

secondary_loss: null
alpha: 0.0
start_epoch: 0
```

### Creating Custom Losses

To implement a custom loss function:

1. Create a new Python file in the `losses/` directory:

```python
# losses/boundary_aware_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryAwareLoss(nn.Module):
    """
    Loss function that gives higher weight to boundary regions
    """
    def __init__(self, boundary_weight=5.0, smooth=1e-6):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.smooth = smooth
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, y_pred, y_true):
        # Basic BCE loss
        bce_loss = self.bce(y_pred, y_true)
        
        # Calculate boundary using morphological operations
        # Simple approach: Dilated - Eroded = Boundary
        kernel_size = 3
        padding = kernel_size // 2
        
        # Max pooling for dilation
        pooled = F.max_pool2d(y_true, kernel_size=3, stride=1, padding=1)
        # Min pooling for erosion (using 1-y_true trick)
        eroded = 1 - F.max_pool2d(1 - y_true, kernel_size=3, stride=1, padding=1)
        # Boundary mask
        boundary = (pooled - eroded) > 0
        
        # Create weight map
        weights = torch.ones_like(y_true)
        weights[boundary] = self.boundary_weight
        
        # Apply weights to BCE loss
        weighted_loss = (bce_loss * weights).mean()
        
        return weighted_loss
```

2. Create a configuration file for your loss:

```yaml
# configs/loss/boundary_aware.yaml
primary_loss:
  path: "losses.boundary_aware_loss"  # Path to your loss module
  class: "BoundaryAwareLoss"          # Loss class name
  params:
    boundary_weight: 5.0              # Weight for boundary pixels
    smooth: 0.000001                  # Numerical stability constant

secondary_loss: null
alpha: 0.0
start_epoch: 0
```

3. Update your main configuration to use this loss:

```yaml
# configs/main.yaml
loss_config: "boundary_aware.yaml"
```

### Mixed Losses

SegLab supports combining multiple loss functions with different weights and activation epochs:

```yaml
# configs/loss/mixed_loss.yaml
# Primary loss (used from the beginning)
primary_loss:
  class: "BCELoss"
  params: {}

# Secondary loss (activated after start_epoch)
secondary_loss:
  path: "losses.boundary_aware_loss"
  class: "BoundaryAwareLoss"
  params:
    boundary_weight: 5.0
    smooth: 0.000001

# Mixing parameters
alpha: 0.3          # Weight of secondary loss
start_epoch: 10     # Epoch to start using the secondary loss
```

This configuration will use only `BCELoss` for the first 10 epochs, then gradually incorporate `BoundaryAwareLoss` with a 0.3 weight.

## Metrics

### Built-in Metrics

SegLab comes with several built-in metrics and supports integration with `torchmetrics`:

```yaml
# configs/metrics/segmentation.yaml
metrics:
  # Dice coefficient
  - alias: "dice"
    path: "torchmetrics.classification"
    class: "Dice"
    params:
      threshold: 0.5
      zero_division: 1.0

  # IoU (Jaccard index)
  - alias: "iou"
    path: "torchmetrics.classification"
    class: "JaccardIndex"
    params:
      task: "binary"
      threshold: 0.5
      num_classes: 2

  # Connected Components Quality
  - alias: "ccq"
    path: "metrics.connected_components"
    class: "ConnectedComponentsQuality"
    params:
      min_size: 5
      tolerance: 2
```

### Creating Custom Metrics

To implement a custom metric:

1. Create a new Python file in the `metrics/` directory:

```python
# metrics/boundary_accuracy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryAccuracy(nn.Module):
    """
    Metric that evaluates segmentation accuracy specifically at boundaries
    """
    def __init__(self, boundary_width=2, threshold=0.5):
        super().__init__()
        self.boundary_width = boundary_width
        self.threshold = threshold
        
    def forward(self, y_pred, y_true):
        # Threshold predictions
        y_pred_bin = (y_pred > self.threshold).float()
        
        # Find boundaries in the ground truth
        kernel_size = self.boundary_width * 2 + 1
        padding = kernel_size // 2
        
        # Dilated - Eroded = Boundary
        dilated = F.max_pool2d(y_true, kernel_size=kernel_size, stride=1, padding=padding)
        eroded = 1 - F.max_pool2d(1 - y_true, kernel_size=kernel_size, stride=1, padding=padding)
        boundaries = (dilated - eroded) > 0
        
        # Accuracy at boundaries
        correct = (y_pred_bin == y_true).float()
        boundary_correct = correct * boundaries
        
        # Calculate accuracy
        if torch.sum(boundaries) > 0:
            accuracy = torch.sum(boundary_correct) / torch.sum(boundaries)
        else:
            accuracy = torch.tensor(1.0, device=y_pred.device)  # No boundaries
            
        return accuracy
```

2. Update your metrics configuration to include your new metric:

```yaml
# configs/metrics/segmentation.yaml
metrics:
  # ... existing metrics
  
  # Custom boundary accuracy metric
  - alias: "boundary_acc"
    path: "metrics.boundary_accuracy"
    class: "BoundaryAccuracy"
    params:
      boundary_width: 2
      threshold: 0.5
```

3. Optionally set different frequencies for computing metrics:

```yaml
# Per-metric frequencies for training
train_frequencies:
  dice: 1          # Compute every epoch
  iou: 1           # Compute every epoch
  boundary_acc: 5  # Compute every 5 epochs (if expensive)

# Per-metric frequencies for validation
val_frequencies:
  dice: 1
  iou: 1
  boundary_acc: 2  # Compute every 2 validation runs
```

## Configuration System

SegLab uses a hierarchical configuration system with YAML files:

1. **Main configuration** (`configs/main.yaml`): References all sub-configs and sets high-level parameters

```yaml
# Main configuration file
dataset_config: "massroads.yaml"
model_config: "unet_base.yaml"
loss_config: "mixed_loss.yaml"
metrics_config: "segmentation.yaml"
inference_config: "chunk.yaml"

# Output directory
output_dir: "outputs/experiment_1"

# Trainer configuration
trainer:
  max_epochs: 100
  val_check_interval: 1.0
  skip_validation_until_epoch: 5
  val_every_n_epochs: 1
  log_every_n_epochs: 2

# Optimizer configuration
optimizer:
  name: "Adam"
  params:
    lr: 0.001
    weight_decay: 0.0001
  
  # Optional learning rate scheduler
  scheduler:
    name: "ReduceLROnPlateau"
    params:
      patience: 10
      factor: 0.5
      monitor: "val_loss"
      mode: "min"
      min_lr: 0.00001

# Input/output keys for dynamic dataset access
target_x: "image_patch"
target_y: "label_patch"
```

2. Create configuration files for each component (dataset, model, loss, metrics) as described in the previous sections.

## Training and Evaluation

Basic training commands:

```bash
# Start training with the main config
python train.py --config configs/main.yaml

# Resume training from a checkpoint
python train.py --config configs/main.yaml --resume outputs/experiment_1/checkpoints/last.ckpt

# Run testing on the best checkpoint
python train.py --config configs/main.yaml --test
```

## Monitoring with TensorBoard

SegLab automatically logs metrics, loss values, and sample visualizations to TensorBoard. To view the logs:

```bash
# Launch TensorBoard
tensorboard --logdir outputs/experiment_1/logs --port 6006

# Open your browser at http://localhost:6006
```

### Sharing TensorBoard with Cloudflare

To share your TensorBoard with others (especially useful for remote servers), you can use Cloudflare:

1. Install Cloudflared:
```bash
# Download Cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared
```

2. Create a tunnel to your TensorBoard:
```bash
# Make sure TensorBoard is running
tensorboard --logdir outputs/experiment_1/logs --port 6006

# In a new terminal, create the tunnel
./cloudflared tunnel --url http://localhost:6006
```

3. Cloudflare will provide a public URL that you can share with others. This URL will be displayed in the terminal after running the command.

## Advanced Features

SegLab includes several advanced features:

1. **Chunked Inference**: Process large images by splitting them into overlapping patches
   ```yaml
   # configs/inference/chunk.yaml
   patch_size: [512, 512]  # Process images in 512x512 patches
   patch_margin: [64, 64]  # With 64-pixel overlap
   ```

2. **Selective Metric Calculation**: Control which metrics are calculated and when
   ```yaml
   # Apply in configs/metrics/segmentation.yaml
   train_frequencies:
     dice: 1    # Calculate every epoch
     iou: 1     # Calculate every epoch
     apls: 10   # Calculate every 10 epochs (expensive metric)
   ```

3. **Mixed Precision Training**: Enable via trainer config
   ```yaml
   # In configs/main.yaml
   trainer:
     # ...
     extra_args:
       precision: 16  # Enable mixed precision training
   ```

4. **Gradient Accumulation**: For training with effective larger batch sizes
   ```yaml
   # In configs/main.yaml
   trainer:
     # ...
     extra_args:
       accumulate_grad_batches: 4  # Accumulate gradients over 4 batches
   ```

## Troubleshooting

### Common Issues

1. **Image loading issues with libjpeg/libpng**:
   If you encounter `UserWarning: Failed to load image Python extension: … undefined symbol: _XXXXX`, install the required libraries:
   ```bash
   sudo apt-get install libjpeg-dev libpng-dev
   ```

2. **CUDA out of memory**:
   - Reduce batch size in your dataset configuration
   - Use chunked inference with smaller patch sizes
   - Enable mixed precision training (precision: 16)

3. **Input size validation errors**:
   The UNet models require input dimensions to be divisible by 2^(n_levels+1). If you encounter an error, adjust your patch size or model depth.

4. **Slow metric calculation**:
   Use the `train_frequencies` and `val_frequencies` settings to calculate expensive metrics less frequently.

For more help, check the framework documentation or open an issue on the GitHub repository.

---

This framework is designed to be modular and easily extensible. If you have specific requirements or encounter any issues, please open an issue on GitHub.

____
# Share your Tensorboard
```
tensorboard --logdir path/to/logs --port 6006 # or any open port
```

```
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared
```

```
./cloudflared tunnel --url http://localhost:6006
```


# UserWarning
Failed to load image Python extension: … undefined symbol: _XXXXX

``` 
sudo apt-get install libjpeg-dev libpng-dev
```