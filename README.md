 
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

___

# Prompt for Creating SegLab-Compatible Models and Losses

## For Models

Please write a PyTorch model compatible with the SegLab framework with the following specifications:

[DESCRIBE YOUR MODEL HERE - e.g., "a UNet variation with attention gates" or "a lightweight segmentation model for road extraction"]

The model must follow these requirements to be compatible with SegLab:
1. Inherit from `torch.nn.Module`
2. Have an `__init__` method that accepts parameters configurable via YAML
3. Have a `forward` method that takes a single input tensor and returns the output tensor
4. For binary segmentation, apply sigmoid in the forward method
5. Handle tensor dimensions properly (N, C, H, W)

Here's a simplified example of the structure:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomModel(nn.Module):
    def __init__(
        self,
        in_channels=3,         # Number of input channels
        out_channels=1,        # Number of output channels
        base_filters=64,       # Initial number of filters
        depth=4,               # Network depth
        dropout=0.2            # Dropout rate
    ):
        super().__init__()
        
        # Define your model architecture here
        # ...
        
    def forward(self, x):
        # Forward pass implementation
        # ...
        
        # Apply sigmoid for binary segmentation
        return torch.sigmoid(x)
```

The model will be loaded by SegLab's model_loader using the path and class name specified in a YAML configuration file.

---

## For Loss Functions

Please write a PyTorch custom loss function compatible with the SegLab framework with the following specifications:

[DESCRIBE YOUR LOSS FUNCTION HERE - e.g., "a boundary-aware loss function that emphasizes road edges" or "a combined loss function that balances pixel-wise and structural similarity"]

The loss function must follow these requirements to be compatible with SegLab:
1. Inherit from `torch.nn.Module`
2. Have an `__init__` method that accepts parameters configurable via YAML
3. Have a `forward` method that takes `y_pred` and `y_true` tensors and returns a scalar loss value
4. Handle both binary and multi-class segmentation if specified
5. Be differentiable for backpropagation

Here's a simplified example of the structure:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomLoss(nn.Module):
    def __init__(
        self,
        weight=1.0,            # Weight for the loss component
        smooth=1e-6            # Smoothing factor for numerical stability
    ):
        super().__init__()
        self.weight = weight
        self.smooth = smooth
        
    def forward(self, y_pred, y_true):
        # Implement your loss calculation here
        # y_pred: model predictions (N, C, H, W)
        # y_true: ground truth (N, C, H, W)
        
        # Example: compute a simple weighted BCE
        bce = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
        
        # Return the final loss value
        return self.weight * bce
```

The loss function will be loaded by SegLab's loss_loader using the path and class name specified in a YAML configuration file.

---

## For Metrics

Please write a PyTorch custom metric compatible with the SegLab framework with the following specifications:

[DESCRIBE YOUR METRIC HERE - e.g., "a boundary F1 score that focuses on accuracy at object boundaries" or "a connectivity-aware metric for road networks"]

The metric must follow these requirements to be compatible with SegLab:
1. Inherit from `torch.nn.Module`
2. Have an `__init__` method that accepts parameters configurable via YAML
3. Have a `forward` method that takes `y_pred` and `y_true` tensors and returns a scalar metric value (higher should be better)
4. Handle both binary and multi-class segmentation if specified
5. Be able to handle batched inputs

Here's a simplified example of the structure:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomMetric(nn.Module):
    def __init__(
        self,
        threshold=0.5,         # Threshold for binary predictions
        eps=1e-6               # Small value for numerical stability
    ):
        super().__init__()
        self.threshold = threshold
        self.eps = eps
        
    def forward(self, y_pred, y_true):
        # Convert predictions to binary
        y_pred_bin = (y_pred > self.threshold).float()
        
        # Implement your metric calculation here
        # y_pred: model predictions (N, C, H, W)
        # y_true: ground truth (N, C, H, W)
        
        # Example: compute accuracy
        correct = (y_pred_bin == y_true).float()
        accuracy = torch.mean(correct)
        
        # Return the metric value (higher should be better)
        return accuracy
```

The metric will be loaded by SegLab's metric_loader using the path and class name specified in a YAML configuration file.


___

# Implementing Unsupervised and Semi-Supervised Learning in SegLab

This guide explains how to use SegLab for unsupervised learning (like autoencoders) and semi-supervised learning (like contrastive approaches) without needing to modify the dataset structure.

## Table of Contents
- [Unsupervised Learning with Autoencoders](#unsupervised-learning-with-autoencoders)
  - [Autoencoder Model Implementation](#autoencoder-model-implementation)
  - [Reconstruction Loss Functions](#reconstruction-loss-functions)
  - [Configuration and Training](#configuration-and-training)
- [Semi-Supervised Learning with Contrastive Approaches](#semi-supervised-learning-with-contrastive-approaches)
  - [Contrastive Model Implementation](#contrastive-model-implementation)
  - [Contrastive Loss Implementation](#contrastive-loss-implementation)
  - [Configuration and Training](#configuration-and-training-1)
- [Using the Models for Downstream Tasks](#using-the-models-for-downstream-tasks)

## Unsupervised Learning with Autoencoders

Autoencoders can be implemented within the SegLab framework by creating appropriate model and loss components that follow the framework's conventions.

### Autoencoder Model Implementation

Create an autoencoder model in the `models/` directory:

```python
# models/autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, hidden_dims=[32, 64, 128, 256]):
        super().__init__()
        
        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(hidden_dims[-1] * 4, latent_dim)
    
    def forward(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        z = self.fc_z(result)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3, hidden_dims=[256, 128, 64, 32]):
        super().__init__()
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 4)
        
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                     kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], out_channels,
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # For normalized image output
        )
    
    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 256, 2, 2)  # Adjust dimensions based on your input size
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

class Autoencoder(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        out_channels=3, 
        latent_dim=128,
        encoder_hidden_dims=[32, 64, 128, 256],
        decoder_hidden_dims=[256, 128, 64, 32]
    ):
        super().__init__()
        
        self.encoder = Encoder(in_channels, latent_dim, encoder_hidden_dims)
        self.decoder = Decoder(latent_dim, out_channels, decoder_hidden_dims)
    
    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction
    
    def encode(self, x):
        return self.encoder(x)
```

### Reconstruction Loss Functions

Create a reconstruction loss function in the `losses/` directory:

```python
# losses/reconstruction_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    def __init__(self, loss_type="mse", weight=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight
    
    def forward(self, y_pred, y_true):
        """
        y_pred: reconstructed image
        y_true: original image (same as input)
        
        Note: For use with SegLab, y_true would be passed in as the target/label,
        but for autoencoders, we compute loss against the original input.
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(y_pred, y_true)
        elif self.loss_type == "bce":
            loss = F.binary_cross_entropy(y_pred, y_true)
        elif self.loss_type == "l1":
            loss = F.l1_loss(y_pred, y_true)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        return self.weight * loss
```

### Configuration and Training

Create configuration files for the autoencoder:

```yaml
# configs/model/autoencoder.yaml
path: "models.autoencoder"
class: "Autoencoder"
params:
  in_channels: 3
  out_channels: 3
  latent_dim: 128
  encoder_hidden_dims: [32, 64, 128, 256]
  decoder_hidden_dims: [256, 128, 64, 32]
```

```yaml
# configs/loss/reconstruction.yaml
primary_loss:
  path: "losses.reconstruction_loss"
  class: "ReconstructionLoss"
  params:
    loss_type: "mse"
    weight: 1.0

secondary_loss: null
alpha: 0.0
start_epoch: 0
```

To use these in training, modify your main configuration:

```yaml
# configs/unsupervised_main.yaml
dataset_config: "your_dataset.yaml"
model_config: "autoencoder.yaml"
loss_config: "reconstruction.yaml"
metrics_config: "reconstruction_metrics.yaml"  # Define appropriate metrics
inference_config: "chunk.yaml"

# Output directory
output_dir: "outputs/autoencoder_experiment"

# Make sure your dataset contains the same values for both input and target keys
target_x: "image_patch"  # Input images
target_y: "image_patch"  # For autoencoders, target is the same as input

# Rest of configuration remains the same
```

## Semi-Supervised Learning with Contrastive Approaches

### Contrastive Model Implementation

Create a contrastive model in the `models/` directory:

```python
# models/contrastive_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderWithProjection(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        base_channels=64, 
        projection_dim=128, 
        n_levels=4
    ):
        super().__init__()
        
        # Use an existing encoder backbone (e.g., UNet encoder)
        from models.base_models import DownBlock
        
        # Encoder blocks - reusing existing components
        self.inc = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        
        # First block is special (no downsampling)
        self.down_blocks.append(DownBlock(base_channels, base_channels, 
                                        is_first=True, n_convs=2, three_dimensional=False))
        
        # Remaining blocks with downsampling
        current_channels = base_channels
        for i in range(1, n_levels):
            out_channels = current_channels * 2
            self.down_blocks.append(DownBlock(current_channels, out_channels, 
                                            is_first=False, n_convs=2, three_dimensional=False))
            current_channels = out_channels
        
        # Projection head
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(current_channels, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x):
        # Initial conv
        x = self.inc(x)
        
        # Down blocks
        features = []
        for block in self.down_blocks:
            x = block(x)
            features.append(x)
        
        # Projection
        projection = self.projection(x)
        # Normalize projection to unit sphere
        projection = F.normalize(projection, dim=1)
        
        return projection, features
    
class DecoderWithSegmentation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=1,
        n_levels=4
    ):
        super().__init__()
        
        # Use existing decoder structure (e.g., from UNet)
        from models.base_models import UpBlock
        
        self.up_blocks = nn.ModuleList()
        
        # Create decoder blocks
        current_channels = in_channels
        for i in range(n_levels - 1):
            self.up_blocks.append(UpBlock(current_channels, n_convs=2, three_dimensional=False))
            current_channels = current_channels // 2
        
        # Final convolution for segmentation
        self.final_conv = nn.Conv2d(current_channels, out_channels, kernel_size=1)
    
    def forward(self, features):
        """
        features: List of features from encoder (in reverse order)
        """
        # Start with the bottleneck features
        x = features[-1]
        
        # Upsampling path
        for i, block in enumerate(self.up_blocks):
            skip = features[-(i+2)]  # Corresponding skip connection
            x = block(x, skip)
        
        # Final convolution
        x = self.final_conv(x)
        
        return torch.sigmoid(x)  # Apply sigmoid for binary segmentation

class ContrastiveSegmentationModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base_channels=64,
        projection_dim=128,
        n_levels=4
    ):
        super().__init__()
        
        self.encoder = EncoderWithProjection(
            in_channels, base_channels, projection_dim, n_levels
        )
        
        # Last encoder output channels = base_channels * 2**(n_levels-1)
        last_channels = base_channels * (2 ** (n_levels - 1))
        
        self.decoder = DecoderWithSegmentation(
            last_channels, out_channels, n_levels
        )
        
    def forward(self, x):
        """Standard forward pass for segmentation"""
        projection, features = self.encoder(x)
        segmentation = self.decoder(features)
        return segmentation
    
    def encode_project(self, x):
        """Forward pass for contrastive learning"""
        projection, _ = self.encoder(x)
        return projection
    
    def train_step(self, x1, x2=None):
        """
        Training step for contrastive learning
        
        If x2 is provided:
          - Compute projections for both views
          - Return both projections for contrastive loss
        
        If x2 is None:
          - Perform regular segmentation
        """
        if x2 is not None:
            # Contrastive learning mode
            z1, _ = self.encoder(x1)
            z2, _ = self.encoder(x2)
            return {'projections': (z1, z2)}
        else:
            # Segmentation mode
            return {'segmentation': self.forward(x1)}
```

### Contrastive Loss Implementation

Create a contrastive loss function:

```python
# losses/contrastive_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (from SimCLR)
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        
    def forward(self, z_i, z_j):
        """
        z_i, z_j are batches of embeddings, where corresponding pairs are positive examples
        """
        batch_size = z_i.shape[0]
        
        # Concatenate embeddings from both augmentations
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), 
                                               representations.unsqueeze(0), dim=2)
        
        # Remove self-similarity
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        # Remove diagonals - don't compare a representation with itself
        mask = (~torch.eye(batch_size*2, batch_size*2, dtype=bool, device=z_i.device))
        negatives = similarity_matrix[mask].view(batch_size*2, -1)
        
        # Concatenate positives and negatives for each row
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        
        # Divide by temperature
        logits = logits / self.temperature
        
        # Labels always indicate the positive sample is at index 0
        labels = torch.zeros(batch_size*2, dtype=torch.long, device=z_i.device)
        
        loss = self.criterion(logits, labels)
        
        return loss

class SegmentationWithContrastiveLoss(nn.Module):
    """
    Loss function that combines segmentation and contrastive learning
    """
    def __init__(
        self, 
        contrastive_weight=1.0, 
        segmentation_weight=1.0,
        temperature=0.5,
        segmentation_loss_type='bce'
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.segmentation_weight = segmentation_weight
        self.contrastive_loss = NTXentLoss(temperature)
        
        # Segmentation loss
        if segmentation_loss_type == 'bce':
            self.seg_loss = nn.BCELoss()
        elif segmentation_loss_type == 'dice':
            # You could add a custom Dice loss here
            self.seg_loss = nn.BCELoss()
        else:
            self.seg_loss = nn.BCELoss()
            
    def forward(self, y_pred, y_true):
        """
        For compatibility with SegLab, handle either dict or tensor input:
        
        If y_pred is dict with 'projections' key:
            - Compute contrastive loss
        If y_pred is a tensor:
            - Compute segmentation loss
        """
        if isinstance(y_pred, dict) and 'projections' in y_pred:
            # Contrastive mode
            z1, z2 = y_pred['projections']
            return self.contrastive_weight * self.contrastive_loss(z1, z2)
        elif isinstance(y_pred, dict) and 'segmentation' in y_pred:
            # Segmentation mode
            return self.segmentation_weight * self.seg_loss(y_pred['segmentation'], y_true)
        else:
            # Default (assume segmentation output)
            return self.segmentation_weight * self.seg_loss(y_pred, y_true)
```

### Configuration and Training

Create configuration files for the contrastive approach:

```yaml
# configs/model/contrastive.yaml
path: "models.contrastive_model"
class: "ContrastiveSegmentationModel"
params:
  in_channels: 3
  out_channels: 1
  base_channels: 64
  projection_dim: 128
  n_levels: 4
```

```yaml
# configs/loss/contrastive.yaml
primary_loss:
  path: "losses.contrastive_loss"
  class: "SegmentationWithContrastiveLoss"
  params:
    contrastive_weight: 1.0
    segmentation_weight: 1.0
    temperature: 0.5
    segmentation_loss_type: 'bce'

secondary_loss: null
alpha: 0.0
start_epoch: 0
```

## Using the Models for Downstream Tasks

Once you've trained your unsupervised or semi-supervised models, you can use them for downstream tasks:

### 1. Using Pretrained Encoder for Segmentation

```python
# Load pretrained autoencoder
autoencoder = load_model(autoencoder_config)
autoencoder.load_state_dict(torch.load('path/to/autoencoder/checkpoint.ckpt'))

# Create a segmentation model using the pretrained encoder
class SegmentationModel(nn.Module):
    def __init__(self, pretrained_encoder, out_channels=1):
        super().__init__()
        self.encoder = pretrained_encoder.encoder
        # Add your decoder for segmentation
        self.decoder = Decoder(...)
        
    def forward(self, x):
        features = self.encoder(x)
        segmentation = self.decoder(features)
        return segmentation

# Initialize with pretrained weights
segmentation_model = SegmentationModel(autoencoder, out_channels=1)
```

### 2. Fine-tuning a Contrastive Model for Segmentation

For contrastive models that already have a segmentation head, you can simply fine-tune the model:

```python
# Load pretrained contrastive model
contrastive_model = load_model(contrastive_config)
contrastive_model.load_state_dict(torch.load('path/to/contrastive/checkpoint.ckpt'))

# Fine-tune with segmentation loss only
segmentation_loss = nn.BCELoss()

# Use the model directly for segmentation tasks
# The forward method already outputs segmentation
```

This approach allows you to leverage unsupervised and semi-supervised learning techniques within the SegLab framework without needing to modify the dataset structure fundamentally. You just need to be careful about how the data is processed during training and how the loss functions are applied.