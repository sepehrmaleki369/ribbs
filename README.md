 
# SegLab: Modular PyTorch-Lightning Segmentation Framework

SegLab is a flexible, modular framework for image segmentation experiments built with PyTorch Lightning. The framework is designed to be easily extensible, allowing researchers and practitioners to implement custom models, loss functions, and metrics with minimal effort.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Implementing Custom Components](#implementing-custom-components)
  - [Custom Models](#custom-models)
  - [Custom Loss Functions](#custom-loss-functions)
  - [Custom Metrics](#custom-metrics)
- [Configuration](#configuration)
- [Training and Evaluation](#training-and-evaluation)
- [Resuming Training](#resuming-training)
- [Advanced Features](#advanced-features)

## Project Overview

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

## Implementing Custom Components

### Custom Models

SegLab makes it easy to implement custom segmentation models. Models should be defined as PyTorch modules that inherit from `nn.Module`.

1. Create a new Python file in the `models/` directory:

```python
# models/my_custom_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomModel(nn.Module):
    """
    My custom segmentation model.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_filters: int = 64,
        depth: int = 5,
        dropout: float = 0.2
    ):
        """
        Initialize the custom model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            base_filters: Initial number of filters
            depth: Network depth (number of downsampling steps)
            dropout: Dropout rate
        """
        super().__init__()
        
        # Define your model architecture here
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        # Example: initial block
        self.initial_conv = nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(base_filters)
        
        # Encoder blocks
        current_channels = base_filters
        for i in range(depth):
            # Add encoder blocks here
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, current_channels * 2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(current_channels * 2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )
            )
            current_channels *= 2
            
        # Decoder blocks
        for i in range(depth):
            # Add decoder blocks here
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(current_channels, current_channels // 2, kernel_size=2, stride=2),
                    nn.Conv2d(current_channels // 2, current_channels // 2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(current_channels // 2),
                    nn.ReLU(inplace=True)
                )
            )
            current_channels //= 2
            
        # Final layers
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Initial block
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Encoder path with skip connections
        skip_connections = []
        for block in self.encoder_blocks:
            skip_connections.append(x)
            x = block(x)
        
        # Decoder path with skip connections
        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
            # Handle skip connections
            skip = skip_connections[-(i+1)]
            # Handle size mismatches if necessary
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip  # Skip connection
        
        # Final layers
        x = self.final_conv(x)
        
        return torch.sigmoid(x)  # Apply activation for segmentation
```

2. Create a configuration file for your model in `configs/model/`:

```yaml
# configs/model/my_custom_model.yaml
# Path and class
path: "models.my_custom_model"  # Path to the module containing the model
class: "MyCustomModel"          # Name of the model class

# Model parameters
params:
  in_channels: 3                # Number of input channels
  out_channels: 1               # Number of output channels (1 for binary, >1 for multi-class)
  base_filters: 64              # Initial number of filters
  depth: 5                      # Network depth
  dropout: 0.2                  # Dropout rate
```

3. Update the main configuration to use your model:

```yaml
# configs/main.yaml
# ... other configuration options

# Set model config to your custom model
model_config: "my_custom_model.yaml"

# ... rest of configuration
```

### Custom Loss Functions

Implementing custom loss functions follows a similar pattern:

1. Create a new Python file in the `losses/` directory:

```python
# losses/my_custom_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomLoss(nn.Module):
    """
    Custom loss function for segmentation.
    
    Combines binary cross-entropy with a boundary-aware term.
    """
    
    def __init__(
        self,
        boundary_weight: float = 0.5,
        smooth: float = 1e-5
    ):
        """
        Initialize the custom loss.
        
        Args:
            boundary_weight: Weight for the boundary component
            smooth: Smoothing factor for numerical stability
        """
        super().__init__()
        self.boundary_weight = boundary_weight
        self.smooth = smooth
        self.bce_loss = nn.BCELoss()
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the custom loss.
        
        Args:
            y_pred: Predicted segmentation masks
            y_true: Ground truth segmentation masks
            
        Returns:
            Tensor containing the calculated loss
        """
        # Binary cross-entropy component
        bce = self.bce_loss(y_pred, y_true)
        
        # Compute boundaries using simple edge detection
        # For example, using a simple gradient-based approach
        y_true_boundaries = self._compute_boundaries(y_true)
        y_pred_boundaries = self._compute_boundaries(y_pred)
        
        # Boundary loss component
        boundary_loss = F.mse_loss(y_pred_boundaries, y_true_boundaries)
        
        # Combined loss
        total_loss = (1 - self.boundary_weight) * bce + self.boundary_weight * boundary_loss
        
        return total_loss
    
    def _compute_boundaries(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract boundaries from segmentation masks using gradients.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with highlighted boundaries
        """
        # Simple sobel-like edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=x.device).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=x.device).reshape(1, 1, 3, 3)
        
        if x.dim() == 4:  # (batch, channels, height, width)
            # Process each channel separately
            edges = []
            for c in range(x.shape[1]):
                # Extract channel
                channel = x[:, c:c+1, :, :]
                # Apply edge detection
                grad_x = F.conv2d(channel, sobel_x, padding=1)
                grad_y = F.conv2d(channel, sobel_y, padding=1)
                # Compute magnitude
                edge = torch.sqrt(grad_x**2 + grad_y**2 + self.smooth)
                edges.append(edge)
            # Combine edges from all channels
            return torch.cat(edges, dim=1)
        else:
            # Apply directly if no channel dimension
            grad_x = F.conv2d(x, sobel_x, padding=1)
            grad_y = F.conv2d(x, sobel_y, padding=1)
            return torch.sqrt(grad_x**2 + grad_y**2 + self.smooth)
```

2. Create a configuration file for your loss in `configs/loss/`:

```yaml
# configs/loss/my_custom_loss.yaml
# Primary loss
primary_loss:
  path: "losses.my_custom_loss"  # Path to the module containing the loss
  class: "MyCustomLoss"          # Name of the loss class
  params:
    boundary_weight: 0.5         # Weight for the boundary component
    smooth: 0.00001              # Smoothing factor

# Secondary loss (optional, can be null)
secondary_loss: null

# Mixing parameters (used only if secondary_loss is defined)
alpha: 0.0                      # Weight for the secondary loss
start_epoch: 0                  # Epoch to start using the secondary loss
```

3. Update the main configuration to use your loss:

```yaml
# configs/main.yaml
# ... other configuration options

# Set loss config to your custom loss
loss_config: "my_custom_loss.yaml"

# ... rest of configuration
```

### Custom Metrics

Implementing custom metrics follows the same pattern:

1. Create a new Python file in the `metrics/` directory:

```python
# metrics/my_custom_metric.py
import torch
import torch.nn as nn

class MyCustomMetric(nn.Module):
    """
    Custom metric for evaluating segmentation quality.
    
    This example implements a weighted F1 score with emphasis on boundary regions.
    """
    
    def __init__(
        self,
        boundary_weight: float = 2.0,
        eps: float = 1e-6
    ):
        """
        Initialize the custom metric.
        
        Args:
            boundary_weight: Weight multiplier for boundary pixels
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.boundary_weight = boundary_weight
        self.eps = eps
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the custom metric.
        
        Args:
            y_pred: Predicted segmentation masks (B, C, H, W)
            y_true: Ground truth segmentation masks (B, C, H, W)
            
        Returns:
            Tensor containing the metric value (higher is better)
        """
        # Threshold predictions to get binary masks
        y_pred_bin = (y_pred > 0.5).float()
        
        # Compute boundary weights
        boundaries = self._detect_boundaries(y_true)
        weights = 1.0 + (self.boundary_weight - 1.0) * boundaries
        
        # Compute weighted true positives, false positives, false negatives
        tp = torch.sum(weights * y_pred_bin * y_true, dim=[1, 2, 3])
        fp = torch.sum(weights * y_pred_bin * (1 - y_true), dim=[1, 2, 3])
        fn = torch.sum(weights * (1 - y_pred_bin) * y_true, dim=[1, 2, 3])
        
        # Compute precision and recall
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        
        # Compute F1 score
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        
        # Return batch average
        return f1.mean()
    
    def _detect_boundaries(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Detect boundary regions in masks.
        
        Args:
            masks: Input segmentation masks
            
        Returns:
            Tensor with 1s at boundary locations, 0s elsewhere
        """
        # Detect edges with simple morphological operations
        # For example, using dilation and erosion to identify boundaries
        b, c, h, w = masks.shape
        
        # Create kernels for morphological operations
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=masks.device)
        
        boundaries = torch.zeros_like(masks)
        for i in range(b):
            for j in range(c):
                mask = masks[i, j:j+1]
                # Dilate
                dilated = torch.clamp(
                    F.conv2d(mask, kernel, padding=kernel_size//2), 0, 1
                )
                # Erode
                eroded = 1.0 - torch.clamp(
                    F.conv2d(1.0 - mask, kernel, padding=kernel_size//2), 0, 1
                )
                # Boundary is difference between dilation and erosion
                boundary = torch.clamp(dilated - eroded, 0, 1)
                boundaries[i, j] = boundary
        
        return boundaries
```

2. Update the metrics configuration to include your metric:

```yaml
# configs/metrics/segmentation.yaml
# List of metrics to evaluate
metrics:
  # ... existing metrics
  
  # Your custom metric
  - alias: "custom_f1"  # Shorthand name for the metric
    path: "metrics.my_custom_metric"
    class: "MyCustomMetric"
    params:
      boundary_weight: 2.0
      eps: 0.000001
```

3. The metrics will automatically be loaded and used during validation and testing.

## Configuration

SegLab uses a hierarchical configuration system with YAML files:

1. **Main configuration** (`configs/main.yaml`): References all sub-configs and sets high-level parameters:
   - Output directory
   - Trainer settings (epochs, validation frequency)
   - Optimizer configuration

2. **Dataset configuration** (`configs/dataset/*.yaml`): Defines the dataset details:
   - Paths to data
   - Batch sizes
   - Augmentation settings
   - Patch size and sampling strategy

3. **Model configuration** (`configs/model/*.yaml`): Specifies the model architecture:
   - Path to the model module
   - Class name
   - Model-specific parameters

4. **Loss configuration** (`configs/loss/*.yaml`): Defines the loss function(s):
   - Primary loss
   - Optional secondary loss
   - Mixing parameters

5. **Metrics configuration** (`configs/metrics/*.yaml`): Lists evaluation metrics:
   - Metric aliases
   - Paths to metric implementations
   - Metric parameters

6. **Inference configuration** (`configs/inference/*.yaml`): Configures inference:
   - Patch size for validation/testing
   - Patch margin
   - Test-time augmentation settings

## Training and Evaluation

Basic training with the framework:

```bash
# Start training with default main config
python train.py --config configs/main.yaml

# Use a different main config
python train.py --config configs/my_experiment.yaml

# Run testing on the best checkpoint
python train.py --config configs/main.yaml --test
```

## Resuming Training

Resume training from a checkpoint:

```bash
# Resume from the last checkpoint
python train.py --config configs/main.yaml --resume outputs/experiment_1/checkpoints/last.ckpt

# Resume from a specific metric checkpoint
python train.py --config configs/main.yaml --resume outputs/experiment_1/checkpoints/best_dice.ckpt
```

## Advanced Features

1. **Chunked inference**: Process large images in overlapping patches
2. **Mixed precision training**: Speed up training with automatic mixed precision
3. **Learning rate scheduling**: Configure various learning rate schedules
4. **Distributed training**: Automatic multi-GPU training support
5. **TensorBoard integration**: Visualize training progress, metrics, and predictions
6. **Flexible checkpoint management**: Save best checkpoints for each metric

---

This framework is designed to be modular and easily extensible. If you encounter any issues or have questions, please open an issue on GitHub.