import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from core.regression_dataset import RegressionDataset
from models.base_models import UNet
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import sys
from io import StringIO
from contextlib import contextmanager

# New imports for snake losses and graph loading
from losses_n import SnakeFastLoss, SnakeSimpleLoss
from utils.graphs import load_training_graph_by_id
import networkx as nx


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout prints"""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


def save_evolved_graphs_correctly(snake_loss_obj, batch_data, epoch, output_dir='graphs/evolved', debug_level='batch'):
    """
    CORRECTLY save evolved snake graphs for each sample in the batch.
    
    Args:
        snake_loss_obj: The snake loss object after optimization
        batch_data: The batch data containing correct sample IDs
        epoch: Current training epoch
        output_dir: Directory to save evolved graphs
        debug_level: Debug level for controlling output
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the loss function has the fixed attributes
    if not hasattr(snake_loss_obj, 'snakes') or not hasattr(snake_loss_obj, 'snake_sample_ids'):
        if debug_level == "batch":
            print("⚠️ Warning: Loss function doesn't have fixed attributes. Make sure to apply the losses.py patch first!")
        return
    
    # Get the correct sample IDs from the batch
    sample_ids = batch_data.get('sample_ids', [])
    if not sample_ids:
        if debug_level == "batch":
            print("⚠️ No sample IDs found in batch data")
        return
    
    if debug_level == "batch":
        print(f"🔍 Saving evolved graphs for {len(sample_ids)} samples: {sample_ids}")
    
    # Save each evolved graph with its correct sample ID
    for i, snake in enumerate(snake_loss_obj.snakes):
        if snake is not None and i < len(sample_ids):
            correct_sample_id = sample_ids[i]
            filename = f"evolved_sample_{correct_sample_id}_epoch_{epoch}.pkl"
            filepath = os.path.join(output_dir, filename)
            
            try:
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(snake, f)
                if debug_level in ["batch", "epoch"]:
                    print(f"✅ Saved evolved graph for sample {correct_sample_id} at epoch {epoch}")
                
            except Exception as e:
                if debug_level in ["batch", "epoch"]:
                    print(f"❌ Error saving evolved graph for sample {correct_sample_id}: {e}")
        else:
            if debug_level == "batch":
                if i < len(sample_ids):
                    print(f"⚠️ No evolved graph available for sample {sample_ids[i]}")
                else:
                    print(f"⚠️ No evolved graph available for index {i}")


def train_regression():
    """Train regression model for distance map prediction with optional Snake losses"""
    
    # Load configuration
    with open('configs/dataset/drive_regression.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract training parameters
    num_epochs = config['training']['num_epochs']
    batch_size = config['train_batch_size']
    # Optimizer configuration with robust fallbacks
    optimizer_cfg = config['training'].get('optimizer', {})
    learning_rate = optimizer_cfg.get('lr', config['training'].get('learning_rate', config.get('learning_rate', 0.001)))
    weight_decay = optimizer_cfg.get('weight_decay', 0.0)
    lr_decay_enabled = optimizer_cfg.get('lr_decay', False)
    lr_decay_factor = optimizer_cfg.get('lr_decay_factor', 0.1)
    
    # Check if we should resume from checkpoint
    resume_checkpoint = config['training'].get('resume_checkpoint', None)
    start_epoch = 1
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = UNet(
        in_channels=3,
        m_channels=64,
        out_channels=1,
        n_convs=2,
        n_levels=2,
        dropout=0.1,
        norm_type='batch',
        upsampling='bilinear',
        pooling="max",
        three_dimensional=False,
        apply_final_relu=False
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Load checkpoint if specified
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint from: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 1) + 1
        else:
            model.load_state_dict(checkpoint)
            start_epoch = 1
            
        print(f"✅ Resumed from checkpoint, starting from epoch {start_epoch}")
    else:
        print("Starting training from scratch")
    
    # Learning rate scheduler (ReduceLROnPlateau) - configured via YAML
    scheduler = None
    print(f"🔧 LR Decay Configuration:")
    print(f"  Enabled: {lr_decay_enabled}")
    print(f"  Factor: {lr_decay_factor}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Weight Decay: {weight_decay}")
    print("")
    if lr_decay_enabled:
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=lr_decay_factor,
                patience=2,
                min_lr=1e-7
            )
            if lr_decay_factor <= 0 or lr_decay_factor >= 1:
                if lr_decay_factor >= 1:
                    print("⚠️ lr_decay_factor >= 1.0; LR will not decrease.")
                else:
                    print("⚠️ lr_decay_factor <= 0; invalid factor.")
            print(f"✅ LR Scheduler created successfully")
            print(f"  Type: ReduceLROnPlateau")
            print(f"  Mode: min")
            print(f"  Patience: 2")
            print(f"  Min LR: 1e-7")
            print("")
        except Exception as e:
            print(f"⚠️ Could not create LR scheduler: {e}")
    else:
        print("❌ LR Decay is DISABLED - no scheduler will be created")
        print("")
    
    # Create datasets and dataloaders
    train_dataset = RegressionDataset(config, split='train')
    val_dataset = RegressionDataset(config, split='valid')
    
    # Custom collate function to preserve sample IDs
    def collate_fn(batch):
        """Custom collate function to preserve sample IDs."""
        images = torch.stack([item['image'] for item in batch])
        targets = torch.stack([item['distance_map'] for item in batch])
        
        # Handle missing label_graphs - create empty list for now
        label_graphs = []  # Will be populated during training
        
        # Extract sample IDs
        sample_ids = [item['sample_id'] for item in batch]
        
        return {
            'image': images,
            'distance_map': targets,
            'label_graphs': label_graphs,
            'sample_ids': sample_ids
        }
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.get('val_batch_size', 1), 
        shuffle=False, 
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
        collate_fn=collate_fn
    )
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Snake loss configuration
    snake_config = config.get('snake_loss', {})
    use_snake_loss = snake_config.get('enabled', False)
    snake_start_epoch = snake_config.get('start_epoch', 1)
    snake_type = snake_config.get('type', 'simple')
    evolved_graphs_dir = snake_config.get('evolved_graphs_dir', 'graphs/evolved')
    
    # Debug configuration
    debug_config = config.get('debug', {})
    debug_level = debug_config.get('level', 'batch')  # batch, epoch, minimal
    snake_loss_verbose = debug_config.get('snake_loss_verbose', True)
    device_checks = debug_config.get('device_checks', True)
    graph_loading = debug_config.get('graph_loading', True)
    save_evolved_graphs = debug_config.get('save_evolved_graphs', True)
    
    # Only show debug config for batch level and above
    if debug_level in ["batch", "epoch"]:
        print(f"🔧 Debug level: {debug_level}")
        print(f"🔧 Snake loss verbose: {snake_loss_verbose}")
        print(f"🔧 Device checks: {device_checks}")
        print(f"🔧 Graph loading: {graph_loading}")
        print(f"🔧 Save evolved graphs: {save_evolved_graphs}")
    
    if use_snake_loss:
        if debug_level in ["batch", "epoch"]:
            print(f"🐍 Snake loss enabled: {snake_type}")
            print(f"🐍 Snake loss starts at epoch: {snake_start_epoch}")
            print(f"🐍 Evolved graphs will be saved to: {evolved_graphs_dir}")
        
        # Always show snake loss configuration when training starts (regardless of debug level)
        print(f"\n🐍 Snake Loss Configuration:")
        print(f"  Type: {snake_type}")
        print(f"  Start epoch: {snake_start_epoch}")
        print(f"  NDims: {snake_config.get('ndims', 2)}")
        print(f"  Step size: {snake_config.get('stepsz', 0.1)}")
        print(f"  Alpha (elasticity): {snake_config.get('alpha', 0.0001)}")
        print(f"  Beta (curvature): {snake_config.get('beta', 0.01)}")
        print(f"  Filter std: {snake_config.get('fltrstdev', 0.5)}")
        print(f"  N steps: {snake_config.get('nsteps', 5)}")
        print(f"  Crop size: {snake_config.get('cropsz', [16, 16])}")
        print(f"  D max: {snake_config.get('dmax', 15.0)}")
        print(f"  Max edge length: {snake_config.get('maxedgelen', 3.0)}")
        print(f"  External grad factor: {snake_config.get('extgradfac', 1.0)}")
        print("")
        
        # Create snake loss object
        if snake_type == 'fast':
            snake_loss = SnakeFastLoss(
                ndims=snake_config.get('ndims', 2),
                stepsz=snake_config.get('stepsz', 0.1),
                alpha=snake_config.get('alpha', 0.0001),
                beta=snake_config.get('beta', 0.01),
                fltrstdev=snake_config.get('fltrstdev', 0.5),
                nsteps=snake_config.get('nsteps', 10),
                cropsz=snake_config.get('cropsz', [32, 32]),
                dmax=snake_config.get('dmax', 15.0),
                maxedgelen=snake_config.get('maxedgelen', 5.0),
                extgradfac=snake_config.get('extgradfac', 2.0)
            )
            
            if device.type == 'cuda':
                snake_loss = snake_loss.cuda()
                snake_loss.fltrt = snake_loss.fltrt.cuda()
                if device_checks and debug_level == "batch":
                    print(f"🔍 After cuda() - Filter device: {snake_loss.fltrt.device}")
                    print(f"🔍 iscuda flag: {snake_loss.iscuda}")
            else:
                snake_loss = snake_loss.to(device)
        else:  # simple
            snake_loss = SnakeSimpleLoss(
                ndims=snake_config.get('ndims', 2),
                stepsz=snake_config.get('stepsz', 0.1),
                alpha=snake_config.get('alpha', 0.0001),
                beta=snake_config.get('beta', 0.01),
                fltrstdev=snake_config.get('fltrstdev', 0.5),
                nsteps=snake_config.get('nsteps', 5),
                cropsz=snake_config.get('cropsz', [16, 16]),
                dmax=snake_config.get('dmax', 15.0),
                maxedgelen=snake_config.get('maxedgelen', 3.0),
                extgradfac=snake_config.get('extgradfac', 1.0)
            )
            
            if device.type == 'cuda':
                snake_loss = snake_loss.cuda()
                snake_loss.fltrt = snake_loss.fltrt.cuda()
                if device_checks and debug_level == "batch":
                    print(f"🔍 After cuda() - Filter device: {snake_loss.fltrt.device}")
                    print(f"🔍 iscuda flag: {snake_loss.iscuda}")
            else:
                snake_loss = snake_loss.to(device)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        # Determine which loss to use this epoch
        use_snake_this_epoch = use_snake_loss and epoch >= snake_start_epoch
        
        # Only show loss type and current LR for epoch level and above
        if debug_level in ["batch", "epoch"]:
            if use_snake_this_epoch:
                print(f"Epoch {epoch}: using loss = Snake{snake_type.capitalize()}")
            else:
                print(f"Epoch {epoch}: using loss = MSE")
            try:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"🔧 Learning rate: {current_lr:.8f}")
            except Exception:
                pass
        
        # Training loop
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        
        # Initialize epoch gradient tracking
        total_gradient_norm = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_pbar):
            images = batch['image'].to(device)
            targets = batch['distance_map'].to(device)
            sample_ids = batch['sample_ids']
            
            # Debug level control - only show batch details for "batch" level
            if debug_level == "batch":
                print(f"🔍 Processing batch {batch_idx} with sample IDs: {sample_ids}")
            
            # Store original dimensions for proper padding removal
            original_h, original_w = images.shape[2], images.shape[3]
            
            # Add padding to make dimensions divisible by 4 (for UNet with 2 levels)
            h, w = images.shape[2], images.shape[3]
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            
            if pad_h > 0 or pad_w > 0:
                images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
                targets = torch.nn.functional.pad(targets, (0, pad_w, 0, pad_h), mode='reflect')
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Remove padding for loss calculation using original dimensions
            if pad_h > 0 or pad_w > 0:
                outputs = outputs[:, :, :original_h, :original_w]
                targets = targets[:, :original_h, :original_w]
            
            # Calculate loss
            if use_snake_this_epoch:
                # Load graph data for snake loss using correct sample IDs
                lbl_graphs = []
                for sample_id in sample_ids:
                    try:
                        # Suppress graph loading prints based on debug level
                        if graph_loading and debug_level == "batch":
                            # Allow prints for batch level
                            graph = load_training_graph_by_id(str(sample_id))
                        else:
                            # Suppress prints for epoch and minimal levels
                            with suppress_stdout():
                                graph = load_training_graph_by_id(str(sample_id))
                        
                        if graph is not None:
                            lbl_graphs.append(graph)
                            # Only print our own graph loading info for batch level
                            if graph_loading and debug_level == "batch":
                                print(f"✅ Loaded graph for sample {sample_id}")
                        else:
                            if graph_loading and debug_level == "batch":
                                print(f"⚠️ No graph found for sample {sample_id}")
                            lbl_graphs.append(None)
                    except Exception as e:
                        if graph_loading and debug_level == "batch":
                            print(f"⚠️ Could not load graph for sample {sample_id}: {e}")
                        lbl_graphs.append(None)
                
                # Debug device status only for batch level
                if device_checks and debug_level == "batch":
                    print(f"🔍 outputs device: {outputs.device}")
                    print(f"🔍 snake_loss.fltrt device: {snake_loss.fltrt.device}")
                    print(f"🔍 Device match: {outputs.device == snake_loss.fltrt.device}")
                
                # Ensure filter is on the same device as outputs
                if outputs.device != snake_loss.fltrt.device:
                    if device_checks and debug_level == "batch":
                        print(f"⚠️ Device mismatch detected! Moving filter from {snake_loss.fltrt.device} to {outputs.device}")
                    snake_loss.fltrt = snake_loss.fltrt.to(outputs.device)
                
                # Calculate snake loss - handle verbose output properly
                if snake_loss_verbose and debug_level == "batch":
                    # Allow snake loss internal prints only for batch level
                    loss = snake_loss(outputs, lbl_graphs)
                else:
                    # Suppress snake loss internal prints
                    with suppress_stdout():
                        loss = snake_loss(outputs, lbl_graphs)
                
                # Save evolved graphs every 5 epochs with proper debug level
                if save_evolved_graphs and epoch % 5 == 0:
                    try:
                        save_evolved_graphs_correctly(snake_loss, batch, epoch, evolved_graphs_dir, debug_level)
                    except Exception as e:
                        if debug_level in ["batch", "epoch"]:
                            print(f"⚠️ Could not save evolved graphs: {e}")
            else:
                loss = criterion(outputs, targets.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            
            # FIXED: Add gradient monitoring and clipping for ALL training (MSE and Snake)
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            # Accumulate for epoch average
            total_gradient_norm += total_norm
            batch_count += 1
            
            # Debug level control for gradient info
            if debug_level in ["epoch", "batch"]:
                loss_type = "Snake" if use_snake_this_epoch else "MSE"
                # Only show batch-level gradient info for batch debug level
                if debug_level == "batch":
                    print(f"🔍 {loss_type} Gradient norm: {total_norm:.3f}")
            
            # Add gradient clipping to prevent explosion (for both MSE and Snake)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate average training loss and gradient norm
        avg_train_loss = total_train_loss / len(train_loader)
        avg_gradient_norm = total_gradient_norm / batch_count if batch_count > 0 else 0.0
        
        # Epoch-level debugging summary
        if debug_level in ["epoch", "batch"]:
            print(f"🔍 Epoch {epoch} Summary:")
            print(f"🔍 Average train loss: {avg_train_loss:.4f}")
            print(f"🔍 Average gradient norm: {avg_gradient_norm:.4f}")
            if use_snake_this_epoch:
                print(f"🔍 Used snake loss: {snake_type}")
            else:
                print(f"🔍 Used MSE loss")
        
        # Validation every 5 epochs
        avg_val_loss = None
        if epoch % 5 == 0:
            print(f"🔍 Running validation at epoch {epoch}...")
            model.eval()
            total_val_loss = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Valid]")
                for batch in val_pbar:
                    images = batch['image'].to(device)
                    targets = batch['distance_map'].to(device)
                    
                    # Store original dimensions for proper padding removal
                    original_h, original_w = images.shape[2], images.shape[3]
                    
                    # Add padding to make dimensions divisible by 4
                    h, w = images.shape[2], images.shape[3]
                    pad_h = (4 - h % 4) % 4
                    pad_w = (4 - w % 4) % 4
                    
                    if pad_h > 0 or pad_w > 0:
                        images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
                        targets = torch.nn.functional.pad(targets, (0, pad_w, 0, pad_h), mode='reflect')
                    
                    outputs = model(images)
                    
                    # Remove padding for loss calculation
                    if pad_h > 0 or pad_w > 0:
                        outputs = outputs[:, :, :original_h, :original_w]
                        targets = targets[:, :original_h, :original_w]
                    
                    # Debug tensor shapes only for batch level
                    if debug_level == "batch":
                        print(f"🔍 Validation - outputs shape: {outputs.shape}, targets shape: {targets.shape}")
                    
                    # Always use MSE loss for validation
                    val_loss = criterion(outputs, targets.unsqueeze(1))
                    total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            # Print epoch summary with validation
            print(f"Epoch {epoch}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Step LR scheduler on validation result
            if scheduler is not None:
                try:
                    old_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(avg_val_loss)
                    new_lr = optimizer.param_groups[0]['lr']
                    if old_lr != new_lr:
                        print(f"🔧 LR CHANGED: {old_lr:.8f} → {new_lr:.8f}")
                        print(f"🔧 Scheduler reduced LR due to validation plateau")
                    else:
                        print(f"🔧 LR unchanged: {old_lr:.8f} (validation improving)")
                except Exception as e:
                    print(f"⚠️ Scheduler step failed: {e}")
            else:
                print("⚠️ No scheduler available - LR will not change")
            
            # Save best model only when we have validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = 'checkpoints_regression/best_model.pth'
                os.makedirs('checkpoints_regression', exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'snake_loss_enabled': use_snake_loss,
                    'snake_loss_type': snake_type if use_snake_loss else None,
                    'evolved_graphs_dir': evolved_graphs_dir if use_snake_loss else None
                }, best_model_path)
                
                print(f"✅ Saved best model: {best_model_path}")
        else:
            # Print epoch summary without validation
            print(f"Epoch {epoch}/{num_epochs}: Train Loss: {avg_train_loss:.4f}")
        
        # Save checkpoint every 5 epochs (same schedule as validation)
        if epoch % 10 == 0:
            checkpoint_path = f'checkpoints_regression/checkpoint_epoch_{epoch}.pth'
            os.makedirs('checkpoints_regression', exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss if avg_val_loss is not None else None,
                'snake_loss_enabled': use_snake_loss,
                'snake_loss_type': snake_type if use_snake_loss else None,
                'evolved_graphs_dir': evolved_graphs_dir if use_snake_loss else None
            }, checkpoint_path)
            
            print(f"✅ Saved checkpoint: {checkpoint_path}")
    
    print(f"\n=== TRAINING COMPLETED ===")
    print(f"✅ Total epochs: {num_epochs}")
    print(f"✅ Started from epoch: {start_epoch}")
    print(f"✅ Snake loss enabled: {use_snake_loss}")
    if use_snake_loss:
        print(f"✅ Snake loss type: {snake_type}")
        print(f"✅ Snake loss started at epoch: {snake_start_epoch}")
        print(f"✅ Evolved graphs saved to: {evolved_graphs_dir}")
    print(f"✅ Best validation loss: {best_val_loss:.4f}")
    print(f"✅ Checkpoints saved every 5 epochs")
    print(f"✅ Best model saved to: checkpoints_regression/best_model.pth")


if __name__ == "__main__":
    train_regression()