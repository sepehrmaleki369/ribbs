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

# New imports for snake losses and graph loading
from losses_n import SnakeFastLoss, SnakeSimpleLoss
from utils.graphs import load_training_graph_by_id
import networkx as nx


def save_evolved_graphs_correctly(snake_loss_obj, batch_data, epoch, output_dir='graphs/evolved'):
    """
    CORRECTLY save evolved snake graphs for each sample in the batch.
    
    Args:
        snake_loss_obj: The snake loss object after optimization
        batch_data: The batch data containing correct sample IDs
        epoch: Current training epoch
        output_dir: Directory to save evolved graphs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the loss function has the fixed attributes
    if not hasattr(snake_loss_obj, 'snakes') or not hasattr(snake_loss_obj, 'snake_sample_ids'):
        print("⚠️ Warning: Loss function doesn't have fixed attributes. Make sure to apply the losses.py patch first!")
        return
    
    # Get the correct sample IDs from the batch
    sample_ids = batch_data.get('sample_ids', [])
    if not sample_ids:
        print("⚠️ No sample IDs found in batch data")
        return
    
    print(f"🔍 Saving evolved graphs for {len(sample_ids)} samples: {sample_ids}")
    
    # Save each evolved graph with its correct sample ID
    # FIXED: Use batch sample IDs instead of snake sample IDs
    for i, snake in enumerate(snake_loss_obj.snakes):
        if snake is not None and i < len(sample_ids):
            # Get the correct sample ID from the batch
            correct_sample_id = sample_ids[i]
            
            # Create filename with correct sample ID
            filename = f"evolved_sample_{correct_sample_id}_epoch_{epoch}.pkl"
            filepath = os.path.join(output_dir, filename)
            
            try:
                # Save the evolved graph
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(snake, f)
                print(f"✅ Saved evolved graph for sample {correct_sample_id} at epoch {epoch}")
                
            except Exception as e:
                print(f"❌ Error saving evolved graph for sample {correct_sample_id}: {e}")
        else:
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
    learning_rate = config.get('learning_rate', 0.001)
    
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
        n_levels=2,        # Changed: from 3 to 2 levels for 584x565 images
        dropout=0.1,
        norm_type='batch',
        upsampling='bilinear',
        pooling="max",
        three_dimensional=False,
        apply_final_relu=False
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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
    
    # Create datasets and dataloaders
    train_dataset = RegressionDataset(config, split='train')
    val_dataset = RegressionDataset(config, split='valid')
    
    # FIXED: Custom collate function to preserve sample IDs
    def collate_fn(batch):
        """Custom collate function to preserve sample IDs."""
        images = torch.stack([item['image'] for item in batch])
        targets = torch.stack([item['distance_map'] for item in batch])
        
        # Handle missing label_graphs - create empty list for now
        # We'll load the actual graphs during training
        label_graphs = []  # Will be populated during training
        
        # Extract sample IDs
        sample_ids = [item['sample_id'] for item in batch]
        
        return {
            'image': images,
            'distance_map': targets,
            'label_graphs': label_graphs,  # Empty list, will be filled during training
            'sample_ids': sample_ids  # Include sample IDs in batch
        }
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
        collate_fn=collate_fn  # FIXED: Use custom collate function
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.get('val_batch_size', 1), 
        shuffle=False, 
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
        collate_fn=collate_fn  # FIXED: Use custom collate function
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
    
    print(f"🔧 Debug level: {debug_level}")
    print(f"🔧 Snake loss verbose: {snake_loss_verbose}")
    print(f"🔧 Device checks: {device_checks}")
    print(f"🔧 Graph loading: {graph_loading}")
    print(f"🔧 Save evolved graphs: {save_evolved_graphs}")
    
    if use_snake_loss:
        print(f"🐍 Snake loss enabled: {snake_type}")
        print(f"🐍 Snake loss starts at epoch: {snake_start_epoch}")
        print(f"🐍 Evolved graphs will be saved to: {evolved_graphs_dir}")
        
        # Create snake loss object
        if snake_type == 'fast':
            snake_loss = SnakeFastLoss(
                ndims=snake_config.get('ndims', 2),  # Get from config or default to 2
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
            
            # Move to device properly for SnakeFastLoss
            if device.type == 'cuda':
                snake_loss = snake_loss.cuda()
                # Force move the filter tensor to CUDA
                snake_loss.fltrt = snake_loss.fltrt.cuda()
                if device_checks:
                    print(f"🔍 After cuda() - Filter device: {snake_loss.fltrt.device}")
                    print(f"🔍 iscuda flag: {snake_loss.iscuda}")
            else:
                snake_loss = snake_loss.to(device)
        else:  # simple
            snake_loss = SnakeSimpleLoss(
                ndims=snake_config.get('ndims', 2),  # Get from config or default to 2
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
            
            # Move to device properly for SnakeSimpleLoss
            if device.type == 'cuda':
                snake_loss = snake_loss.cuda()
                # Force move the filter tensor to CUDA
                snake_loss.fltrt = snake_loss.fltrt.cuda()
                if device_checks:
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
        
        if use_snake_this_epoch:
            print(f"Epoch {epoch}: using loss = Snake{snake_type.capitalize()}")
        else:
            print(f"Epoch {epoch}: using loss = MSE")
        
        # Training loop
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(train_pbar):
            images = batch['image'].to(device)
            targets = batch['distance_map'].to(device)
            
            # FIXED: Get correct sample IDs from batch
            sample_ids = batch['sample_ids']  # Now this will be a list of actual sample IDs
            
            # Debug level control
            if debug_level == "batch":
                print(f"🔍 Processing batch {batch_idx} with sample IDs: {sample_ids}")
            elif debug_level == "minimal":
                pass  # No batch-level output
            # epoch level will be handled in epoch summary
            
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
            
            # FIXED: Remove padding for loss calculation using original dimensions
            if pad_h > 0 or pad_w > 0:
                outputs = outputs[:, :, :original_h, :original_w]
                targets = targets[:, :original_h, :original_w]  # targets has 3 dimensions, not 4
            
            # Calculate loss
            if use_snake_this_epoch:
                # FIXED: Load graph data for snake loss using correct sample IDs
                lbl_graphs = []
                for sample_id in sample_ids:
                    try:
                        graph = load_training_graph_by_id(str(sample_id))
                        if graph is not None:
                            lbl_graphs.append(graph)
                            if graph_loading:
                                print(f"✅ Loaded graph for sample {sample_id}")
                        else:
                            if graph_loading:
                                print(f"⚠️ No graph found for sample {sample_id}")
                            lbl_graphs.append(None)
                    except Exception as e:
                        if graph_loading:
                            print(f"⚠️ Could not load graph for sample {sample_id}: {e}")
                        # Create a dummy graph if needed
                        lbl_graphs.append(None)
                
                # Debug: Check device status before snake loss
                if device_checks:
                    print(f"🔍 outputs device: {outputs.device}")
                    print(f"🔍 snake_loss.fltrt device: {snake_loss.fltrt.device}")
                    print(f"🔍 Device match: {outputs.device == snake_loss.fltrt.device}")
                
                # Ensure filter is on the same device as outputs
                if outputs.device != snake_loss.fltrt.device:
                    if device_checks:
                        print(f"⚠️ Device mismatch detected! Moving filter from {snake_loss.fltrt.device} to {outputs.device}")
                    snake_loss.fltrt = snake_loss.fltrt.to(outputs.device)
                
                # FIXED: Calculate snake loss - the loss function now stores all snakes
                if snake_loss_verbose:
                    loss = snake_loss(outputs, lbl_graphs)
                else:
                    # Temporarily disable print statements in snake loss
                    import sys
                    from io import StringIO
                    
                    # Redirect stdout to suppress print statements
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    
                    try:
                        loss = snake_loss(outputs, lbl_graphs)
                    finally:
                        # Restore stdout
                        sys.stdout = old_stdout
                
                # FIXED: Save evolved graphs every epoch (or every 10 epochs if you prefer)
                if save_evolved_graphs and epoch % 10 == 0:
                    try:
                        save_evolved_graphs_correctly(snake_loss, batch, epoch, evolved_graphs_dir)
                    except Exception as e:
                        print(f"⚠️ Could not save evolved graphs: {e}")
            else:
                loss = criterion(outputs, targets.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Epoch-level debugging
        if debug_level in ["epoch", "batch"]:
            print(f"🔍 Epoch {epoch} Summary:")
            print(f"🔍 Average train loss: {avg_train_loss:.4f}")
            if use_snake_this_epoch:
                print(f"🔍 Used snake loss: {snake_type}")
            else:
                print(f"🔍 Used MSE loss")
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Valid]"):
                images = batch['image'].to(device)
                targets = batch['distance_map'].to(device)
                
                # Store original dimensions for proper padding removal
                original_h, original_w = images.shape[2], images.shape[3]
                
                # Add padding to make dimensions divisible by 4 (for UNet with 2 levels)
                h, w = images.shape[2], images.shape[3]
                pad_h = (4 - h % 4) % 4
                pad_w = (4 - w % 4) % 4
                
                if pad_h > 0 or pad_w > 0:
                    images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
                    targets = torch.nn.functional.pad(targets, (0, pad_w, 0, pad_h), mode='reflect')
                
                outputs = model(images)
                
                # FIXED: Remove padding for loss calculation using original dimensions
                if pad_h > 0 or pad_w > 0:
                    outputs = outputs[:, :, :original_h, :original_w]
                    targets = targets[:, :original_h, :original_w]  # targets has 3 dimensions, not 4
                
                # Debug: Check tensor shapes
                print(f"🔍 Validation - outputs shape: {outputs.shape}, targets shape: {targets.shape}")
                
                if use_snake_this_epoch:
                    # For validation, use MSE loss to compare fairly
                    val_loss = criterion(outputs, targets.unsqueeze(1))
                else:
                    val_loss = criterion(outputs, targets.unsqueeze(1))
                
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Print epoch summary
        print(f"Epoch {epoch}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint every 50 epochs
        if epoch % 10 == 0:
            checkpoint_path = f'checkpoints_regression/checkpoint_epoch_{epoch}.pth'
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
            }, checkpoint_path)
            
            print(f"✅ Saved checkpoint: {checkpoint_path}")
        
        # Save best model
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
    
    print(f"\n=== TRAINING COMPLETED ===")
    print(f"✅ Total epochs: {num_epochs}")
    print(f"✅ Started from epoch: {start_epoch}")
    print(f"✅ Snake loss enabled: {use_snake_loss}")
    if use_snake_loss:
        print(f"✅ Snake loss type: {snake_type}")
        print(f"✅ Snake loss started at epoch: {snake_type}")
        print(f"✅ Evolved graphs saved to: {evolved_graphs_dir}")
    print(f"✅ Best validation loss: {best_val_loss:.4f}")
    print(f"✅ Checkpoints saved every 50 epochs")
    print(f"✅ Best model saved to: checkpoints_regression/best_model.pth")


if __name__ == "__main__":
    train_regression()
