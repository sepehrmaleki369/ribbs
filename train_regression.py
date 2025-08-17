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


def save_snake_evolution_graphs(snake_loss_obj, sample_ids, epoch, output_dir='graphs/evolved'):
    """
    Extract and save evolved snake graphs after optimization.
    Updated to work with the actual snake loss object structure.
    
    Args:
        snake_loss_obj: The snake loss object after optimization
        sample_ids: List of sample IDs for this batch
        epoch: Current training epoch
        output_dir: Directory to save evolved graphs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Try to access the snake attribute which contains the evolved graphs
        if hasattr(snake_loss_obj, 'snake'):
            snake_data = snake_loss_obj.snake
            
            print(f"✅ Found snake data, type: {type(snake_data)}")
            
            # The snake data might be a list or contain multiple graphs
            if isinstance(snake_data, list):
                graphs_list = snake_data
                print(f"✅ Snake data is a list with {len(graphs_list)} items")
            elif hasattr(snake_data, '__len__'):
                graphs_list = list(snake_data)
                print(f"✅ Snake data has length {len(graphs_list)}")
            else:
                graphs_list = [snake_data]
                print(f"✅ Snake data is a single item")
            
            # Save each evolved graph
            for i, sample_id in enumerate(sample_ids):
                if i < len(graphs_list):
                    evolved_graph = graphs_list[i]
                    
                    # Try to convert to NetworkX format
                    try:
                        if hasattr(evolved_graph, 'to_networkx'):
                            nx_graph = evolved_graph.to_networkx()
                            print(f"✅ Converted using to_networkx()")
                        elif hasattr(evolved_graph, 'get_graph'):
                            nx_graph = evolved_graph.get_graph()
                            print(f"✅ Converted using get_graph()")
                        elif hasattr(evolved_graph, 'graph'):
                            nx_graph = evolved_graph.graph
                            print(f"✅ Converted using .graph attribute")
                        elif isinstance(evolved_graph, nx.Graph):
                            nx_graph = evolved_graph
                            print(f"✅ Already a NetworkX graph")
                        else:
                            # If we can't convert, save the raw data as pickle
                            print(f"⚠️ Cannot convert graph format, saving raw data")
                            filename = f"evolved_sample_{sample_id}_epoch_{epoch}.pkl"
                            filepath = os.path.join(output_dir, filename)
                            
                            import pickle
                            with open(filepath, 'wb') as f:
                                pickle.dump(evolved_graph, f)
                            print(f"✅ Saved raw snake data: {filename}")
                            continue
                        
                        # Save the NetworkX graph
                        filename = f"evolved_sample_{sample_id}_epoch_{epoch}.graph"
                        filepath = os.path.join(output_dir, filename)
                        nx.write_gpickle(nx_graph, filepath)
                        print(f"✅ Saved evolved graph: {filename}")
                        
                    except Exception as e:
                        print(f"⚠️ Error processing graph {i} for sample {sample_id}: {e}")
                        # Fallback: save raw data
                        try:
                            filename = f"evolved_sample_{sample_id}_epoch_{epoch}_raw.pkl"
                            filepath = os.path.join(output_dir, filename)
                            import pickle
                            with open(filepath, 'wb') as f:
                                pickle.dump(evolved_graph, f)
                            print(f"✅ Saved raw data as fallback: {filename}")
                        except Exception as e2:
                            print(f"❌ Failed to save even raw data: {e2}")
                        continue
                        
                else:
                    print(f"⚠️ No evolved graph available for sample {sample_id}")
                    
        else:
            print(f"⚠️ Snake loss object has no 'snake' attribute")
            print(f"Available attributes: {[attr for attr in dir(snake_loss_obj) if not attr.startswith('_')]}")
            
            # Try alternative approaches
            if hasattr(snake_loss_obj, 'getGraph'):
                print(f"✅ Found getGraph() method, trying to use it")
                try:
                    evolved_graphs = snake_loss_obj.getGraph()
                    # Process evolved_graphs similar to above
                    # ... (similar logic for processing)
                except Exception as e:
                    print(f"⚠️ getGraph() method failed: {e}")
            
    except Exception as e:
        print(f"⚠️ Error saving evolved graphs: {e}")
        print(f"Snake loss object type: {type(snake_loss_obj)}")
        print(f"Available attributes: {[attr for attr in dir(snake_loss_obj) if not attr.startswith('_')]}")


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
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.get('val_batch_size', 1), 
        shuffle=False, 
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False)
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
    
    if use_snake_loss:
        print(f"�� Snake loss enabled: {snake_type}")
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
            sample_ids = batch.get('sample_id', [f"batch_{batch_idx}_sample_{i}" for i in range(len(images))])
            
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
            
            # Remove padding for loss calculation
            if pad_h > 0 or pad_w > 0:
                outputs = outputs[:, :, :h, :w]
                targets = targets[:, :h, :w]  # targets has 3 dimensions, not 4
            
            # Calculate loss
            if use_snake_this_epoch:
                # Load graph data for snake loss
                lbl_graphs = []
                for sample_id in sample_ids:
                    try:
                        graph = load_training_graph_by_id(str(sample_id))
                        lbl_graphs.append(graph)
                    except Exception as e:
                        print(f"⚠️ Could not load graph for sample {sample_id}: {e}")
                        # Create a dummy graph if needed
                        lbl_graphs.append(None)
                
                # Debug: Check device status before snake loss
                print(f"🔍 outputs device: {outputs.device}")
                print(f"🔍 snake_loss.fltrt device: {snake_loss.fltrt.device}")
                print(f"🔍 Device match: {outputs.device == snake_loss.fltrt.device}")
                
                # Ensure filter is on the same device as outputs
                if outputs.device != snake_loss.fltrt.device:
                    print(f"⚠️ Device mismatch detected! Moving filter from {snake_loss.fltrt.device} to {outputs.device}")
                    snake_loss.fltrt = snake_loss.fltrt.to(outputs.device)
                
                # Calculate snake loss
                loss = snake_loss(outputs, lbl_graphs)
                
                # Save evolved graphs every 10 epochs
                if epoch % 10 == 0:
                    try:
                        save_snake_evolution_graphs(snake_loss, sample_ids, epoch, evolved_graphs_dir)
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
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Valid]"):
                images = batch['image'].to(device)
                targets = batch['distance_map'].to(device)
                
                # Add padding to make dimensions divisible by 4 (for UNet with 2 levels)
                h, w = images.shape[2], images.shape[3]
                pad_h = (4 - h % 4) % 4
                pad_w = (4 - w % 4) % 4
                
                if pad_h > 0 or pad_w > 0:
                    images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
                    targets = torch.nn.functional.pad(targets, (0, pad_w, 0, pad_h), mode='reflect')
                
                outputs = model(images)
                
                # Remove padding for loss calculation
                if pad_h > 0 or pad_w > 0:
                    outputs = outputs[:, :, :h, :w]
                    targets = targets[:, :h, :w]  # targets has 3 dimensions, not 4
                
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
        if epoch % 50 == 0:
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
        print(f"✅ Snake loss started at epoch: {snake_start_epoch}")
        print(f"✅ Evolved graphs saved to: {evolved_graphs_dir}")
    print(f"✅ Best validation loss: {best_val_loss:.4f}")
    print(f"✅ Checkpoints saved every 50 epochs")
    print(f"✅ Best model saved to: checkpoints_regression/best_model.pth")


if __name__ == "__main__":
    train_regression()