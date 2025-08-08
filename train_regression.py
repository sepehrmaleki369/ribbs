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


def train_regression():
    """Train regression model for distance map prediction with optional Snake losses"""
    
    print("=== TRAINING REGRESSION MODEL ===")
    
    # Load configuration
    with open('configs/dataset/drive_regression.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Training schedule config
    training_cfg = config.get('training', {})
    total_epochs = int(training_cfg.get('num_epochs', 200))
    resume_checkpoint = training_cfg.get('resume_checkpoint', None)
    additional_epochs = training_cfg.get('additional_epochs', None)
    
    # Snake loss config (fallback defaults)
    snake_cfg = config.get('snake_loss', {})
    use_snake = bool(snake_cfg.get('enabled', False))
    snake_type = snake_cfg.get('type', 'fast')  # 'fast' or 'simple'
    snake_epoch_start = int(snake_cfg.get('start_epoch', 99999))
    snake_params = {
        'stepsz': float(snake_cfg.get('stepsz', 0.5)),
        'alpha': float(snake_cfg.get('alpha', 0.1)),
        'beta': float(snake_cfg.get('beta', 0.1)),
        'fltrstdev': float(snake_cfg.get('fltrstdev', 2.0)),
        'ndims': 2,
        'nsteps': int(snake_cfg.get('nsteps', 10)),
        'cropsz': snake_cfg.get('cropsz', [128, 128]),
        'dmax': float(snake_cfg.get('dmax', 155.0)),
        'maxedgelen': float(snake_cfg.get('maxedgelen', 10.0)),
        'extgradfac': float(snake_cfg.get('extgradfac', 1.0)),
    }
    pretrained_checkpoint = snake_cfg.get('pretrained_checkpoint', None)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = RegressionDataset(config, split='train')
    val_dataset = RegressionDataset(config, split='valid')
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('train_batch_size', 1),  # Use config value, default to 1
        shuffle=True,
        num_workers=config.get('num_workers', 0),      # Use config value, default to 0
        pin_memory=config.get('pin_memory', False)     # Use config value, default to False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('val_batch_size', 1),    # Use config value, default to 1
        shuffle=False,
        num_workers=config.get('num_workers', 0),      # Use config value, default to 0
        pin_memory=config.get('pin_memory', False)     # Use config value, default to False
    )
    
    # Create model using existing UNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = UNet(
        in_channels=3,
        m_channels=32,        # Reduced from 64 to 32 to match baseline
        out_channels=1,
        n_convs=2,
        n_levels=3,           # Increased from 2 to 3 for better feature extraction
        dropout=0.1,
        norm_type='batch',
        upsampling='bilinear',
        pooling="max",
        three_dimensional=False,
        apply_final_relu=False
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Load pretrained or resume checkpoints as requested
    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Resuming full state from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("Optimizer state restored.")
            except Exception as e:
                print(f"Warning: could not restore optimizer state: {e}")
        start_epoch = int(ckpt.get('epoch', 0))
        print(f"Starting from epoch {start_epoch}")
    elif pretrained_checkpoint and os.path.exists(pretrained_checkpoint):
        print(f"Loading pretrained weights: {pretrained_checkpoint}")
        ckpt = torch.load(pretrained_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print("Pretrained model weights loaded.")
    
    # Determine end epoch (support continuing for N additional epochs)
    if additional_epochs is not None:
        end_epoch = start_epoch + int(additional_epochs)
    else:
        end_epoch = total_epochs
    if end_epoch <= start_epoch:
        raise ValueError(f"end_epoch ({end_epoch}) must be greater than start_epoch ({start_epoch}).")
    
    # Losses
    criterion = nn.MSELoss()
    if use_snake:
        if snake_type == 'fast':
            snake_loss = SnakeFastLoss(**snake_params)
        else:
            snake_loss = SnakeSimpleLoss(**snake_params)
        if device.type == 'cuda':
            snake_loss = snake_loss.cuda()
    
    # Create checkpoint directory
    local_checkpoint_dir = 'checkpoints_regression'
    os.makedirs(local_checkpoint_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nStarting training: epochs {start_epoch+1}..{end_epoch} (total target={total_epochs})")
    print(f"Checkpoints will be saved every 50 epochs")
    
    def save_checkpoint(epoch, model, optimizer, train_losses, val_losses, is_best=False):
        """Save checkpoint locally only"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        
        # Save locally
        if is_best:
            local_path = os.path.join(local_checkpoint_dir, 'best_model.pth')
        else:
            local_path = os.path.join(local_checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, local_path)
        print(f"✅ Checkpoint saved: {local_path}")
    
    for epoch in range(start_epoch, end_epoch):
        # Training
        model.train()
        epoch_num_print = epoch + 1
        use_snake_this_epoch = use_snake and (epoch_num_print) >= snake_epoch_start
        loss_name = (
            'SnakeFast' if (use_snake_this_epoch and snake_type == 'fast') else 
            ('SnakeSimple' if (use_snake_this_epoch and snake_type == 'simple') else 'MSE')
        )
        print(f"Epoch {epoch_num_print}: using loss = {loss_name}")
        
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch_num_print}/{end_epoch} [Train]')
        
        for batch in train_pbar:
            images = batch['image'].to(device)
            targets = batch['distance_map'].to(device)
            sample_ids = batch.get('sample_id')
            
            # Pad images to make them divisible by 4 (for 2 levels)
            h, w = images.shape[2], images.shape[3]
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            
            if pad_h > 0 or pad_w > 0:
                images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
                targets = torch.nn.functional.pad(targets, (0, pad_w, 0, pad_h), mode='reflect')
            
            # Forward pass
            outputs = model(images)
            
            # Remove padding from outputs and targets for loss calculation
            if pad_h > 0 or pad_w > 0:
                outputs = outputs[:, :, :h, :w]
                targets = targets[:, :h, :w]
            
            # Compute loss
            if use_snake_this_epoch:
                # Build lbl_graphs for this batch
                lbl_graphs = []
                for sid in sample_ids:
                    G = load_training_graph_by_id(str(sid))
                    lbl_graphs.append(G)
                loss = snake_loss(outputs, lbl_graphs)
            else:
                loss = criterion(outputs, targets.unsqueeze(1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation every 10 epochs (or when saving checkpoints)
        run_validation = (epoch_num_print % 10 == 0) or (epoch_num_print % 50 == 0)
        
        if run_validation:
            model.eval()
            val_loss = 0.0
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch_num_print}/{end_epoch} [Val]')
            
            with torch.no_grad():
                for batch in val_pbar:
                    images = batch['image'].to(device)
                    targets = batch['distance_map'].to(device)
                    
                    h, w = images.shape[2], images.shape[3]
                    pad_h = (4 - h % 4) % 4
                    pad_w = (4 - w % 4) % 4
                    
                    if pad_h > 0 or pad_w > 0:
                        images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
                        targets = torch.nn.functional.pad(targets, (0, pad_w, 0, pad_h), mode='reflect')
                    
                    outputs = model(images)
                    
                    if pad_h > 0 or pad_w > 0:
                        outputs = outputs[:, :, :h, :w]
                        targets = targets[:, :h, :w]
                    
                    vloss = criterion(outputs, targets.unsqueeze(1))
                    val_loss += vloss.item()
                    val_pbar.set_postfix({'Loss': f'{vloss.item():.4f}'})
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                save_checkpoint(epoch_num_print, model, optimizer, train_losses, val_losses, is_best=True)
            
            print(f"Epoch {epoch_num_print}/{end_epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch_num_print}/{end_epoch}: Train Loss: {avg_train_loss:.4f}")
        
        if (epoch_num_print % 50) == 0:
            save_checkpoint(epoch_num_print, model, optimizer, train_losses, val_losses, is_best=False)
    
    save_checkpoint(end_epoch, model, optimizer, train_losses, val_losses, is_best=False)
    
    # Plot curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        # Estimate validation epoch indices relative to start
        val_epochs = list(range(((start_epoch // 10) + 1) * 10, end_epoch + 1, 10))
        plt.plot(val_epochs[:len(val_losses)], val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('regression_training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved as regression_training_curves.png")
    
    print(f"\n✅ Regression training completed successfully!")
    print(f"📁 Checkpoints saved in: {local_checkpoint_dir}")
    print(f"🏆 Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_regression()