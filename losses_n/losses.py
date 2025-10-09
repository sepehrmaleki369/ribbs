import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as dist
from . import gradImSnake
import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

class MSELoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target, weights=None):
        loss = (pred-target).pow(2)
        if weights is not None:
            loss *= weights

        if self.ignore_index is not None:
            loss = loss[target!=self.ignore_index]

        return loss.mean()
    
class MAELoss(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target, weights=None):
        loss = torch.abs(pred-target)
        if weights is not None:
            loss *= weights

        if self.ignore_index is not None:
            loss = loss[target!=self.ignore_index]

        return loss.mean()

class SnakeFastLoss(nn.Module):
    def __init__(self, stepsz,alpha,beta,fltrstdev,ndims,nsteps,
                       cropsz,dmax,maxedgelen,extgradfac):
        super(SnakeFastLoss,self).__init__()
        self.stepsz = stepsz
        self.alpha = alpha
        self.beta = beta
        self.fltrstdev = fltrstdev
        self.ndims = ndims
        self.cropsz = cropsz
        self.dmax = dmax
        self.maxedgelen = maxedgelen
        self.extgradfac = extgradfac
        self.nsteps = nsteps

        self.fltr = gradImSnake.makeGaussEdgeFltr(self.fltrstdev,self.ndims)
        self.fltrt = torch.from_numpy(self.fltr).type(torch.float32)

        self.iscuda = False

    def cuda(self):
        super(SnakeFastLoss,self).cuda()
        self.fltrt = self.fltrt.cuda()
        self.iscuda = True
        return self

    def forward(self,pred_dmap,lbl_graphs,crops=None):
        print(f"🔍 === SNAKE FAST LOSS FORWARD START ===")
        print(f"🔍 Input pred_dmap shape: {pred_dmap.shape}")
        print(f"🔍 Input lbl_graphs length: {len(lbl_graphs)}")
        print(f"🔍 Self object: {self}")
        print(f"🔍 Self type: {type(self)}")
        print(f"🔍 Self dir before: {[attr for attr in dir(self) if not attr.startswith('_')]}")
        
        # Ensure filter is on the same device as input
        if pred_dmap.device != self.fltrt.device:
            self.fltrt = self.fltrt.to(pred_dmap.device)
            # Update iscuda flag based on actual device
            self.iscuda = pred_dmap.device.type == 'cuda'
    
        pred_ = pred_dmap
        
        # Final safety check before cmptGradIm
        if pred_.device != self.fltrt.device:
            # Force move one more time
            self.fltrt = self.fltrt.to(pred_.device)
        
        gimg = gradImSnake.cmptGradIm(pred_,self.fltrt)
        gimg *= self.extgradfac
        snake_dmap = []

        print(f"🔍 === BEFORE SNAKES INITIALIZATION ===")
        print(f"🔍 Has 'snakes' attribute: {hasattr(self, 'snakes')}")
        print(f"🔍 Has 'snake_sample_ids' attribute: {hasattr(self, 'snake_sample_ids')}")
        if hasattr(self, 'snakes'):
            print(f"🔍 Current snakes length: {len(self.snakes)}")
        if hasattr(self, 'snake_sample_ids'):
            print(f"🔍 Current snake_sample_ids length: {len(self.snake_sample_ids)}")

        # FIXED: Reset snakes list every forward pass (baseline approach)
        print(f"🔍 === RESETTING SNAKES LISTS ===")
        self.snakes = []  # Clear every batch (baseline behavior)
        self.snake_sample_ids = []  # Clear every batch (baseline behavior)
        print(f"🔍 Reset self.snakes: {self.snakes}")
        print(f"🔍 Reset self.snake_sample_ids: {self.snake_sample_ids}")
        print(f"🔍 Has 'snakes' after reset: {hasattr(self, 'snakes')}")
        print(f"🔍 Has 'snake_sample_ids' after reset: {hasattr(self, 'snake_sample_ids')}")

        print(f"🔍 === STARTING SNAKE PROCESSING LOOP ===")
        for i,lg in enumerate(zip(lbl_graphs,gimg)):
            print(f"🔍 Processing snake {i}")
            l = lg[0]
            g = lg[1]

            if crops:
                crop = crops[i]
            else:
                crop=[slice(0,s) for s in g.shape[1:]]
            
            # DEBUG: Show initial graph structure
            print(f"🔍 Initial graph {i}: {len(l.nodes())} nodes, {len(l.edges())} edges")
            if len(l.nodes()) > 0:
                initial_positions = np.array([l.nodes[node]['pos'] for node in l.nodes()])
                print(f"🔍 Initial node positions range: [{initial_positions.min():.2f}, {initial_positions.max():.2f}]")
                print(f"🔍 Initial node positions mean: {initial_positions.mean():.2f}")
            
            # Create NEW snake from oversampled graph (baseline approach)
            s = gradImSnake.GradImSnake(l,crop,self.stepsz,self.alpha,
                                      self.beta,self.ndims,g)
            if self.iscuda: s.cuda()

            # DEBUG: Show snake parameters
            print(f"🔍 Snake {i} parameters: stepsz={self.stepsz}, alpha={self.alpha}, beta={self.beta}, nsteps={self.nsteps}")

            s.optim(self.nsteps)

            # DEBUG: Show evolved graph structure
            evolved_graph = s.getGraph()
            if evolved_graph:
                print(f"🔍 Evolved graph {i}: {len(evolved_graph.nodes())} nodes, {len(evolved_graph.edges())} edges")
                if len(evolved_graph.nodes()) > 0:
                    evolved_positions = np.array([evolved_graph.nodes[node]['pos'] for node in evolved_graph.nodes()])
                    print(f"🔍 Evolved node positions range: [{evolved_positions.min():.2f}, {evolved_positions.max():.2f}]")
                    print(f"🔍 Evolved node positions mean: {evolved_positions.mean():.2f}")
                    
                    # Calculate displacement
                    if len(initial_positions) == len(evolved_positions):
                        displacement = np.linalg.norm(evolved_positions - initial_positions, axis=1)
                        print(f"🔍 Node displacement: mean={displacement.mean():.3f}, max={displacement.max():.3f}, min={displacement.min():.3f}")
                        if displacement.max() > 0.1:  # Threshold for "significant" movement
                            print(f"🔍 ✅ Graph {i} evolved significantly!")
                        else:
                            print(f"🔍 ⚠️ Graph {i} barely moved")
                    else:
                        print(f"🔍 ⚠️ Node count mismatch: initial={len(initial_positions)}, evolved={len(evolved_positions)}")
            else:
                print(f"🔍 ❌ No evolved graph available for snake {i}")

            # FIXED: Store this snake with CORRECT sample ID
            print(f"🔍 Storing snake {i} in self.snakes")
            self.snakes.append(s)
            print(f"🔍 self.snakes length after append: {len(self.snakes)}")
            
            # Get sample ID from the graph and store with correct mapping
            sample_id = getattr(l, 'sample_id', str(i))
            self.snake_sample_ids.append(sample_id)
            print(f"🔍 Stored sample ID {sample_id} for snake {i}")
            print(f"🔍 self.snake_sample_ids length after append: {len(self.snake_sample_ids)}")

            dmap = s.renderDistanceMap(g.shape[1:],self.cropsz,self.dmax,
                                     self.maxedgelen)
            snake_dmap.append(dmap)

        print(f"🔍 === AFTER SNAKE PROCESSING LOOP ===")
        print(f"🔍 Final self.snakes length: {len(self.snakes)}")
        print(f"🔍 Final self.snake_sample_ids length: {len(self.snake_sample_ids)}")
        print(f"🔍 Final snake_sample_ids: {self.snake_sample_ids}")
        print(f"🔍 Has 'snakes' attribute: {hasattr(self, 'snakes')}")
        print(f"🔍 Has 'snake_sample_ids' attribute: {hasattr(self, 'snake_sample_ids')}")

        snake_dm = torch.stack(snake_dmap,0).unsqueeze(1)
        # Ensure snake_dm is on the same device as pred_dmap
        if snake_dm.device != pred_dmap.device:
            snake_dm = snake_dm.to(pred_dmap.device)
        
        loss = torch.pow(pred_dmap-snake_dm,2).mean()
        
        # FIXED: Keep backward compatibility by setting self.snake to last snake
        if self.snakes:
            self.snake = self.snakes[-1]  # Last snake for backward compatibility
            print(f"🔍 Set self.snake to last snake from self.snakes")
        else:
            self.snake = None
            print(f"🔍 No snakes available, set self.snake to None")
                  
        self.gimg = gimg
        
        print(f"🔍 === SNAKE FAST LOSS FORWARD END ===")
        print(f"🔍 Final loss value: {loss.item()}")
        print(f"🔍 Final self.snakes length: {len(self.snakes)}")
        print(f"🔍 Final self.snake_sample_ids length: {len(self.snake_sample_ids)}")
        print(f"🔍 Final hasattr(self, 'snakes'): {hasattr(self, 'snakes')}")
        print(f"🔍 Final hasattr(self, 'snake_sample_ids'): {hasattr(self, 'snakes')}")
        
        return loss
    
class SnakeSimpleLoss(nn.Module):
    def __init__(self, stepsz,alpha,beta,fltrstdev,ndims,nsteps,
                       cropsz,dmax,maxedgelen,extgradfac):
        super(SnakeSimpleLoss,self).__init__()
        self.stepsz=stepsz
        self.alpha=alpha
        self.beta=beta
        self.fltrstdev=fltrstdev
        self.ndims=ndims
        self.cropsz=cropsz
        self.dmax=dmax
        self.maxedgelen=maxedgelen
        self.extgradfac=extgradfac
        self.nsteps=nsteps

        self.fltr =gradImSnake.makeGaussEdgeFltr(self.fltrstdev,self.ndims)
        self.fltrt=torch.from_numpy(self.fltr).type(torch.float32)

        self.iscuda=False
        
        # Tracking variables for debugging
        self.current_epoch = None
        self.batch_counter = 0
        self.total_forward_calls = 0
        
        # Create output directory for evolved distance maps
        self.output_dir = "evolved_distance_maps_snake_simple"
        os.makedirs(self.output_dir, exist_ok=True)

    def cuda(self):
        super(SnakeSimpleLoss,self).cuda()
        self.fltrt=self.fltrt.cuda()
        self.iscuda=True
        return self
    
    def set_epoch(self, epoch):
        """Set current epoch for tracking and saving"""
        self.current_epoch = epoch
        self.batch_counter = 0  # Reset batch counter for new epoch

    def forward(self,pred_dmap,lbl_graphs,crops=None):
        # Increment counters
        self.total_forward_calls += 1
        self.batch_counter += 1
        
        epoch_str = f"epoch_{self.current_epoch}" if self.current_epoch is not None else "epoch_unknown"
        batch_str = f"batch_{self.batch_counter}"
        
        # Ensure filter is on the same device as input
        if pred_dmap.device != self.fltrt.device:
            self.fltrt = self.fltrt.to(pred_dmap.device)
            self.iscuda = pred_dmap.device.type == 'cuda'
    
        pred_=pred_dmap.detach()
        
        # Compute gradient image
        gimg=gradImSnake.cmptGradIm(pred_,self.fltrt)
        gimg*=self.extgradfac
        
        snake_dmap=[]

        # Reset snakes list every forward pass
        self.snakes = []
        self.snake_sample_ids = []

        for i,lg in enumerate(zip(lbl_graphs,gimg)):
            l = lg[0]
            g = lg[1]
            if crops:
                crop = crops[i]
            else:
                crop=[slice(0,s) for s in g.shape[1:]]
            
            # Create and evolve snake
            s=gradImSnake.GradImSnake(l,crop,self.stepsz,self.alpha,
                                      self.beta,self.ndims,g)
            if self.iscuda: s.cuda()
            
            # Store sample ID
            sample_id = getattr(l, 'sample_id', str(i))
            
            # === RENDER INITIAL STATE (BEFORE EVOLUTION) ===
            lbl_initial = np.zeros(g.shape[1:])
            lbl_initial = s.renderSnakeWithLines(lbl_initial)
            
            if np.sum(lbl_initial) == 0:
                dmap_initial = self.dmax * np.ones(lbl_initial.shape)
            else:
                dmap_initial = dist(1-lbl_initial)
                dmap_initial[dmap_initial > self.dmax] = self.dmax

            # === EVOLVE SNAKE ===
            s.optim(self.nsteps)

            # Store snake
            self.snakes.append(s)
            self.snake_sample_ids.append(sample_id)

            # === CREATE DISTANCE MAP (AFTER EVOLUTION) ===
            lbl = np.zeros(g.shape[1:])
            lbl = s.renderSnakeWithLines(lbl)
            
            if np.sum(lbl) == 0:
                dmap = self.dmax * np.ones(lbl.shape)
            else:
                dmap = dist(1-lbl)
                dmap[dmap > self.dmax] = self.dmax
            
            # === SAVE DISTANCE MAP TO DISK ===
            try:
                # Create epoch subdirectory
                epoch_dir = os.path.join(self.output_dir, epoch_str)
                os.makedirs(epoch_dir, exist_ok=True)
                
                # Save evolved distance map as numpy array
                evolved_npy_filename = f"{batch_str}_sample_{sample_id}_evolved_dmap.npy"
                evolved_npy_path = os.path.join(epoch_dir, evolved_npy_filename)
                np.save(evolved_npy_path, dmap)
                
                # Get ground truth distance map from stored targets
                if hasattr(self, 'current_targets') and self.current_targets is not None:
                    if self.current_targets.dim() == 4:  # (B, C, H, W)
                        gt_dmap = self.current_targets[i, 0].detach().cpu().numpy()
                    else:  # (B, H, W)
                        gt_dmap = self.current_targets[i].detach().cpu().numpy()
                else:
                    gt_dmap = pred_dmap[i, 0].detach().cpu().numpy()
                
                # Get prediction distance map for visualization
                pred_dmap_viz = pred_dmap[i, 0].detach().cpu().numpy()
                
                # Save ground truth distance map as numpy array
                gt_npy_filename = f"{batch_str}_sample_{sample_id}_true_dmap.npy"
                gt_npy_path = os.path.join(epoch_dir, gt_npy_filename)
                np.save(gt_npy_path, gt_dmap)
                
                # Create 4-panel comparison: GT | Prediction | Initial | Evolved
                comparison_filename = f"{batch_str}_sample_{sample_id}_comparison.png"
                comparison_path = os.path.join(epoch_dir, comparison_filename)
                
                fig, axes = plt.subplots(1, 4, figsize=(24, 6))
                
                # Ground truth
                im0 = axes[0].imshow(gt_dmap, cmap='viridis', interpolation='nearest', vmin=0, vmax=15)
                axes[0].set_title(f'Ground Truth\nShape: {gt_dmap.shape}', fontsize=10, fontweight='bold')
                axes[0].axis('off')
                plt.colorbar(im0, ax=axes[0], label='Distance', fraction=0.046, pad=0.04)
                
                # Model prediction
                im1 = axes[1].imshow(pred_dmap_viz, cmap='viridis', interpolation='nearest', vmin=0, vmax=15)
                axes[1].set_title(f'Model Prediction\nShape: {pred_dmap_viz.shape}', fontsize=10, fontweight='bold')
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], label='Distance', fraction=0.046, pad=0.04)
                
                # Initial snake (before evolution)
                im2 = axes[2].imshow(dmap_initial, cmap='plasma', interpolation='nearest', vmin=0, vmax=15)
                axes[2].set_title(f'Initial Snake (Before Evolution)\nShape: {dmap_initial.shape}', fontsize=11, fontweight='bold')
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2], label='Distance', fraction=0.046, pad=0.04)
                
                # Evolved snake (after evolution)
                im3 = axes[3].imshow(dmap, cmap='viridis', interpolation='nearest', vmin=0, vmax=15)
                axes[3].set_title(f'Evolved Snake (After Evolution)\nShape: {dmap.shape}', fontsize=11, fontweight='bold')
                axes[3].axis('off')
                plt.colorbar(im3, ax=axes[3], label='Distance', fraction=0.046, pad=0.04)
                
                fig.suptitle(f'{epoch_str} | {batch_str} | Sample {sample_id}', fontsize=16, y=0.98, fontweight='bold')
                plt.tight_layout()
                plt.savefig(comparison_path, dpi=100, bbox_inches='tight')
                plt.close()
            except Exception as e:
                pass
                
            # Create tensor and move to device
            dmap_tensor = torch.Tensor(dmap).type(torch.float32)
            if pred_dmap.device.type == 'cuda':
                dmap_tensor = dmap_tensor.cuda()
            else:
                dmap_tensor = dmap_tensor.to(pred_dmap.device)
            
            # Store distance map
            snake_dmap.append(dmap_tensor)

        # Stack all distance maps
        snake_dm=torch.stack(snake_dmap,0).unsqueeze(1)
        if snake_dm.device != pred_dmap.device:
            snake_dm = snake_dm.to(pred_dmap.device)
        
        # Compute loss - shapes should match now!
        loss=torch.pow(pred_dmap-snake_dm,2).mean()
        
        # Keep backward compatibility
        if self.snakes:
            self.snake = self.snakes[-1]
        else:
            self.snake = None
            
        self.gimg=gimg
        
        return loss
    