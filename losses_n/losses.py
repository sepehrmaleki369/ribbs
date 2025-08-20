import torch
from torch import nn
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as dist
from . import gradImSnake

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

    def cuda(self):
        super(SnakeSimpleLoss,self).cuda()
        self.fltrt=self.fltrt.cuda()
        self.iscuda=True
        return self

    def forward(self,pred_dmap,lbl_graphs,crops=None):
        print(f"🔍 === SNAKE SIMPLE LOSS FORWARD START ===")
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
    
        pred_=pred_dmap.detach()
        
        # Final safety check before cmptGradIm
        if pred_.device != self.fltrt.device:
            # Force move one more time
            self.fltrt = self.fltrt.to(pred_.device)
        
        gimg=gradImSnake.cmptGradIm(pred_,self.fltrt)
        gimg*=self.extgradfac
        snake_dmap=[]

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
            s=gradImSnake.GradImSnake(l,crop,self.stepsz,self.alpha,
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
                    print(f"🔍 Evolved node positions range: [{evolved_positions.min():.2f}, {initial_positions.max():.2f}]")
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

            lbl = np.zeros(g.shape[1:])
            lbl = s.renderSnakeWithLines(lbl)
            if np.sum(lbl) == 0:
                dmap = self.dmax * np.ones(lbl.shape)
            else:
                dmap = dist(1-lbl)
                dmap[dmap > self.dmax] = self.dmax
                
            # Create tensor and move to the same device as pred_dmap
            dmap_tensor = torch.Tensor(dmap).type(torch.float32)
            if pred_dmap.device.type == 'cuda':
                dmap_tensor = dmap_tensor.cuda()
            else:
                dmap_tensor = dmap_tensor.to(pred_dmap.device)
            
            snake_dmap.append(dmap_tensor)

        print(f"🔍 === AFTER SNAKE PROCESSING LOOP ===")
        print(f"🔍 Final self.snakes length: {len(self.snakes)}")
        print(f"🔍 Final self.snake_sample_ids length: {len(self.snake_sample_ids)}")
        print(f"🔍 Final snake_sample_ids: {self.snake_sample_ids}")
        print(f"🔍 Has 'snakes' attribute: {hasattr(self, 'snakes')}")
        print(f"🔍 Has 'snake_sample_ids' attribute: {hasattr(self, 'snake_sample_ids')}")

        snake_dm=torch.stack(snake_dmap,0).unsqueeze(1)
        # Ensure snake_dm is on the same device as pred_dmap
        if snake_dm.device != pred_dmap.device:
            snake_dm = snake_dm.to(pred_dmap.device)
        
        loss=torch.pow(pred_dmap-snake_dm,2).mean()
        
        # FIXED: Keep backward compatibility by setting self.snake to last snake
        if self.snakes:
            self.snake = self.snakes[-1]  # Last snake for backward compatibility
            print(f"🔍 Set self.snake to last snake from self.snakes")
        else:
            self.snake = None
            print(f"🔍 No snakes available, set self.snake to None")
            
        self.gimg=gimg
        
        print(f"🔍 === SNAKE SIMPLE LOSS FORWARD END ===")
        print(f"🔍 Final loss value: {loss.item()}")
        print(f"🔍 Final self.snakes length: {len(self.snakes)}")
        print(f"🔍 Final self.snake_sample_ids length: {len(self.snake_sample_ids)}")
        print(f"🔍 Final hasattr(self, 'snakes'): {hasattr(self, 'snakes')}")
        print(f"🔍 Final hasattr(self, 'snake_sample_ids'): {hasattr(self, 'snake_sample_ids')}")
        
        return loss
    