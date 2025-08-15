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

        for i,lg in enumerate(zip(lbl_graphs,gimg)):
            l = lg[0]
            g = lg[1]

            if crops:
                crop = crops[i]
            else:
                crop=[slice(0,s) for s in g.shape[1:]]
            s = gradImSnake.GradImSnake(l,crop,self.stepsz,self.alpha,
                                      self.beta,self.ndims,g)
            if self.iscuda: s.cuda()

            s.optim(self.nsteps)

            dmap = s.renderDistanceMap(g.shape[1:],self.cropsz,self.dmax,
                                     self.maxedgelen)
            snake_dmap.append(dmap)

        snake_dm = torch.stack(snake_dmap,0).unsqueeze(1)
        # Ensure snake_dm is on the same device as pred_dmap
        if snake_dm.device != pred_dmap.device:
            snake_dm = snake_dm.to(pred_dmap.device)
        
        loss = torch.pow(pred_dmap-snake_dm,2).mean()
                  
        self.snake = s
        self.gimg = gimg
        
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

        for i,lg in enumerate(zip(lbl_graphs,gimg)):
            l = lg[0]
            g = lg[1]
            if crops:
                crop = crops[i]
            else:
                crop=[slice(0,s) for s in g.shape[1:]]
            s=gradImSnake.GradImSnake(l,crop,self.stepsz,self.alpha,
                                      self.beta,self.ndims,g)
            if self.iscuda: s.cuda()

            s.optim(self.nsteps)

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

        snake_dm=torch.stack(snake_dmap,0).unsqueeze(1)
        # Ensure snake_dm is on the same device as pred_dmap
        if snake_dm.device != pred_dmap.device:
            snake_dm = snake_dm.to(pred_dmap.device)
        
        loss=torch.pow(pred_dmap-snake_dm,2).mean()
                  
        self.snake=s
        self.gimg=gimg
        
        return loss
    