import copy
from typing import List


import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math


class _BaseMerger:
    def __init__(self):
        """Merges feature embedding by name."""

    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxC
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(
            axis=-1
        )


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxCWH
        return features.reshape(len(features), -1)
#----------------------------------------------

class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        #self.device=("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_dims = input_dims
        self.output_dim = output_dim
        #self.preprocessing_modules = torch.nn.ModuleList()
        #self.convmod=Block(sum(self.input_dims),mlp_ratio=4)
        #self.preprocessing_modules.append(convmod)
        #self.module =MeanMapper(self.output_dim)
        #self.preprocessing_modules.append(module)



    def forward(self, features):
        _features = []
        for i in range(0,len(features),2):
            #print("bcat",len(features),features[i].shape,features[i+1].shape)
            x=torch.cat((features[i],features[i+1]), dim=2)
            #print("cat",x.shape)
        #x=torch.unsqueeze(x,0)
        #print("cat",x.shape,int(math.sqrt(x.shape[1])))
        #hp=int(math.sqrt(x.shape[1]))
        #print("out_d",self.output_dim[0],self.output_dim[1])
        #x=patch_to_image(x,(self.output_dim[0],self.output_dim[1])) 
        x= x.permute(0, 2, 3, 4,1) 
        x = x.contiguous().view(x.shape[0], x.shape[1]*3*3, -1)
        x=F.fold(x,(self.output_dim[0],self.output_dim[1]),(3,3),1,int((3- 1) / 2)) 
        #print("img",x.shape)
        
        #x=self.module(x)
        
    
        #x=self.convmod(x)
        #x=x.view(*x.shape[:2],hp,3,-1)
        #x=x.view(*x.shape[:4],hp,3)
        #x=x.permute(0,2,4,1,3,5).contiguous()
        #x=torch.squeeze(x)
        #x=x.view(-1, *x.shape[-3:])
        #print(x.shape)
        return x

#---------------------------------------------------------------------------

class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim=preprocessing_dim

    def forward(self, features):
        #features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool2d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()
        
    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        
        return features






#--------------------------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        self.device=("cuda:0" if torch.cuda.is_available() else "cpu")

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")# Number of input channels
        
        self.fc1 = nn.Conv2d(in_channels=dim,out_channels= dim * mlp_ratio, kernel_size=1).to(self.device)
        self.pos = nn.Conv2d(in_channels=dim * mlp_ratio, out_channels=dim * mlp_ratio, kernel_size=3, padding=1, groups=dim * mlp_ratio).to(self.device)
        self.fc2 = nn.Conv2d(in_channels=dim * mlp_ratio, out_channels=dim, kernel_size=1).to(self.device)
        self.act = nn.GELU().to(self.device)

    def forward(self, x):
        B, C, H, W = x.shape

        
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x

class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.device=("cuda:0" if torch.cuda.is_available() else "cpu")

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        ).to(self.device)

        self.v = nn.Conv2d(dim, dim, 1).to(self.device)
        self.proj = nn.Conv2d(dim, dim, 1).to(self.device)

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.norm(x)   
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        return x
    
class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.device=("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weight = nn.Parameter(torch.ones(normalized_shape)).to(self.device)
        self.bias = nn.Parameter(torch.zeros(normalized_shape)).to(self.device)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps,True,self.device)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.device=("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.attn = ConvMod(dim)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6           
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True).to(self.device)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True).to(self.device)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity().to(self.device)

    def forward(self, x):
        x = x + (self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))).to(self.device)
        x = x + (self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))).to(self.device)
        return x
    
# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x
    
def patch_to_image(x, grid_size=[64, 64]):
    # x shape is batch_size x num_patches x c x jigsaw_h x jigsaw_w
    batch_size, num_patches, c, jigsaw_h, jigsaw_w = x.size()
    assert num_patches == grid_size[0] * grid_size[1]
    x_image = x.view(batch_size, grid_size[0], grid_size[1], c, jigsaw_h, jigsaw_w)
    output_h = grid_size[0] * jigsaw_h
    output_w = grid_size[1] * jigsaw_w
    x_image = x_image.permute(0, 3, 1, 4, 2, 5).contiguous()
    x_image = x_image.view(batch_size, c, output_h, output_w)
    return x_image


