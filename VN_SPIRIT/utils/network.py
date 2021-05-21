import math
from typing import List, Tuple
import numpy as np

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms

from fastmri.models.unet import Unet
from fastmri.models.varnet import *

import sigpy as sp
from sigpy import from_pytorch, to_pytorch_function
from VN_SPIRIT.utils.complexLayers import ConvBlock, ComplexConv2d


class VarNet2(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
    ):
        super().__init__()

        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
        
        im_coil = fastmri.ifft2c(kspace_pred)
        im_comb = fastmri.complex_mul(im_coil, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )
        
        return im_comb, kspace_pred


class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """
        super().__init__()

        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        im_coil = fastmri.ifft2c(kspace_pred)
        
        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1), im_coil, kspace_pred


    
    
    
class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
        )

        return current_kspace - soft_dc - model_term

    
class rakiSpirit(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        
        self.layer1 = ConvBlock(in_chans, in_chans*2, kernel_size=5, act_type='relu')
        self.layer2 = ConvBlock(in_chans*2, in_chans*4, kernel_size=3, act_type='relu')
        self.layer3 = ConvBlock(in_chans*4, in_chans*2, kernel_size=3, act_type='relu')
        self.layer4 = ConvBlock(in_chans*2, in_chans, kernel_size=5, act_type='none')
        self.layers = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)

    def forward(self, input):

        out = self.layers(input)
        return out
    