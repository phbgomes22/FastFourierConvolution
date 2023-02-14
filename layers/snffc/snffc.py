'''
Authors: Chi, Lu and Jiang, Borui and Mu, Yadong
Adaptations: Pedro Gomes 
'''

import torch.nn as nn
from util import *
from torch.nn.utils import spectral_norm
from ..ffc import *

## Not Used
class SN_FFC(nn.Module):
    '''
    The SNFFC Layer

    It represents the module that receives the total signal, splits into local and global signals and returns the complete signal in the end.
    This represents the layer of the Fast Fourier Convolution that comes in place of a vanilla convolution layer.

    It contains:
        Conv2ds with a kernel_size received as a parameter from the __init__ in `kernel_size`.
        The Spectral Transform module for the processing of the global signal. 
    
    It also contains Spectral Normalization
    '''
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 ratio_gin: float, ratio_gout: float, stride: int = 1, padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = False, enable_lfu: bool = True,
                 num_classes: int = 1):
        super(SNFFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        # calculate the number of input and output channels based on the ratio (alpha) 
        # of the local and global signals 
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        # defines the module as a Conv2d unless the channels input or output are zero
        condition = not (in_cl == 0 or out_cl == 0)
        # this is the convolution that processes the local signal and contributes 
        # for the formation of the outputted local signal
        self.convl2l = self.snconv2d(condition, 
                                    in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias)

        condition = not (in_cl == 0 or out_cg == 0)
        # this is the convolution that processes the local signal and contributes 
        # for the formation of the outputted global signal
        self.convl2g = self.snconv2d(condition, 
                                    in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias)

        condition = not (in_cg == 0 or out_cl)
        # this is the convolution that processes the global signal and contributes 
        # for the formation of the outputted local signal
        self.convg2l = self.snconv2d(condition, 
                                    in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias)
        
        # defines the module as the Spectral Transform unless the channels output are zero
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SNSpectralTransform

        # (Fourier)
        # this is the convolution that processes the global signal and contributes (in the spectral domain)
        # for the formation of the outputted global signal 
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)


    def snconv2d(self, condition:bool, in_cg: int, out_cl:int, kernel_size:int,
                 stride: int, padding: int, dilation: int, groups: int, bias: int):
        if condition:
            return spectral_norm(nn.Conv2d(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias))
        return nn.Identity(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias)

    # receives the signal as a tuple containing the local signal in the first position
    # and the global signal in the second position
    def forward(self, x):
        # splits the received signal into the local and global signals
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            # creates the output local signal passing the right signals to the right convolutions
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            # creates the output global signal passing the right signals to the right convolutions
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        # returns both signals as a tuple
        return out_xl, out_xg



## Creating through inherintance
class SNFFC(FFC):
    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            # creates the output local signal passing the right signals to the right convolutions
            out_xl = spectral_norm(self.convl2l(x_l)) + spectral_norm(self.convg2l(x_g))
        if self.ratio_gout != 0:
            # creates the output global signal passing the right signals to the right convolutions
            out_xg = spectral_norm(self.convl2g(x_l)) + self.convg2g(x_g)

        # returns both signals as a tuple
        return out_xl, out_xg
