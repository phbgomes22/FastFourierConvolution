'''
Authors: Pedro Gomes 
'''

import torch.nn as nn
from util import *
from .spectral_transform import SpectralTransform
from config import Config
from ..print_layer import *
from ..attention_layer import *


class FFCTranspose(nn.Module):
    '''
    The FFC Transposed Layer

    New layer created to make upsampling possible with Fourer Convolutions.
    This represents the layer of the Transposed Fourier Convolution that comes 
    in place of a vanilla transposed convolution.
    '''

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 ratio_gin: float, ratio_gout: float, stride: int = 1, padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False, 
                 enable_lfu: bool = True, out_padding: int = 0):
        '''
        in_channels: number of channels that the FFCTranspose receives,
        out_channels: number of channes that the FFCTranspose returns in the output tensor,
        ratio_gin: the split between the local and global signals in the input (0, 1)
        ratio_gout: the split between the local and global signals in the output (0, 1)
        enable_lfu: if the local fourier unit is active or not 
        '''
        super(FFCTranspose, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        # calculate the number of input and output channels based on the ratio (alpha) 
        # of the local and global signals 
        in_cg = int(in_channels * ratio_gin)
        in_cl = int(in_channels - in_cg)
        out_cg = int(out_channels * ratio_gout)
        out_cl = int(out_channels - out_cg)
        
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        # this is the convolution that processes the local signal and contributes 
        # for the formation of the outputted local signal

        condition = in_cl == 0 or out_cl == 0
        # defines the module as a Conv2d unless the channels input or output are zero
        # (in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t=1, padding: _size_2_t=0, 
        # output_padding: _size_2_t=0, groups: int=1, bias: bool=True, dilation: int=1, padding_mode: str='zeros', device=None, dtype=None)
        self.convl2l = self.convtransp2d(condition, in_cl, out_cl, kernel_size,
                              stride, padding, output_padding=out_padding, groups=groups, bias=bias, dilation=dilation)

        condition = in_cl == 0 or out_cg == 0
        # this is the convolution that processes the local signal and contributes 
        # for the formation of the outputted global signal
        self.convl2g = nn.Sequential( self.convtransp2d(condition, in_cl, out_cg, kernel_size,
                              stride, padding, output_padding=out_padding, groups=groups, bias=bias, dilation=dilation),
                            ## Testing local attention like NL-FFC
                              # SpatialAttn(out_cg, True)
                            ## Testing attention like SAGAN
                              Self_Attn(out_cg, 'relu') )


       
        condition = in_cg == 0 or out_cl == 0
        # this is the convolution that processes the global signal and contributes 
        # for the formation of the outputted local signal
        self.convg2l = self.convtransp2d(condition, in_cg, out_cl, kernel_size,
                              stride, padding, output_padding=out_padding, groups=groups, bias=bias, dilation=dilation)

        # defines the module as the Spectral Transform unless the channels output are zero
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        # (Fourier)
        # this is the convolution that processes the global signal and contributes (in the spectral domain)
        # for the formation of the outputted global signal 
        self.convg2g =  module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu),
            # Upsample with convolution
        self.convg2gup = nn.ConvTranspose2d(out_cg,  out_cg*2, kernel_size,
                              stride, padding, output_padding=out_padding, groups=groups, bias=bias, dilation=dilation)
        

        ## -- debugging
        self.print_size = nn.Sequential(Print(debug=Config.shared().DEBUG))
        
    def convtransp2d(self, condition:bool, in_ch: int, out_ch:int, kernel_size:int,
                 stride: int, padding: int, output_padding: int, groups: int, bias: int, dilation: int):
        if condition:
            return nn.Identity(in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias)

        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size,
                              stride, padding, output_padding=output_padding, 
                              groups=groups, bias=bias, dilation=dilation) 


    # receives the signal as a tuple containing the local signal in the first position
    # and the global signal in the second position
    def forward(self, x):
        # splits the received signal into the local and global signals
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0


        if self.ratio_gout != 1:
            # creates the output local signal passing the right signals to the right convolutions
            debug_print(".  --- FFC Transp")
            self.print_size(x_l)
            out_xl = self.convl2l(x_l) 
            debug_print(".  --- Conv2l2")

            self.print_size(out_xl)
            out_xl = out_xl + self.convg2l(x_g)
            debug_print(".  --- Convgl2")

            self.print_size(out_xl)
            debug_print(".  --- Fim FFC Transp")

        if self.ratio_gout != 0:
            # creates the output global signal passing the right signals to the right convolutions
            out_xg = self.convl2g(x_l)
            if type(x_g) is tuple:
                ## testing upsampling first, then Spectral Transform
                x_g = self.convg2gup(x_g)
                out_xg = out_xg + self.convg2g(x_g)
               
        
        # returns both signals as a tuple
        return out_xl, out_xg

