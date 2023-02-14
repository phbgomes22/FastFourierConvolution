'''
Authors: Chi, Lu and Jiang, Borui and Mu, Yadong
Adaptations: Pedro Gomes 
'''

import torch.nn as nn
from util import *
from config import Config
from .ffc import *
from .ffc_transpose import *
from ..noise_injection import *
from ..print_layer import *
from ..snffc.snffc import *



class FFC_BN_ACT(nn.Module):
    '''
    Creates a single FFC -> Batch normalization -> Activation Module flow.

    This is the class that is put in the models as a blackbox. 
    So this is on of the entry point of all code related to the FFC (the other being the FFCSE_block).

    It has:
        The FFC layer module.
        Followed by Bach Normalization components for both the local and global signals.
        Followed by an ActivationLayer 
            -   The default activation layer is nn.Identity, so I think we are supposed to change it.
    '''

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 enable_lfu=True, upsampling=False, out_padding=0,
                 uses_noise: bool = False, uses_sn: bool = False, attention: bool = False):
        '''
        The parameter `upsampling` controls whether the FFC module or the FFCTransposed module will be used. 
        The FFC works for downsampling, while FFCTransposed, for upsampling.
        '''
        super(FFC_BN_ACT, self).__init__()

        # Creates the FFC layer, that will process the signal 
        # (divided into local and global and apply the convolutions and Fast Fourier)
        if upsampling:
            self.ffc = FFCTranspose(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, out_padding, attention=attention)
        else:
            if uses_sn:
                self.ffc = SNFFC(in_channels, out_channels, kernel_size,
                        ratio_gin, ratio_gout, stride, padding, dilation,
                        groups, bias, enable_lfu, attention=attention)
            else:
                self.ffc = FFC(in_channels, out_channels, kernel_size,
                        ratio_gin, ratio_gout, stride, padding, dilation,
                        groups, bias, enable_lfu, attention=attention)

        # create the BatchNormalization layers
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer

        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))

        # create the activation function layers
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer

        if lact is nn.Tanh or lact is nn.Sigmoid:
            self.act_l = lact() # was inplace=True, had to change due to new Tanh function
        else:
            self.act_l = lact(inplace=True)

        if gact is nn.Tanh or gact is nn.Sigmoid:
            self.act_g = gact() # was inplace=True, had to change due to new Tanh function
        else:
            self.act_g = gact(inplace=True)

        self.print_size = Print(debug=Config.shared().DEBUG)

        ## Add Noise - PG
        self.noise = NoiseInjection() if uses_noise else nn.Identity()

    def forward(self, x):
        debug_print(" -- FFC_BN_ACT")
        x_l, x_g = self.ffc(x)
        self.print_size(x_l)
        
        x_l = self.act_l(self.bn_l(x_l))
        self.print_size(x_l)

        x_g = self.act_g(self.bn_g(x_g))
        debug_print(" -- Fim FFC_BN_ACT")

        ## Add Noise - PG
        x_l = self.noise(x_l)
        if type(x_g) != int:
            x_g = self.noise(x_g)

        return x_l, x_g
