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
from ..snffc.snffc_transpose import *

from torch.nn.utils import spectral_norm


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
                 norm_layer:nn.Module=nn.Identity, activation_layer:nn.Module=nn.Identity,
                 enable_lfu=True, upsampling=False, out_padding=0,
                 uses_noise: bool = False, uses_sn: bool = False, num_classes: int = 1):
        '''
        The parameter `upsampling` controls whether the FFC module or the FFCTransposed module will be used. 
        The FFC works for downsampling, while FFCTransposed, for upsampling.
        '''
        super(FFC_BN_ACT, self).__init__()

        # Creates the FFC layer, that will process the signal 
        # (divided into local and global and apply the convolutions and Fast Fourier)
        self.uses_sn = uses_sn
        if upsampling:
            transposed = SNFFCTranspose if uses_sn else FFCTranspose
            print("Upsampling")
            print("Using FFCTranspose with spectral norm by hand!")
            self.ffc = FFCTranspose(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, out_padding=out_padding, num_classes=num_classes)
        else:
            regular = SNFFC if uses_sn else FFC
            print("Using FFC with spectral norm by hand!")
            self.ffc = FFC(in_channels, out_channels, kernel_size,
                        ratio_gin, ratio_gout, stride, padding, dilation,
                        groups, bias, enable_lfu, num_classes=num_classes)
             
        out_ch_l = int(out_channels * (1 - ratio_gout))
        out_ch_g = int(out_channels * ratio_gout)
        # create the BatchNormalization layers
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer

        if num_classes > 1:
            self.bn_l = lnorm(out_ch_l, num_classes)
            self.bn_g = gnorm(out_ch_g, num_classes)
        else:
            self.bn_l = lnorm(out_ch_l)
            self.bn_g = gnorm(out_ch_g)

        # create the activation function layers
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        
        self.act_l = lact(0.1, inplace=True) if isinstance(lact(), nn.LeakyReLU) else lact()
        self.act_g = gact(0.1, inplace=True) if isinstance(gact(), nn.LeakyReLU) else gact()

        self.print_size = Print(debug=Config.shared().DEBUG)

        ## Add Noise - PG
        self.noise_l = NoiseInjection(out_ch_l) if uses_noise else nn.Identity()
        ## Add Noise - PG
        self.noise_g = NoiseInjection(out_ch_g) if uses_noise else nn.Identity()


    def forward(self, x, y=None):
        debug_print(" -- FFC_BN_ACT")
        x_l, x_g = self.ffc(x, y)
        self.print_size(x_l)
        
        if y is not None:
            x_l = self.act_l(self.bn_l(x_l, y))
        else:
            x_l = self.act_l(self.bn_l(x_l))
        self.print_size(x_l)

        if y is not None:
            x_g = self.act_g(self.bn_g(x_g, y))
        else:
            x_g = self.act_g(self.bn_g(x_g))
        debug_print(" -- Fim FFC_BN_ACT")

        # Add Noise - PG
        # x_l = self.noise_l(x_l)
        # if type(x_g) != int:
        #     x_g = self.noise_g(x_g)
        
        return x_l, x_g
