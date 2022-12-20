'''
Authors: Chi, Lu and Jiang, Borui and Mu, Yadong
Adaptations: Pedro Gomes 
'''

import torch.nn as nn
from util import *
from config import Config
from .snffc import *
from .snffc_transpose import *



class SNFFC_ACT(nn.Module):
    '''

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
                 activation_layer=nn.Identity, enable_lfu=True, upsampling=False, out_padding=0):
        '''
        The parameter `upsampling` controls whether the FFC module or the FFCTransposed module will be used. 
        The FFC works for downsampling, while FFCTransposed, for upsampling.
        '''
        super(SNFFC_ACT, self).__init__()

        # Creates the FFC layer, that will process the signal 
        # (divided into local and global and apply the convolutions and Fast Fourier)
        if upsampling:
            self.ffc = SNFFCTranspose(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, out_padding)
        else:
            self.ffc = SNFFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)


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

    def forward(self, x):
        debug_print(" -- FFC_BN_ACT")
        x_l, x_g = self.ffc(x)
        self.print_size(x_l)
        
        x_l = self.act_l(x_l)
        self.print_size(x_l)

        x_g = self.act_g(x_g)
        debug_print(" -- Fim FFC_BN_ACT")
        return x_l, x_g
