'''
Author: Pedro Gomes
'''

import torch.nn as nn
from util import *
from layers import *
from ..ffcmodel import FFCModel



# FFC Generator Code

class SNFFCGenerator(FFCModel):
    '''
    The Generator model using Fast Fourier Convolutions (FFC-Generator). 

    It edits the regular discriminator model to add fourier convolutions to all layers of the network.
    The ratio between the global and local signals in the network is set to 0.5 for the hidden layers.
    '''
    def __init__(self, nz: int, nc: int, ngf: int, g_factor: float = 0.5, debug: bool = False):
        super(SNFFCGenerator, self).__init__(inplanes=ngf * 8, debug=debug)

        self.ffc0 = SNFFC_ACT(nz, ngf*16, 4, 0, g_factor, 1, 0, activation_layer=nn.ReLU, upsampling=True)
        self.ffc1 = SNFFC_ACT(ngf*16, ngf*8, 4, g_factor, g_factor, 2, 1, activation_layer=nn.ReLU, upsampling=True)
        self.ffc2 = SNFFC_ACT(ngf*8, ngf*4, 4, g_factor, g_factor, 2, 1, activation_layer=nn.ReLU,  upsampling=True)
        self.ffc3 = SNFFC_ACT(ngf*4, ngf*2, 4, g_factor, g_factor, 2, 1, activation_layer=nn.ReLU,  upsampling=True)
        self.ffc4 = SNFFC_ACT(ngf*2, nc, 4, g_factor, 0, 2, 1, norm_layer=nn.Identity, activation_layer=nn.Tanh, upsampling=True)

    def forward(self, x):
        debug_print('G --')
        x = self.ffc0(x)
        x = self.print_size(x)
        x = self.ffc1(x)
        x = self.print_size(x)
        x = self.ffc2(x)
        x = self.print_size(x)
        x = self.ffc3(x)
        x = self.print_size(x)
        x = self.ffc4(x)
        x = self.resizer(x)
        debug_print("End G --")
        
        return x
