'''
Author: Pedro Gomes
'''

import torch.nn as nn
from util import *
from ffc import *
from .ffcmodel import FFCModel


class FFCDiscriminator(FFCModel):
    '''
    The Discriminator model using Fast Fourier Convolutions (FFC-Discriminator). 

    It edits the regular discriminator model to add four fourier convolutions in the hidden layers of the model.
    The ratio between the global and local signals in the network is set to 0.5 for the hidden layers.
    '''
    def __init__(self, nc: int, ndf: int, debug: bool = False):
        '''
        `nc`: number of color channels (1 for grayscale, 3 for colored images),
        `ndf`: size of feature maps in the discriminator - same as the image siz (64),
        `debug`: if running on debug (with debug prints).
        '''
        super(FFCDiscriminator, self).__init__(inplanes=ndf, debug=debug)

        ## 3 x 64 x 64
        self.convolution1 = nn.Sequential(
            # in_ch, out_ch, kernel_size, stride, padding
           nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
           nn.LeakyReLU(0.2, inplace=True)
        )
        # output: BS x 64 x 32 x 32

        self.ffc1 = FFC_BN_ACT(ndf * 1, ndf*2, 4, 0, 0.5, 2, 1, activation_layer=nn.LeakyReLU)
        self.ffc2 = FFC_BN_ACT(ndf * 2, ndf*4, 4, 0.5, 0.5, 2, 1, activation_layer=nn.LeakyReLU)
        self.ffc3 = FFC_BN_ACT(ndf * 4, ndf*8, 4, 0.5, 0.5, 2, 1, activation_layer=nn.LeakyReLU)
        # output: BS x 516 x 4 x 4
        self.ffc4 = FFC_BN_ACT(ndf * 8, 1, 4, 0.5, 0, 1, 0, norm_layer=nn.Identity, activation_layer=nn.Sigmoid )

        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        debug_print('D --')
        x = self.print_size(x)

        x = self.convolution1(x)
        x = self.print_size(x)
        debug_print('FFC 1')
        x = self.ffc1(x)
        
        debug_print('=')
        x = self.print_size(x)

        debug_print('FFC 2')
        x = self.ffc2(x)
        x = self.print_size(x)
        debug_print('=')
        debug_print('FFC 3')
        x = self.ffc3(x)
        # this need to be done because ffc returns a tuple (real and imaginary)
        x = self.ffc4(x)
        x = self.resizer(x)

       # x = self.conv5(x)
        x = self.print_size(x)
        debug_print("End D --")
        return x