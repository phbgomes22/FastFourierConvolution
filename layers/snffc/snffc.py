'''
Authors: Chi, Lu and Jiang, Borui and Mu, Yadong
Adaptations: Pedro Gomes 
'''

import torch.nn as nn
from util import *
from torch.nn.utils import spectral_norm
from ..ffc.ffc import *

## Creating through inherintance
class SNFFC(FFC):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 ratio_gin: float, ratio_gout: float, stride: int = 1, padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = False, enable_lfu: bool = True,
                 attention: bool = False):

        FFC.__init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride,
                    padding, dilation, groups, bias, enable_lfu, attention)

        
        self.convl2l = spectral_norm(self.convl2l)
        self.convg2l = spectral_norm(self.convg2l)
        

        self.convl2g = spectral_norm(self.convl2g) if isinstance(self.convl2g, nn.Conv2d) else self.convl2g

        # -- changing convg2g
        if not isinstance(self.convg2g, nn.Identity):
            # Create a new convg2g layer that is a copy of the old one
           # new_convg2g = type(self.convg2g)(self.in_cg, self.out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

            # Replace the BatchNorm2d layer with Identity
            for name, module in self.convg2g.named_children():
                if isinstance(module, nn.Conv2d):
                    print("adicionando spectral norm em ", name)
                    print(module)
                    print(type(module))
                    new_module = spectral_norm(module)
                    self.convg2g._modules[name] = module
                #     new_convg2g.add_module(name, spectral_norm(module))
                # else:
                #     new_convg2g.add_module(name, module)

            # Set the new convg2g layer
            # self.convg2g = new_convg2g


    def forward(self, x):
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
