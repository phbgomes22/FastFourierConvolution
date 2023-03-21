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
        self.convg2l = spectral_norm(self.convg2l) if isinstance(self.convg2l, nn.Conv2d) else self.convg2l
        self.convl2g = spectral_norm(self.convl2g) if isinstance(self.convl2g, nn.Conv2d) else self.convl2g

        # -- changing convg2g   
        if not isinstance(self.convg2g, nn.Identity):
            # Replace the BatchNorm2d layer with Identity
            for name, module in self.convg2g.named_children():
                if isinstance(module, nn.Conv2d):
                    new_module = spectral_norm(module)
                    self.convg2g._modules[name] = new_module
