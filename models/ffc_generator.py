import torch.nn as nn
from util import Resizer, Print, debug_print
from ffc import FFC_BN_ACT
from config import nz, nc
from ffcmodel import FFCModel



# FFC Generator Code

class FFCGenerator(FFCModel):
    def __init__(self, ngf, debug=False):
        super(FFCGenerator, self).__init__(inplanes=ngf * 8, debug=debug)

        self.ffc0 = FFC_BN_ACT(nz, ngf*16, 4, 0, 0.5, 1, 0, activation_layer=nn.ReLU, upsampling=True)
        self.ffc1 = FFC_BN_ACT(ngf*16, ngf*8, 4, 0.5, 0.5, 2, 1, activation_layer=nn.ReLU, upsampling=True)
        self.ffc2 = FFC_BN_ACT(ngf*8, ngf*4, 4, 0.5, 0.5, 2, 1, activation_layer=nn.ReLU,  upsampling=True)
        self.ffc3 = FFC_BN_ACT(ngf*4, ngf*2, 4, 0.5, 0.5, 2, 1, activation_layer=nn.ReLU,  upsampling=True)
        self.ffc4 = FFC_BN_ACT(ngf*2, nc, 4, 0.5, 0, 2, 1, norm_layer=nn.Identity, activation_layer=nn.Tanh, upsampling=True)

    def forward(self, x):
        x = self.ffc0(x)
        x = self.ffc1(x)
        x = self.ffc2(x)
        x = self.ffc3(x)
        x = self.ffc4(x)
        x = self.resizer(x)
        
        return x



    def forward(self, x):
        debug_print('Começo G --')
        x = self.ffc0(x)
        x = self.print_size(x)
        
        debug_print("FFC 1")
        x = self.ffc1(x)
        debug_print("=")
        x = self.print_size(x)
        debug_print("FFC 2")
        x = self.ffc2(x)
        debug_print("=")
        x = self.print_size(x)
        debug_print("FFC 3")
        x = self.ffc3(x)

        x = self.ffc4(x)
        x = self.resizer(x)
        
        return x
