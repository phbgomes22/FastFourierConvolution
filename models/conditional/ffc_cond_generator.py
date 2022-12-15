'''
Author: Pedro Gomes
'''

import torch.nn as nn
from util import *
from ffc import *
from ..ffcmodel import FFCModel
from .cond_bn import ConditionalBatchNorm2d


# Generator Code
class FFCCondGenerator(FFCModel):
    def __init__(self, nz: int, nc: int, ngf: int, num_classes: int, image_size: int, embed_size: int, debug=False):
        super(FFCCondGenerator, self).__init__(inplanes=ngf * 8, debug=debug)
        self.image_size = image_size
        self.embed_size = embed_size
        self.nz = nz
        self.main = FFC_BN_ACT(nz, ngf*16, 4, 0, 0.5, 1, 0, 
                              activation_layer=nn.ReLU, 
                           #   norm_layer=ConditionalBatchNorm2d, 
                              upsampling=True )
                         #     num_classes=num_classes)
        
        self.ffc1 = FFC_BN_ACT(ngf*8, ngf*4, 4, 0, 0.5, 2, 1, 
                               activation_layer=nn.ReLU, 
                           #    norm_layer=ConditionalBatchNorm2d, 
                               upsampling=True )
                            #   num_classes=num_classes)

        self.ffc2 = FFC_BN_ACT(ngf*4, ngf*2, 4, 0.5, 0.5, 2, 1, 
                               activation_layer=nn.ReLU, 
                            #   norm_layer=ConditionalBatchNorm2d,  
                               upsampling=True )
                            #   num_classes=num_classes)

        self.ffc3 = FFC_BN_ACT(ngf*2, ngf, 4, 0.5, 0.5, 2, 1, 
                               activation_layer=nn.ReLU, 
                           #    norm_layer=ConditionalBatchNorm2d,  
                               upsampling=True )
                             #  num_classes=num_classes)

        self.ffc4 = FFC_BN_ACT(ngf, nc, 4, 0.5, 0, 2, 1, 
                               norm_layer=nn.Identity, 
                               activation_layer=nn.Tanh, upsampling=True)
        
        self.ylabel=nn.Sequential(
            nn.Linear(num_classes,embed_size),
            nn.ReLU(True)
        )

        self.yz=nn.Sequential(
            nn.Linear(nz, nz + embed_size),
            nn.ReLU(True)
        )

    def forward(self, input, labels):
        # latent vector z: N x noise_dim x 1 x 1 
        embedding = self.ylabel(labels).unsqueeze(2).unsqueeze(3)
 
        z = input #self.yz(input)
        x = torch.cat([z, embedding], dim=1)
        x = x.view(input.shape[0], self.nz + self.embed_size, 1, 1) # pq nz * 2 ? pq não nz?

        x = self.main(x)
        x = self.ffc1(x)
        x = self.ffc2(x)
        x = self.ffc3(x)
        x = self.ffc4(x)
        x = self.resizer(x)

        return x