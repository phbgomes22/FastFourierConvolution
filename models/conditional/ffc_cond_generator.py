'''
Author: Pedro Gomes
'''

import torch.nn as nn
from util import *
from layers import *
from ..ffcmodel import FFCModel
import math



# - This is the one bringing good results!
class FFCCondGenerator(FFCModel):

    def __init__(self, nz: int, nc: int, ngf: int, num_classes: int, embed_size: int):
        super(FFCCondGenerator, self).__init__(inplanes=ngf * 8, debug=False)
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.nz = nz
        self.ngf = ngf

        self.label_embed = nn.Embedding(num_classes, num_classes)

        # why - 3? 
        # the first convolution has no padding and stride 1 
        # -ie: it moves from a 1x1 dim to a 4x4 dim
        # so we would subtract -2, the extra -1 is for the last layer.
        self.number_convs = int(math.log2(ngf)) - 3

        self.label_conv = nn.Sequential(
            nn.ConvTranspose2d(num_classes, ngf*self.number_convs, 4, 1, 0),
            nn.BatchNorm2d(ngf*self.number_convs),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.input_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf*self.number_convs, 4, 1, 0),
            nn.BatchNorm2d(ngf*self.number_convs),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main = self.create_layers(nc=nc, ngf=ngf)

    def create_layers(self, nc: int, ngf: int):
        layers = []

        # adds the hidden layers
        for itr in range(self.number_convs, 0, -1):
            mult = int(math.pow(2, itr)) # 2^iter
            g_in = 0 if itr == self.number_convs else 0.5
            layers.append(
                FFC_BN_ACT(ngf*mult, ngf*(mult//2), 4, g_in, 0.5, 2, 1, 
                               activation_layer=nn.LeakyReLU, 
                               upsampling=True),
            )
        # adds the last layer
        layers.append(
            FFC_BN_ACT(ngf*1, nc, 4, 0.5, 0, 2, 1, 
                               norm_layer=nn.Identity, 
                               activation_layer=nn.Tanh, upsampling=True)
        )

        return nn.Sequential(*layers)


    def forward(self, input, labels):
        ## conv for the embedding
        # latent vector z: N x noise_dim x 1 x 1 
        embedding = self.label_embed(labels).unsqueeze(2).unsqueeze(3)
        embedding = embedding.view(labels.shape[0], -1, 1, 1) # labels.shape[0]
        embedding = self.label_conv(embedding)
        self.print_size(embedding)
        ## convolution of the noise entry
        input = self.input_conv(input)
        self.print_size(embedding)
        ## joins input noise with embedding labels
        x = torch.cat([input, embedding], dim=1)

        ## main convolutions with ffc layer
        x = self.main(x)
        x = self.resizer(x)

        return x





# Generator with Conditional Batch Normalization Code
class FFCCondBNGenerator(FFCModel):
    def __init__(self, nz: int, nc: int, ngf: int, num_classes: int, image_size: int, embed_size: int, debug=False):
        super(FFCCondBNGenerator, self).__init__(inplanes=ngf * 16, debug=debug)
        self.image_size = image_size
        self.embed_size = embed_size
        self.nz = nz
        self.ffc0 = FFC_BN_ACT_COND(nz + embed_size, ngf*8, 4, 0, 0.5, 1, 0, 
                              activation_layer=nn.LeakyReLU, 
                              norm_layer=ConditionalBatchNorm2d, 
                              upsampling=True,
                              num_classes=num_classes)
        
        self.ffc1 = FFC_BN_ACT_COND(ngf*8, ngf*4, 4, 0.5, 0.5, 2, 1, 
                               activation_layer=nn.LeakyReLU, 
                               norm_layer=ConditionalBatchNorm2d, 
                               upsampling=True,
                               num_classes=num_classes)

        self.ffc2 = FFC_BN_ACT_COND(ngf*4, ngf*2, 4, 0.5, 0.5, 2, 1, 
                               activation_layer=nn.LeakyReLU, 
                               norm_layer=ConditionalBatchNorm2d,  
                               upsampling=True,
                               num_classes=num_classes)

        self.ffc3 = FFC_BN_ACT_COND(ngf*2, ngf*1, 4, 0.5, 0.5, 2, 1, 
                               activation_layer=nn.LeakyReLU, 
                               norm_layer=ConditionalBatchNorm2d,  
                               upsampling=True,
                               num_classes=num_classes)

        self.ffc4 = FFC_BN_ACT(ngf*1, nc, 4, 0.5, 0, 2, 1, 
                               norm_layer=nn.Identity, 
                               activation_layer=nn.Tanh, upsampling=True)
        
        self.ylabel=nn.Sequential(
            nn.Linear(num_classes, embed_size),
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

        x = self.ffc0(x, labels)
        x = self.ffc1(x, labels)
        x = self.ffc2(x, labels)
        x = self.ffc3(x, labels)
        x = self.ffc4(x)
        x = self.resizer(x)

        return x