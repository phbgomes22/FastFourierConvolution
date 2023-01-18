'''
Author: Pedro Gomes
'''

import torch.nn as nn
from util import *
from layers import *
from ..ffcmodel import FFCModel


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



# - This is the one bringing good results!
class FFCCondGenerator(FFCModel):

    def __init__(self, nz: int, nc: int, ngf: int, num_classes: int, image_size: int, embed_size: int):
        super(FFCCondGenerator, self).__init__(inplanes=ngf * 8, debug=False)
        self.image_size = image_size
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.nz = nz

        self.label_embed = nn.Embedding(num_classes, num_classes)

        self.label_conv = nn.Sequential(
            nn.ConvTranspose2d(num_classes, ngf*4, 4, 1, 0),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.input_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            FFC_BN_ACT(ngf*8, ngf*4, 4, 0, 0.5, 2, 1, 
                               activation_layer=nn.LeakyReLU, 
                               upsampling=True),
            # state size. (ngf*4) x 8 x 8
            FFC_BN_ACT(ngf*4, ngf*2, 4, 0.5, 0.5, 2, 1, 
                               activation_layer=nn.LeakyReLU, 
                               upsampling=True),
            # state size. (ngf*2) x 16 x 16
            FFC_BN_ACT(ngf*2, ngf*1, 4, 0.5, 0.5, 2, 1, 
                               activation_layer=nn.LeakyReLU, 
                               upsampling=True),
            # state size. (ngf) x 32 x 32
            FFC_BN_ACT(ngf*1, nc, 4, 0.5, 0, 2, 1, 
                               norm_layer=nn.Identity, 
                               activation_layer=nn.Tanh, upsampling=True)
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, labels):
        ## conv for the embedding
        # latent vector z: N x noise_dim x 1 x 1 
        embedding = self.label_embed(labels).unsqueeze(2).unsqueeze(3)
        embedding = embedding.view(labels.shape[0], -1, 1, 1) # labels.shape[0]
        embedding = self.label_conv(embedding)

        ## convolution of the noise entry
        input = self.input_conv(input)

        ## joins input noise with embedding labels
        x = torch.cat([input, embedding], dim=1)

        ## main convolutions with ffc layer
        x = self.main(x)
        x = self.resizer(x)

        return x

# Generator Code
# class FFCCondGenerator(FFCModel):
#     def __init__(self, nz: int, nc: int, ngf: int, num_classes: int, image_size: int, embed_size: int, debug=False):
#         super(FFCCondGenerator, self).__init__(inplanes=ngf * 16, debug=debug)
#         self.image_size = image_size
#         self.embed_size = embed_size
#         self.nz = nz
#         self.ffc0 = FFC_BN_ACT(nz + embed_size, ngf*8, 4, 0, 0.5, 1, 0, 
#                               activation_layer=nn.LeakyReLU, 
#                               upsampling=True)
        
#         self.ffc1 = FFC_BN_ACT(ngf*8, ngf*4, 4, 0.5, 0.5, 2, 1, 
#                                activation_layer=nn.LeakyReLU, 
#                                upsampling=True)

#         self.ffc2 = FFC_BN_ACT(ngf*4, ngf*2, 4, 0.5, 0.5, 2, 1, 
#                                activation_layer=nn.LeakyReLU, 
#                                upsampling=True)

#         self.ffc3 = FFC_BN_ACT(ngf*2, ngf*1, 4, 0.5, 0.5, 2, 1, 
#                                activation_layer=nn.LeakyReLU,
#                                upsampling=True)

#         self.ffc4 = FFC_BN_ACT(ngf*1, nc, 4, 0.5, 0, 2, 1, 
#                                norm_layer=nn.Identity, 
#                                activation_layer=nn.Tanh, upsampling=True)
        

#         self.ylabel = nn.Sequential(
#             nn.Linear(num_classes, embed_size),
#             nn.ReLU(True)
#         )


#     def forward(self, input, labels):
#         # latent vector z: N x noise_dim x 1 x 1 
#         embedding = self.ylabel(labels).unsqueeze(2).unsqueeze(3)
 
#         z = input #self.yz(input)
#         x = torch.cat([z, embedding], dim=1)
#         x = x.view(input.shape[0], self.nz + self.embed_size, 1, 1) # pq nz * 2 ? pq não nz?

#         x = self.ffc0(x, labels)
#         x = self.ffc1(x, labels)
#         x = self.ffc2(x, labels)
#         x = self.ffc3(x, labels)
#         x = self.ffc4(x)
#         x = self.resizer(x)

#         return x