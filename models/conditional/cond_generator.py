'''
Author: Pedro Gomes
'''

import torch.nn as nn
from util import *
from layers import *



# Generator Code
class CondGenerator(nn.Module):

    def __init__(self, nz: int, nc: int, ngf: int, num_classes: int, image_size: int, embed_size: int):
        super(CondGenerator, self).__init__()
        self.image_size = image_size
        self.embed_size = embed_size
        self.nz = nz
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz + embed_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            # Batch normalization conditioned to class
           # ConditionalBatchNorm2d(ngf * 8, num_classes=num_classes),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
           # ConditionalBatchNorm2d(ngf * 4, num_classes=num_classes),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
           # ConditionalBatchNorm2d(ngf * 2, num_classes=num_classes),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
           # ConditionalBatchNorm2d(ngf, num_classes=num_classes),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        
        self.ylabel=nn.Sequential(
            nn.Linear(num_classes, embed_size),
            nn.ReLU(True)
        )

        self.yz=nn.Sequential(
            nn.Linear(nz, nz*2),
            nn.ReLU(True)
        )

    def forward(self, input, labels):
        # latent vector z: N x noise_dim x 1 x 1 
        embedding = self.ylabel(labels).unsqueeze(2).unsqueeze(3)
        
        z = input #self.yz(input)
        x = torch.cat([z, embedding], dim=1)
        x = x.view(input.shape[0], self.nz + self.embed_size, 1, 1) # input.shape[0]

        return self.main(x)