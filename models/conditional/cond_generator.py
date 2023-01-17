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
        self.num_classes = num_classes
        self.nz = nz
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz + embed_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        
        self.ylabel=nn.Sequential(
            nn.Linear(num_classes,embed_size),
            nn.ReLU(True)
        )

        self.yz=nn.Sequential(
            nn.Linear(nz, nz + embed_size),
            nn.ReLU(True)
        )

        self.lbl_embed = nn.Embedding(num_classes, embed_size)

    def forward(self, input, labels):
        # latent vector z: N x noise_dim x 1 x 1 
        
        c = self.lbl_embed(labels).unsqueeze(2).unsqueeze(3)

        x = torch.cat([input, c], dim=1)
       # x = x.view(input.shape[0], self.nz + self.num_classes, 1, 1) # pq nz * 2 ? pq não nz?

        return self.main(x)



class CondCvGenerator(nn.Module):

    def __init__(self, nz: int, nc: int, ngf: int, num_classes: int, image_size: int, embed_size: int):
        super(CondCvGenerator, self).__init__()
        self.image_size = image_size
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.nz = nz

        self.label_embed = nn.Embedding(num_classes, self.image_size * self.image_size)

        self.label_conv = nn.Sequential(
            nn.ConvTranspose2d(1, ngf*4, 4, 2, 1),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.input_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf*4, 4, 2, 1),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, labels):
        ## conv for the embedding
        # latent vector z: N x noise_dim x 1 x 1 
        embedding = self.label_embed(labels).unsqueeze(2).unsqueeze(3)
        embedding = embedding.view(labels.shape[0], 1, self.image_size, self.image_size) # labels.shape[0]
        embedding = self.label_conv(embedding)

        ## convolution of the noise entry
        input = self.input_conv(input)
        print(input.shape)
        print(embedding.shape)
        x = torch.cat([input, embedding], dim=1)
       # x = x.view(input.shape[0], self.nz + self.num_classes, 1, 1) # pq nz * 2 ? pq não nz?

        return self.main(x)