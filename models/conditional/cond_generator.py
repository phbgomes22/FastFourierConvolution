'''
Author: Pedro Gomes
'''

import torch.nn as nn
from util import *
from layers import *
import math


# https://github.com/znxlwm/pytorch-MNIST-CelebA-cGAN-cDCGAN/blob/master/pytorch_MNIST_cDCGAN.py

class CondCvGenerator(nn.Module):

    def __init__(self, nz: int, nc: int, ngf: int, num_classes: int, image_size: int, embed_size: int):
        super(CondCvGenerator, self).__init__()
        self.image_size = image_size
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.nz = nz

        self.label_embed = nn.Embedding(num_classes, num_classes)

        # why - 3? 
        # the first convolution has no padding and stride 1 
        # -ie: it moves from a 1x1 dim to a 4x4 dim
        # so we would subtract -2, the extra -1 is for the last layer.
        self.number_convs = int(math.log2(ngf)) - 3

        mult = int(math.pow(2, self.number_convs - 1))

        self.label_conv = nn.Sequential(
            nn.ConvTranspose2d(num_classes, ngf*mult, 4, 1, 0),
            nn.BatchNorm2d(ngf*mult),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.input_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf*mult, 4, 1, 0),
            nn.BatchNorm2d(ngf*mult),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main = self.create_layers(nc=nc, ngf=ngf) 

    def create_layers(self, nc: int, ngf: int):
        layers = []

        # adds the hidden layers
        for itr in range(self.number_convs, 0, -1):
            mult = int(math.pow(2, itr)) # 2^iter
            
            layers.append( nn.ConvTranspose2d(ngf*mult, ngf*(mult//2), 4, 2, 1, bias=False) )
            layers.append( nn.BatchNorm2d(ngf*(mult//2)) )
            layers.append( nn.ReLU(True) )

        # - testing last ffc convolution with full image size 
        layers.append( nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False) ) 
        # adds the last layer
        layers.append( nn.Tanh() )

        return nn.Sequential(*layers)

    def forward(self, input, labels):
        ## conv for the embedding
        # latent vector z: N x noise_dim x 1 x 1 
        embedding = self.label_embed(labels).unsqueeze(2).unsqueeze(3)
        embedding = embedding.view(labels.shape[0], -1, 1, 1) # labels.shape[0]
        embedding = self.label_conv(embedding)

        ## convolution of the noise entry
        input = self.input_conv(input)

        x = torch.cat([input, embedding], dim=1)
       # x = x.view(input.shape[0], self.nz + self.num_classes, 1, 1) # pq nz * 2 ? pq não nz?

        return self.main(x)
