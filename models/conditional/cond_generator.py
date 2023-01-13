'''
Author: Pedro Gomes
'''

# https://towardsdatascience.com/using-conditional-deep-convolutional-gans-to-generate-custom-faces-from-text-descriptions-e18cc7b8821

import torch.nn as nn
from util import *
from layers import *

class CondGenerator(nn.Module):
    def __init__(self, nz: int, nc: int, ngf: int, num_classes: int, image_size: int, embed_size: int):
        super(CondGenerator, self).__init__()
        self.image_size = image_size
        self.embed_size = embed_size
        self.deconv1_1 = nn.ConvTranspose2d(100, ngf*8, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(ngf*8)
        self.deconv1_2 = nn.ConvTranspose2d(num_classes, ngf*8, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(ngf*8)
        self.deconv2 = nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(ngf*8)
        self.deconv3 = nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(ngf*4)
        self.deconv4 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(ngf*2)
        self.deconv5 = nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1)


        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, input, label):
        x = self.act(self.deconv1_1_bn(self.deconv1_1(input)))
        y = self.act(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = self.act(self.deconv2_bn(self.deconv2(x)))
        x = self.act(self.deconv3_bn(self.deconv3(x)))
        x = self.act(self.deconv4_bn(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))
        return x