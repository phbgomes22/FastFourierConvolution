'''
Author: Pedro Gomes
'''

import torch.nn as nn
from util import *
from layers import *
import math




## - This is the one bringing good results!
class CondCvDiscriminator(nn.Module):
    def __init__(self, nc: int, ndf: int, num_classes: int, image_size: int):
        super(CondCvDiscriminator, self).__init__()
        self.image_size = image_size

        '''
        Embedding layers returns a 2d array with the embed of the class, like a look-up table.
        This way, the class embed works as a new channel.
        '''
        self.label_embed = nn.Embedding(num_classes, image_size*image_size)

        self.label_convs = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.input_conv = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # +1 due to conditional
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.main = nn.Sequential(
            # state size. (ndf*2) x 32 x 32
            self.downsample(in_ch=ndf*2),
            # state size. (ndf*4) x 16 x 16
            self.downsample(in_ch=ndf*4),
            # state size. (ndf*4) x 8 x 8
            self.downsample(in_ch=ndf*8),
            # state size. (ndf*8) x 4 x 4
            self.downsample(in_ch=ndf*16, last_layer=True)
        )

    def downsample(self, in_ch: int, last_layer: bool = False, bias: bool = False):
        layers = []
        if not last_layer:
            noise = GaussianNoise()
            conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch*2,
                            kernel_size=4, stride=2, padding=1, bias=bias)
            bn = nn.BatchNorm2d(num_features=in_ch*2)
            act = nn.LeakyReLU(0.2, inplace=True)
            layers.append(*[noise, conv, bn, act])
        else:
            conv = nn.Conv2d(in_channels=in_ch, out_channels=1, 
                             kernel_size=4, stride=1, padding=0, bias=bias)
            act = nn.Sigmoid()
            layers.append(*[conv, act])

        return nn.Sequential(*layers)


    def forward(self, input, labels):
        ## embedding and convolution of classes
        embedding=self.label_embed(labels)
        embedding = embedding.view(labels.shape[0], 1, self.image_size, self.image_size)
        embedding = self.label_convs(embedding)

        ## embedding and convolution of 
        input = self.input_conv(input)

        # concatenates the embedding with the number of channels (dimension 0)
        inp=torch.cat([input, embedding],1)

        output = self.main(inp)
        
        return output#.view(-1, 1).squeeze(1)


