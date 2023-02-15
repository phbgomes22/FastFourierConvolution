'''
Author: Pedro Gomes
'''

import torch.nn as nn
from util import *
from layers import *
from ..ffcmodel import FFCModel
import math
from torch.nn.utils import spectral_norm


## - This is the one bringing good results!
class FFCCondDiscriminator(FFCModel):
    def __init__(self, nc: int, ndf: int, num_classes: int, 
                num_epochs: int, uses_sn: bool = False, uses_noise: bool = False):
        super(FFCCondDiscriminator, self).__init__( debug=False)
        self.ndf = ndf
        self.num_epochs = num_epochs
        self.uses_sn = uses_sn
        '''
        Embedding layers returns a 2d array with the embed of the class, 
        like a look-up table.
        This way, the class embed works as a new channel.
        '''
        self.label_embed = nn.Embedding(num_classes, self.ndf*self.ndf)

        '''
        why - 2? the first convolution has 1 padding and stride 2 
        ie: it moves from a 64x64 dim to a 32x32 dim
        so we would subtract -1, the extra -1 is for the last layer.
        '''
        self.number_convs = int(math.log2(ndf)) - 2

        self.label_convs = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.input_conv = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # +1 due to conditional
            nn.LeakyReLU(0.2, inplace=True),
        )

        ## States if Cond Discriminator uses noise
        self.uses_noise = uses_noise

        ## Initial std value
        self.noise_stddev = 0.1
        
        ## Noise decay hyperparameter
        self.noise_decay = 0.01

        self.main = self.create_layers(ndf)


    def create_layers(self, ndf: int):
        layers = []

        # adds the hidden layers
        for itr in range(1, self.number_convs):
            mult = int(math.pow(2, itr)) # 2^iter
            g_in = 0 #if itr == 1 else 0.5
            debug_print("in_channels: ", ndf*mult)
            layers.append(
                FFC_BN_ACT(in_channels=ndf*mult, out_channels=ndf*mult*2, kernel_size=4,
                ratio_gin=g_in, ratio_gout=0.0, stride=2, padding=1, bias=False, 
                uses_sn=self.uses_sn, activation_layer=nn.LeakyReLU)
            )

        # adds the last layer
        mult = int(math.pow(2, self.number_convs))

        conv = nn.Conv2d(in_channels=ndf*mult, out_channels=1, 
                             kernel_size=4, stride=1, padding=0, bias=False)

        act = nn.Sigmoid()
        layers.extend([conv, act])
        # layers.append(
        #     FFC_BN_ACT(in_channels=ndf*mult, out_channels=1, kernel_size=4,
        #         ratio_gin=0.0, ratio_gout=0, stride=1, padding=0, bias=False, 
        #         uses_sn=self.uses_sn, activation_layer=nn.Sigmoid)
        # )

        return nn.Sequential(*layers)


    def get_noise_decay(self, epoch: int):
        return self.noise_decay ** (epoch / self.num_epochs)


    def forward(self, input, labels, epoch: int):
        ## embedding and convolution of classes
        embedding = self.label_embed(labels)
        embedding = embedding.view(labels.shape[0], 1, self.ndf, self.ndf)
        embedding = self.label_convs(embedding)

        if self.uses_noise:
            ## add noise to input of discriminator
            noise = torch.randn_like(input) * self.noise_stddev * self.get_noise_decay(epoch)
            input = input + noise
            
        ## run the input through the first convolution
        input = self.input_conv(input)

        # concatenates the embedding with the number of channels (dimension 0)
        inp=torch.cat([input, embedding],1)

        output = self.main(inp)
        
      #  output = self.resizer(output)

        return output