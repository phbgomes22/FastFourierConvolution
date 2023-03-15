'''
Author: Pedro Gomes
'''

import torch.nn as nn
from util import *
from layers import *
import math
from torch.nn.utils import spectral_norm


## - This is the one bringing good results!
class CondDiscriminator(nn.Module):
    def __init__(self, nc: int, ndf: int, num_classes: int, num_epochs: int, uses_sn: bool = False, uses_noise: bool = False):
        super(CondDiscriminator, self).__init__()
        self.ndf = ndf
        self.uses_sn = uses_sn
        self.num_epochs = num_epochs
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

        self.print_size = Print(debug=Config.shared().DEBUG)

        self.main = self.create_layers(ndf)

    def create_layers(self, ndf: int):
        layers = []
        # adds the hidden layers
        for itr in range(1, self.number_convs):
            mult = int(math.pow(2, itr)) # 2^iter
            layers.append(
                self.downsample(in_ch=ndf*mult)
            )

        # adds the last layer
        mult = int(math.pow(2, self.number_convs))
        layers.append(
            self.downsample(in_ch=ndf*mult, last_layer=True)
        )

        return nn.Sequential(*layers)

    def downsample(self, in_ch: int, last_layer: bool = False, bias: bool = False):
        layers = []
        if not last_layer:
            conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch*2,
                            kernel_size=4, stride=2, padding=1, bias=bias)
            bn = nn.BatchNorm2d(num_features=in_ch*2)
            # if we are using spectral normalization, removes batch norm
            # and adds spectral normalization instead
            if self.uses_sn:
                conv = spectral_norm(conv)
                bn = nn.Identity()

            act = nn.LeakyReLU(0.2, inplace=True)
            layers.extend([conv, bn, act])
        else:
            conv = nn.Conv2d(in_channels=in_ch, out_channels=1, 
                             kernel_size=4, stride=1, padding=0, bias=bias)
            if self.uses_sn:
                conv = spectral_norm(conv)
            act = nn.Sigmoid()
            layers.extend([conv, act])

        return nn.Sequential(*layers)


    def get_noise_decay(self, epoch: int):
        return self.noise_decay ** (epoch / self.num_epochs)

    def assert_input(self, input):
        '''
        Check if the last dimensions of the noise tensor have width and height valued 1/
        If not, unsqueeze the tensor to add them.

        This was added due to the torch-fidelity framework to calculate FID, that gives to the 
        Generator a noise input of shape (batch_size, nz), while training gives the Generator
        a noise of shape (batch_size, nz, 1, 1). 
        
        This function should not alter the behavior of the training routine.
        '''
        if input.dim() == 4 and input.size(dim=2) == 1 and input.size(dim=3) == 1:
            return input
        else:
            debug_print("- tensor doesn't end with 1, 1")
            new_input = input.unsqueeze(-1).unsqueeze(-1)
            return new_input


    def forward(self, input, labels, epoch: int):
        ## assert input size
        input = self.assert_input(input)

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

        self.print_size(input)

        # concatenates the embedding with the number of channels (dimension 0)
        inp=torch.cat([input, embedding],1)

        self.print_size(inp)
        output = self.main(inp)

        self.print_size(output)
        
        return output


