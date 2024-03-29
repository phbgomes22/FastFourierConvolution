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

    def __init__(self, nz: int, nc: int, ngf: int, num_classes: int, embed_size: int, uses_sn: bool = False, uses_noise: bool = False, training: bool = True):
        super(FFCCondGenerator, self).__init__(debug=False)
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.uses_noise = uses_noise
        self.uses_sn = uses_sn
        self.training = training

        self.label_embed = nn.Embedding(num_classes, num_classes)

        # why - 3? 
        # the first convolution has no padding and stride 1 
        # -ie: it moves from a 1x1 dim to a 4x4 dim
        # so we would subtract -2, the extra -1 is for the last layer.
        self.number_convs = int(math.log2(ngf)) - 3

        # initial convolutions are concatenated leaving ch + ch = 2*ch in beginning of main(x)
        mult = int(math.pow(2, self.number_convs - 1))

        self.label_conv = nn.Sequential(
            nn.ConvTranspose2d(num_classes, ngf*mult, 4, 1, 0),
            nn.BatchNorm2d(ngf*mult),
            nn.GELU()
        )
        self.input_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf*mult, 4, 1, 0),
            nn.BatchNorm2d(ngf*mult),
            nn.GELU()
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
                               activation_layer=nn.GELU, 
                               upsampling=True),
            )
        # - testing last ffc convolution with full image size 
        layers.append(
            FFC_BN_ACT(ngf, ngf, 4, 0.5, 0.5, stride=2, padding=1, activation_layer=nn.GELU, 
                      upsampling=True, uses_noise=self.uses_noise, uses_sn=self.uses_sn, attention=True) 
        ) 
        # adds the last layer
        layers.append(
            FFC_BN_ACT(ngf*1, nc, 3, 0.5, 0, stride=1, padding=1, 
                       norm_layer=nn.Identity, activation_layer=nn.Tanh,
                       uses_noise=self.uses_noise, uses_sn=self.uses_sn, attention=True) 
        )

        return nn.Sequential(*layers)

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


    def reshape_output(self, output):
        '''
        Reshape output for FID calculations.

        This function should not alter the behavior of the training routine.
        '''
        if not self.training:
            if self.nc == 1:
                ## gets the number of ones in the repeat
                size_ones = (1,) * (len(output.shape) - 3)
                ## repeat the color value, and leave the rest the same
                end_of_repeat = (3, 1, 1) 
                ## transforms grayscale to RGB by making it r==g==b
                output = output.repeat(*size_ones + end_of_repeat)

            output = (255 * (output.clamp(-1, 1) * 0.5 + 0.5))
            output = output.to(torch.uint8)

        return output

    def forward(self, input, labels):

        input = self.assert_input(input)

        debug_print("** FFC_COND_GENERATOR")
        ## allows a subset of classes in the dataset
        labels = torch.remainder(labels, self.num_classes)
        debug_print(labels)

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

        self.print_size(x)

        debug_print("** END FFC_COND_GENERATOR")

        x = self.reshape_output(x)

        return x



# - This is the one bringing good results!
class GeneratorFGAN(FFCModel):

    def __init__(self, nz: int, nc: int, ngf: int, num_classes: int, embed_size: int, uses_sn: bool = False, uses_noise: bool = False):
        super(GeneratorFGAN, self).__init__(debug=False)
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.uses_noise = uses_noise
        self.uses_sn = uses_sn

        self.label_embed = nn.Embedding(num_classes, num_classes)

        # why - 3? 
        # the first convolution has no padding and stride 1 
        # -ie: it moves from a 1x1 dim to a 4x4 dim
        # so we would subtract -2, the extra -1 is for the last layer.
        self.number_convs = int(math.log2(ngf)) - 3

        # initial convolutions are concatenated leaving ch + ch = 2*ch in beginning of main(x)
        mult = int(math.pow(2, self.number_convs - 1))

        self.label_conv = nn.Sequential(
            nn.ConvTranspose2d(num_classes, ngf*mult, 4, 1, 0),
            nn.BatchNorm2d(ngf*mult),
            nn.GELU()
        )
        self.input_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf*mult, 4, 1, 0),
            nn.BatchNorm2d(ngf*mult),
            nn.GELU()
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
                               activation_layer=nn.GELU, 
                               upsampling=True),
            )
        # - testing last ffc convolution with full image size 
        layers.append(
            FFC_BN_ACT(ngf, ngf, 4, 0.5, 0.5, stride=2, padding=1, activation_layer=nn.GELU, 
                      upsampling=True, uses_noise=self.uses_noise, uses_sn=self.uses_sn, attention=True) 
        ) 
        # adds the last layer
        layers.append(
            FFC_BN_ACT(ngf*1, nc, 3, 0.5, 0, stride=1, padding=1, 
                       norm_layer=nn.Identity, activation_layer=nn.Tanh,
                       uses_noise=self.uses_noise, uses_sn=self.uses_sn, attention=True) 
        )

        return nn.Sequential(*layers)

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


    def reshape_output(self, output):
        '''
        Reshape output for FID calculations.

        This function should not alter the behavior of the training routine.
        '''
        if not self.training:
            if self.nc == 1:
                ## gets the number of ones in the repeat
                size_ones = (1,) * (len(output.shape) - 3)
                ## repeat the color value, and leave the rest the same
                end_of_repeat = (3, 1, 1) 
                ## transforms grayscale to RGB by making it r==g==b
                output = output.repeat(*size_ones + end_of_repeat)

            output = (255 * (output.clamp(-1, 1) * 0.5 + 0.5))
            output = output.to(torch.uint8)

        return output

    def forward(self, input, labels):

        input = self.assert_input(input)
        ## allows a subset of classes in the dataset
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

        x = self.reshape_output(x)

        return x