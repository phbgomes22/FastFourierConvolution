import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch

class SNDiscriminator(nn.Module):
    '''
    The SNDiscriminator model - running with regular convolutions, but with Spectral Normalization.
    '''
    def __init__(self, nc: int, ndf: int, ngpu: int = 1):
        '''
        `nc`: number of color channels (1 for grayscale, 3 for colored images),
        `ndf`: size of feature maps in the discriminator - same as the image siz (64),
        `ngpu`: number of available gpus.
        '''
        super(SNDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True)),
          # # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True)),
        # #  nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True)),
         # #  nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True)),
            nn.Sigmoid()
        )

    def forward(self, input):
        ## TEST ADDING NOISE HERE
        ## -- 
        # mean = 0.
        # std = 10.
        # noise = (torch.randn(1, 3, 64, 64) + mean) * std
        # input = input + noise
        return self.main(input)