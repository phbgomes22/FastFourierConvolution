'''
Authors: Chi, Lu and Jiang, Borui and Mu, Yadong
Adaptations: Pedro Gomes 
'''

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from ..cond.cond_bn import ConditionalBatchNorm2d


'''
The deepest block in the class hierarchy. 
It represents the flow of getting the Fourier Transform of the global signal ->
Convolution, Batch Normalization and ReLu in the spectral domain ->
Inverse Fourier Transform to return to pixel domain.
'''
class SNFourierUnit(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, groups: int = 1):
        # bn_layer not used
        super(SNFourierUnit, self).__init__()
        self.groups = groups

        # the convolution layer that will be used in the spectral domain
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        # spectral normalization in the spectral domain
        # relu for the spectral domain
        self.relu = torch.nn.ReLU(inplace=True)


    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()

        # (batch, c, h, w/2+1) complex number
        ffted = torch.fft.rfftn(x,s=(h,w),dim=(2,3),norm='ortho')
        ffted = torch.cat([ffted.real,ffted.imag],dim=1)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(spectral_norm(ffted))

        ffted = torch.tensor_split(ffted,2,dim=1)
        ffted = torch.complex(ffted[0],ffted[1])
        output = torch.fft.irfftn(ffted,s=(h,w),dim=(2,3),norm='ortho')

        return output 