'''
Authors: Chi, Lu and Jiang, Borui and Mu, Yadong
Adaptations: Pedro Gomes 
'''

import torch
import torch.nn as nn
from ..cond.cond_bn import *

## LaMA model uses this, what is this?
## Test, if it improves, study!
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res


class FourierUnitSN(nn.Module):
    def __init__(self, in_channels, out_channels, groups: int = 1, num_classes: int = 1):
        # bn_layer not used
        super(FourierUnitSN, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        if num_classes > 1:
            self.bn = ConditionalBatchNorm2d(out_channels * 2, num_classes)
        else: 
            self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        self.se = SELayer(self.conv_layer.in_channels)

    def forward(self, x, y = None):
        batch, c, h, w = x.size()
        r_size = x.size()

        # with rfftn, dim = (-2, -1)
        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft(x, dim=2, norm="ortho")
        # (batch, c, 2, h, w/2+1)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        if y is not None: 
             ffted = self.relu(self.bn(ffted, y))
        else: 
            ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

        # with irfftn, dim = (-2, -1)
        output = torch.fft.irfft(ffted, dim=2, norm="ortho")

        return output

'''
The deepest block in the class hierarchy. 
It represents the flow of getting the Fourier Transform of the global signal ->
Convolution, Batch Normalization and ReLu in the spectral domain ->
Inverse Fourier Transform to return to pixel domain.
'''
# class FourierUnitSN(nn.Module):

#     def __init__(self, in_channels: int, out_channels: int, groups: int = 1, num_classes: int = 1):
#         # bn_layer not used
#         super(FourierUnitSN, self).__init__()
#         self.groups = groups

#         sn_fn = torch.nn.utils.spectral_norm
#         # the convolution layer that will be used in the spectral domain
#         self.conv_layer = sn_fn(torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
#                                           kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False))
#         # sn_fn()
#         # batch normalization for the spectral domain
#         if num_classes > 1:
#             self.bn = ConditionalBatchNorm2d(out_channels * 2, num_classes)
#         else: 
#             self.bn = torch.nn.BatchNorm2d(out_channels * 2)
#         # relu for the spectral domain
#         self.relu = torch.nn.ReLU(inplace=True)


#     def forward(self, x, y = None):
#         batch, c, h, w = x.size()
#         r_size = x.size()

#         # (batch, c, h, w/2+1) complex number
#         ffted = torch.fft.rfftn(x,s=(h,w),dim=(2,3),norm='ortho')
#         ffted = torch.cat([ffted.real,ffted.imag],dim=1)

#         ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
#         if y is not None: 
#              ffted = self.relu(self.bn(ffted, y))
#         else: 
#             ffted = self.relu(self.bn(ffted))

#         ffted = torch.tensor_split(ffted,2,dim=1)
#         ffted = torch.complex(ffted[0],ffted[1])
#         output = torch.fft.irfftn(ffted,s=(h,w),dim=(2,3),norm='ortho')

#         return output 