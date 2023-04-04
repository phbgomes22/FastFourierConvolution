'''
Authors: Chi, Lu and Jiang, Borui and Mu, Yadong
Adaptations: Pedro Gomes 
'''

import torch
import torch.nn as nn
from .fourier_unity import FourierUnit, FourierUnitSN

'''
Used in the FFC classs,
It defines the flow of the Spectral Transform
Within, the Fourier Unit
'''
class SpectralTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                stride: int = 1, groups: int = 1, enable_lfu: bool = True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu

        # sets a downsample if the stride is set to 2 (default is one)
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride

        # sets the initial 1x1 convolution, batch normalization and relu flow.
        self.conv1 = nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.act1 = nn.ReLU(inplace=True)

        # creates the Fourier Unit that will do convolutions in the spectral domain.
        self.fu = FourierUnitSN(
            out_channels // 2, out_channels // 2, groups)
        
        # creates the enable lfu, if set. I set the default to false.
        if self.enable_lfu:
            self.lfu = FourierUnitSN(
                out_channels // 2, out_channels // 2, groups)
        
        ## sets the convolution that will occur at the end of the Spectral Transform
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)


    def forward(self, x):
        # the default behavior is no downsample - so this is an identity
        x = self.downsample(x)
        # the initial convolution with conv2(1x1), BN and ReLU
        x = self.act1(self.bn1(self.conv1(x)))
        # gets the output from the Fourier Unit (back in pixel domain)
        output = self.fu(x)

        # lfu is optional
        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        # does the final 1x1 convolution with the residual connection (x + output)
        output = self.conv2(x + output + xs)

        return output


class SpectralTransformSN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                stride: int = 1, groups: int = 1, enable_lfu: bool = True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu

        # sets a downsample if the stride is set to 2 (default is one)
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride

        sn_fn = torch.nn.utils.spectral_norm
        # sets the initial 1x1 convolution, batch normalization and relu flow.
        self.conv1 = sn_fn(nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False))
        self.bn1 = nn.Identity()#BatchNorm2d(out_channels // 2)
        self.act1 = nn.ReLU(inplace=True)

        # creates the Fourier Unit that will do convolutions in the spectral domain.
        self.fu = FourierUnitSN(
            out_channels // 2, out_channels // 2, groups)
        
        # creates the enable lfu, if set. I set the default to false.
        if self.enable_lfu:
            self.lfu = FourierUnitSN(
                out_channels // 2, out_channels // 2, groups)
        
        ## sets the convolution that will occur at the end of the Spectral Transform
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)


    def forward(self, x):
        # the default behavior is no downsample - so this is an identity
        x = self.downsample(x)
        # the initial convolution with conv2(1x1), BN and ReLU
        x = self.act1(self.bn1(self.conv1(x)))
        # gets the output from the Fourier Unit (back in pixel domain)
        output = self.fu(x)

        # lfu is optional
        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        # does the final 1x1 convolution with the residual connection (x + output)
        output = self.conv2(x + output + xs)

        return output
