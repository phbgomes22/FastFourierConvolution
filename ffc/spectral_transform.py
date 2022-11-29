
import torch
import torch.nn as nn
from .fourier_unity import FourierUnit 

'''
Used in the FFC classs,
It defines the flow of the Spectral Transform
Within, the Fourier Unit
'''
class SpectralTransform(nn.Module):

    # I changed the enable_lfu default value to False, as LaMa does not use it and it 
    # increases complexity to the architecture.
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=False):
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
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        # creates the Fourier Unit that will do convolutions in the spectral domain.
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        
        # creates the enable lfu, if set. I set the default to false.
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        ## sets the convolution that will occur at the end of the Spectral Transform
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)


    def forward(self, x):
        # the default behavior is no downsample - so this is an identity
        x = self.downsample(x)
        # the initial convolution with conv2(1x1), BN and ReLU
        x = self.conv1(x)
        # gets the output from the Fourier Unit (back in pixel domain)
        output = self.fu(x)

        # lfu is optional, and for the initial tests I will remove it
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

