"""
Implementation of SNGAN for image size 48.
"""
import torch
import torch.nn as nn

from torch_mimicry.modules.layers import SNLinear
from torch_mimicry.modules.resblocks import DBlockOptimized, DBlock, GBlock
from torch_mimicry.nets.sngan import sngan_base


class SNGANGenerator48(sngan_base.SNGANBaseGenerator):
    r"""
    ResNet backbone generator for SNGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self, nz=128, ngf=512, bottom_width=6, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        # Build the layers
        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf >> 1, upsample=True)
        self.block3 = GBlock(self.ngf >> 1, self.ngf >> 2, upsample=True)
        self.block4 = GBlock(self.ngf >> 2, self.ngf >> 3, upsample=True)
        self.b5 = nn.BatchNorm2d(self.ngf >> 3)
        self.c5 = nn.Conv2d(self.ngf >> 3, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.

        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).

        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = torch.tanh(self.c5(h))

        if not self.training:
            min_val = float(h.min())
            max_val = float(h.max())
            h.clamp_(min=min_val, max=max_val)
            h.add_(-min_val).div_(max_val - min_val + 1e-5)
            
            h = h.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to(torch.uint8)


        return h


class SNGANDiscriminator48(sngan_base.SNGANBaseDiscriminator):
    r"""
    ResNet backbone discriminator for SNGAN.

    Attribates:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self, ndf=1024, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        # Build layers
        self.block1 = DBlockOptimized(3, self.ndf >> 4)
        self.block2 = DBlock(self.ndf >> 4, self.ndf >> 3, downsample=True)
        self.block3 = DBlock(self.ndf >> 3, self.ndf >> 2, downsample=True)
        self.block4 = DBlock(self.ndf >> 2, self.ndf >> 1, downsample=True)
        self.block5 = DBlock(self.ndf >> 1, self.ndf, downsample=False)
        self.l5 = SNLinear(self.ndf, 1)

        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)

        # Global sum pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l5(h)

        return output
