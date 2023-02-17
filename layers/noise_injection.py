# ## FROM
# ## https://github.com/JackMcCoy/riemann-noise-pytorch/blob/master/riemann-noise-pytorch/riemann_noise_pytorch.py

import torch
import torch.nn as nn

## FROM:
## https://discuss.pytorch.org/t/add-noise-to-layer-output/127876


# ## FROM
# ## https://github.com/JackMcCoy/riemann-noise-pytorch/blob/master/riemann-noise-pytorch/riemann_noise_pytorch.py

import torch
import torch.nn as nn

## FROM:
## https://discuss.pytorch.org/t/add-noise-to-layer-output/127876

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = x.new_empty(batch, 1, height, width).normal_()

        noise_to_add = self.weight * noise

        return x + noise_to_add