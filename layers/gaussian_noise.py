import torch
import torch.nn as nn
from util import *
#https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/3

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = x.data.new(x.size()).normal_(0, self.stddev)
            return x + noise
        return x