# https://github.com/tczhangzhi/awesome-normalization-techniques/blob/6061d4d7a4b85b2568f1e2b10a10b25ad9248e12/conditional_batch_norm2d.py
import torch
import torch.nn as nn

class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes):
    super().__init__()
    self.num_classes = num_classes
    self.num_features = num_features
    self.bn = nn.BatchNorm2d(num_features, affine=False)
    self.embed = nn.Embedding(num_classes, num_features * 2)
    
    self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
    self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

  def forward(self, x, y):
    y = torch.squeeze(y)
    y = torch.squeeze(y)
    ## conditional
    out = self.bn(x)
    gamma, beta = self.embed(y).chunk(2, 1)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out