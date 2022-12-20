# https://github.com/tczhangzhi/awesome-normalization-techniques/blob/6061d4d7a4b85b2568f1e2b10a10b25ad9248e12/conditional_batch_norm2d.py

import torch.nn as nn

class ConditionalBatchNorm2d(nn.Module):
    """Conditional Batch Normalization
    Parameters
        num_features – C from an expected input of size (N, C, H, W)
        num_classes – Number of classes in the datset.
        bias – if set to True, adds a bias term to the embedding layer
    Shape:
        Input: (N, C, H, W) (N, num_classes)
        Output: (N, C, H, W) (same shape as input)
    Examples:
        >>> m = ConditionalBatchNorm2d(100, 10)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> label = torch.randint(10, (20,))
        >>> output = m(input, label)
    """
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        if self.bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()
            self.embed.weight.data[:, num_features:].zero_()
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.bn(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=1)
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * out
        return out