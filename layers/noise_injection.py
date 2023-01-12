# ## FROM
# ## https://github.com/JackMcCoy/riemann-noise-pytorch/blob/master/riemann-noise-pytorch/riemann_noise_pytorch.py

import torch
import torch.nn as nn

# class RiemannNoise(nn.Module):
#     def __init__(self, size:int, channels:int, device: torch.device):
#         super(RiemannNoise, self).__init__()
#         '''
#         Initializes the module, taking 'size' as input for defining the matrix param.
#         '''
#         self.device = device
#         if type(size) == int:
#             h = w = size
#         elif type(size) == tuple and (type(x)==int for x in size):
#             h = size[-2]
#             w = size[-1]
#         else:
#             raise ValueError("Module must be initialized with a valid int or tuple of ints indicating the input dimensions.")
#         self.params = nn.ParameterList([nn.Parameter(nn.init.normal_(torch.ones(h, w))),
#                                         nn.Parameter(nn.init.constant(torch.ones(channels, ), 0)),
#                                         nn.Parameter(nn.init.normal_(torch.ones(h, w))),
#                                         nn.Parameter(nn.init.constant_(torch.ones(1, ), .5)),
#                                         nn.Parameter(nn.init.constant_(torch.ones(1, ), .5)),
#                                         nn.Parameter(nn.init.constant_(torch.ones(1, ), .5))])
#         self.noise = torch.zeros(1, device=device)

#     def forward(self, x):
#         N, c, h, w = x.shape
#         A, ch, b, alpha, r, w = self.params
#         s, _ = torch.max(-x, dim=1, keepdim=True)
#         s = s - s.mean(dim=(2, 3), keepdim=True)
#         s_max = torch.abs(s).amax(dim=(2, 3), keepdim=True)
#         s = s / (s_max + 1e-8)
#         s = (s + 1) / 2
#         s = s * A + b
#         s = torch.tile(s, (1, c, 1, 1))
#         sp_att_mask = alpha + (1 - alpha) * s
#         sp_att_mask = sp_att_mask * torch.rsqrt(
#             torch.mean(torch.square(sp_att_mask), axis=(2, 3), keepdims=True) + 1e-8)
#         sp_att_mask = r * sp_att_mask
#         x = x + (self.noise.repeat(*x.size()).normal_() * w)
#         x = x * sp_att_mask
#         return 



## FROM:
## https://discuss.pytorch.org/t/add-noise-to-layer-output/127876

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = x.new_empty(batch, 1, height, width).normal_()
        return x + self.weight * noise