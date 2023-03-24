'''
Authors: Pedro Gomes 
'''

import torch
import torch.nn as nn
from .print_layer import Print


class Resizer(nn.Module):
    def __init__(self, debug = False):
        super(Resizer, self).__init__()
        self.print_size = Print(debug=debug)

    def forward(self, x):
        output = x
        if type(x) == tuple:
            if type(x[1]) == int: # x[1] == 0
                output = x[0]
            else:
                output = torch.cat(list(x), dim=1)
               # output = x.view(x.shape[0], -1, *x.shape[3:])
                x = self.print_size(output)
        return output
