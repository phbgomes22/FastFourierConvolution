'''
Authors: Pedro Gomes 
'''

import torch
import torch.nn as nn
from config import Config


def debug_print(*txt):
    if Config.shared().DEBUG:
        print(*txt)

class Print(nn.Module):
    def __init__(self, debug = False):
        super(Print, self).__init__()
        self.debug = debug

    def forward(self, x):
        if self.debug:
            if type(x) == tuple:
                if type(x[1]) == int:
                    aux = x[0]
                    print(aux.shape, "global = 0")
                else:
                    aux  = torch.cat(list(x), dim=1) # cat was stack before, changing
                    aux = aux.view(aux.shape[0], -1, *aux.shape[3:])
                    print(aux.shape)
            else:
                print(x.shape)
    
        return x
