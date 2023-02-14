'''
Authors: Chi, Lu and Jiang, Borui and Mu, Yadong
Adaptations: Pedro Gomes 
'''

import torch.nn as nn
from util import *
from layers import *
from config import *


class FFCModel(nn.Module):
    '''
    Sets default values for the FFC Models. 
    Both FFC-Generator and FFC-Discriminator inherits from this class.
    '''
    def __init__(self, debug=False):
        super(FFCModel, self).__init__()
        
        self.lfu = True
        self.groups = 1
        self.use_se = False
        self.base_width = 64
        self.dilation = 1
        self.debug = debug

        self.print_size = Print(debug=Config.shared().DEBUG)
        self.resizer = Resizer()
    
    def forward(self, x):
        pass