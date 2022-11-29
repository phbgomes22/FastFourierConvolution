import torch.nn as nn
from util import *


class FFCModel(nn.Module):
    def __init__(self, inplanes, debug=False):
        super(FFCModel, self).__init__()
        
        self.inplanes = inplanes
        self.lfu = False # Testei true e false, false pareceu melhor
        self.groups = 1
        self.use_se = False
        self.base_width = 64
        self.dilation = 1
        self.debug = debug

        self.print_size = nn.Sequential(Print(debug=self.debug))
        
        self.resizer = Resizer()
    
    def forward(self, x):
        pass