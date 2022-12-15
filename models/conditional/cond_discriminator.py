'''
Author: Pedro Gomes
'''

import torch.nn as nn
from util import *
from ffc import *
from .cond_bn import ConditionalBatchNorm2d


class CondDiscriminator(nn.Module):
    def __init__(self, nc: int, ndf: int, num_classes: int, image_size: int):
        super(CondDiscriminator, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes

        self.ylabel=nn.Sequential(
            nn.Linear(num_classes, image_size*image_size),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc+1, ndf, 4, 2, 1, bias=False), # +1 due to conditional
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        )
           # nn.BatchNorm2d(ndf * 2),
        self.cbn1 = ConditionalBatchNorm2d(ndf * 2, num_classes=num_classes)
        self.main2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        )
           # nn.BatchNorm2d(ndf * 4),
        self.cbn2 = ConditionalBatchNorm2d(ndf * 4, num_classes=num_classes)

        self.main3 = (
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        )
            #nn.BatchNorm2d(ndf * 8),
            # Batch normalization conditioned to class
        self.cbn3 = ConditionalBatchNorm2d(ndf * 8, num_classes=num_classes)

        self.main4 = (
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, input, labels):
        print(labels)
        y=self.ylabel(labels)
        y=y.view(labels.shape[0],1,64,64)

        inp=torch.cat([input,y],1)
        output = self.main(inp)
        y_input = torch.tensor(labels).long()
        output = self.cbn1(output, y_input)
        output = self.main2(output)
        output = self.cbn2(output, y_input)
        output = self.main3(output)
        output = self.cbn3(output, y_input)
        output = self.main4(output)
        
        return output.view(-1, 1).squeeze(1)