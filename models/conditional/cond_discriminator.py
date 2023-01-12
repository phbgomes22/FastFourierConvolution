'''
Author: Pedro Gomes
'''

import torch.nn as nn
from util import *
from layers import *
from torch.nn.utils import spectral_norm


class CondSNDiscriminator(nn.Module):
    def __init__(self, nc: int, ndf: int, num_classes: int, image_size: int):
        super(CondSNDiscriminator, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes

        self.ylabel=nn.Sequential(
            spectral_norm(nn.Linear(num_classes, image_size*image_size)),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            spectral_norm(nn.Conv2d(nc+1, ndf * 2, 4, 2, 1, bias=False)), # +1 due to conditional
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            spectral_norm(nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=True)),
            nn.Sigmoid()
        )


    def forward(self, input, labels):
        y=self.ylabel(labels)

        y=y.view(labels.shape[0],1,64,64)

        inp=torch.cat([input,y],1)
        output = self.main(inp)
        
        return output.view(-1, 1).squeeze(1)





class CondBNDiscriminator(nn.Module):
    def __init__(self, nc: int, ndf: int, num_classes: int, image_size: int):
        super(CondBNDiscriminator, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes

        self.ylabel=nn.Sequential(
            nn.Linear(num_classes, image_size*image_size),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc+1, ndf * 2, 4, 2, 1, bias=False), # +1 due to conditional
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
        )
           # nn.BatchNorm2d(ndf * 2),
        self.cbn1 = ConditionalBatchNorm2d(ndf * 4, num_classes=num_classes)
        self.main2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True)
        )
           # nn.BatchNorm2d(ndf * 4),
        self.cbn2 = ConditionalBatchNorm2d(ndf * 8, num_classes=num_classes)

        self.main3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=True)
        )
            #nn.BatchNorm2d(ndf * 8),
            # Batch normalization conditioned to class
        self.cbn3 = ConditionalBatchNorm2d(ndf * 16, num_classes=num_classes)

        self.main4 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, input, labels):
        y=self.ylabel(labels)
        # revert one hot to labels to pass it to conditional batch norm
        discrete_labels = torch.argmax(labels, dim=1)

        y=y.view(labels.shape[0],1,64,64)

        inp=torch.cat([input,y],1)
        output = self.main(inp)
        
        output = self.cbn1(output, discrete_labels)
        output = self.main2(output)
        output = self.cbn2(output, discrete_labels)
        output = self.main3(output)
        output = self.cbn3(output, discrete_labels)
        output = self.main4(output)
        
        return output.view(-1, 1).squeeze(1)