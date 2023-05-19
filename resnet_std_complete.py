import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


# ResNet generator and discriminator
from torch import nn
import torch.nn.functional as F

import numpy as np

SpectralNorm = torch.nn.utils.spectral_norm

channels = 3

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

GEN_SIZE=128
DISC_SIZE=128

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
                nn.ReLU(),
                nn.AvgPool2d(8),
            )
        self.fc = nn.Linear(DISC_SIZE, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        return self.fc(self.model(x).view(-1,DISC_SIZE))

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

parser.add_argument('--model', type=str, default='resnet')

args = parser.parse_args()

loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

Z_dim = 128
#number of updates to discriminator for every update to generator 
disc_iters = 5

discriminator = Discriminator().cuda()
generator = Generator(Z_dim).cuda()

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

def train(epoch):
    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data, target = Variable(data.cuda()), Variable(target.cuda())

        # update discriminator
        for _ in range(disc_iters):
            z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            if args.loss == 'hinge':
                disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()
            elif args.loss == 'wasserstein':
                disc_loss = -discriminator(data).mean() + discriminator(generator(z)).mean()
            else:
                disc_loss = nn.BCEWithLogitsLoss()(discriminator(data), Variable(torch.ones(args.batch_size, 1).cuda())) + \
                    nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.zeros(args.batch_size, 1).cuda()))
            disc_loss.backward()
            optim_disc.step()

        z = Variable(torch.randn(args.batch_size, Z_dim).cuda())

        # update generator
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        if args.loss == 'hinge' or args.loss == 'wasserstein':
            gen_loss = -discriminator(generator(z)).mean()
        else:
            gen_loss = nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.ones(args.batch_size, 1).cuda()))
        gen_loss.backward()
        optim_gen.step()

        if batch_idx % 100 == 0:
            print('disc loss', disc_loss.data[0], 'gen loss', gen_loss.data[0])
    scheduler_d.step()
    scheduler_g.step()

fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
def evaluate(epoch):

    samples = generator(fixed_z).cpu().data.numpy()[:64]


    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)

    if not os.path.exists('out/'):
        os.makedirs('out/')

    plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)

os.makedirs(args.checkpoint_dir, exist_ok=True)

for epoch in range(2000):
    train(epoch)
    evaluate(epoch)
    torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))
