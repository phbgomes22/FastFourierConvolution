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
import tqdm

from models import *
from util import *


import torch_fidelity

# ResNet generator and discriminator
from torch import nn
import torch.nn.functional as F

import numpy as np

SpectralNorm = torch.nn.utils.spectral_norm

channels = 3

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class FFCResBlockGenerator(FFCModel):

    def __init__(self, channels: int, gin: float, gout: float, stride: int = 1):
        super(FFCResBlockGenerator, self).__init__()
        self.gin = gin
        self.gout = gout

        in_ch_l = int(channels * (1 - gin))
        in_ch_g = int(channels * gin)
        out_ch_l = int(channels * (1 - gout))
        out_ch_g = int(channels * gout)

        kernel_size = 3
        self.ffc_conv1 = FFC(channels, channels, kernel_size, gin, 0.5, stride=stride, padding=1)
        self.ffc_conv2 = FFC(channels, channels, kernel_size, 0.5, gout, stride=stride, padding=1)
        ## init xavier uniform now inside of FFC

        self.bnl1 = nn.Identity if gin == 1 else nn.BatchNorm2d(in_ch_l)
        self.bnl2 = nn.BatchNorm2d(channels * 0.5)
        
        self.bng1 = nn.Identity if gin == 0 else nn.BatchNorm2d(in_ch_g)
        self.bng2 = nn.BatchNorm2d(channels * 0.5)
        
        self.relul1 = nn.Identity if gin == 1 else nn.ReLU(inplace=True)
        self.relul2 = nn.ReLU(inplace=True)
        self.relug1 = nn.Identity if gin == 0 else nn.ReLU(inplace=True)
        self.relug2 = nn.ReLU(inplace=True)

        self.upsample_l = nn.Upsample(scale_factor=2)
        self.upsample_g = nn.Identity if gin == 0 else nn.Upsample(scale_factor=2)
        
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # breaking x into x_l and x_g
        x_l, x_g = x if type(x) is tuple else (x, 0)
        
        # local BN and ReLU before first convolution
        x_l_out = self.relul1(self.bnl1(x_l))
        x_l_out = self.upsample_l(x_l_out)
        # global BN and ReLU before first convolution
        x_g_out = self.relug1(self.bng1(x_g))
        x_g_out = self.upsample_g(x_g_out)

        # first convolution
        x_l_out, x_g_out = self.ffc_conv1(x_l_out, x_g_out)

        # local and global BN and ReLU after the first convolution
        x_l_out = self.relul2(self.bnl2(x_l_out))
        x_g_out = self.relug2(self.bng2(x_g_out))
        
        # second convolution
        x_l_out, x_g_out = self.ffc_conv2(x_l_out, x_g_out)
    
        # adds the residual connection for both global and local
        x_l_out = x_l_out + self.bypass(x_l)
        x_g_out = x_g_out + self.bypass(x_g)

        return x_l_out, x_g_out


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

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
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

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

class FGenerator(nn.Module):
    def __init__(self, z_dim):
        super(FGenerator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)

        self.resblock1 = FFCResBlockGenerator(GEN_SIZE, 0, 0.5, stride=2)
        self.resblock2 = FFCResBlockGenerator(GEN_SIZE, 0.5, 0.5, stride=2)
        self.resblock3 = FFCResBlockGenerator(GEN_SIZE, 0.5, 0, stride=2)

        self.final_bn = nn.BatchNorm2d(GEN_SIZE)
        self.final_relu = nn.ReLU()
        self.final_conv = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)

        nn.init.xavier_uniform_(self.final_conv.weight.data, 1.)
        self.final = nn.Sequential(
            self.final_conv,
            nn.Tanh()
        )

    def forward(self, z):
        # passes thorugh linear layer
        features = self.dense(z).view(-1, GEN_SIZE, 4, 4)

        # ffc blocks of resnet
        fake = self.resblock1(features)
        fake = self.resblock2(fake)
        fake = self.resblock3(fake)
        fake = self.resizer(fake)

        # last batch norm and relu 
        # -- TODO: only resize after local and global BN and ReLU
        fake = self.final_relu(self.final_bn(fake))
        # -- TODO: transform this last convolution in FFC Conv too
        fake = self.final(fake)

        if not self.training:
            fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
            fake = fake.to(torch.uint8)
        return fake

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
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        return self.fc(self.model(x).view(-1,DISC_SIZE))

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--num_total_steps', type=int, default=100000)

parser.add_argument('--model', type=str, default='resnet')

args = parser.parse_args()

loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

Z_dim = 128
#number of updates to discriminator for every update to generator 
disc_iters = 5

discriminator = Discriminator().cuda()
generator = FGenerator(Z_dim).cuda()

d_params = count_parameters(discriminator)
g_params = count_parameters(generator)
print("Parameters on Discriminator: ", d_params, " \nParameters on Generator: ", g_params)

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.5,0.999))
optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5,0.999))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

leading_metric, last_best_metric, metric_greater_cmp = {
        'ISC': (torch_fidelity.KEY_METRIC_ISC_MEAN, 0.0, float.__gt__),
        'FID': (torch_fidelity.KEY_METRIC_FID, float('inf'), float.__lt__),
        'KID': (torch_fidelity.KEY_METRIC_KID_MEAN, float('inf'), float.__lt__),
        'PPL': (torch_fidelity.KEY_METRIC_PPL_MEAN, float('inf'), float.__lt__),
    }['ISC']



def train():

    pbar = tqdm.tqdm(total=args.num_total_steps, desc='Training', unit='batch')

    loader_iter = iter(loader)

    for step in range(args.num_total_steps):
        try:
            real_img, real_label = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            real_img, real_label = next(loader_iter)

        data = Variable(real_img.cuda())
        target = Variable(real_label.cuda())

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

        next_step = step + 1

        if next_step  % 500 == 0:
            step_info = {'disc loss': disc_loss.cpu().item(), 'gen loss': gen_loss.cpu().item()}
            pbar.set_postfix(step_info)
        pbar.update(1)

        if next_step % 1000 == 0: 
            pbar.close()
            generator.eval()
            evaluate(next_step)

            if next_step % 5000 == 0:
                print('Evaluating the generator...')

                # compute and log generative metrics
                metrics = torch_fidelity.calculate_metrics(
                    input1=torch_fidelity.GenerativeModelModuleWrapper(generator, Z_dim, 'normal', 0),
                    input1_model_num_samples=5000,
                    input2= 'cifar10-train',
                    isc=True,
                    fid=True,
                    kid=True,
                    ppl=False,
                    ppl_epsilon=1e-2,
                    ppl_sample_similarity_resize=64,
                )

            pbar = tqdm.tqdm(total=args.num_total_steps, initial=next_step, desc='Training', unit='batch')
            generator.train()

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
        plt.imshow(sample.transpose((1,2,0))) # * 0.5 + 0.5

    if not os.path.exists('out/'):
        os.makedirs('out/')

    plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)

os.makedirs(args.checkpoint_dir, exist_ok=True)


train()
torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(1)))
torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(1)))
