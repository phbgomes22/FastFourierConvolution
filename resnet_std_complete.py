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
        print(x.shape)
        m = self.model(x)
        print(m.shape)
        b = self.bypass(x)
        print(b.shape)
        return m + b


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()
        self.downsample = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

 
        self.model = nn.Sequential(
            nn.ReLU(),
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2)
        )
            
        self.bypass = nn.Sequential()

        if  in_channels != out_channels:
            self.bypass_conv = SpectralNorm(nn.Conv2d(in_channels, out_channels, 1, 1, padding=0))
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))
        else:
            self.bypass_conv = nn.Identity()


    def forward(self, x):
        input = self.model(x)
        bp = self.bypass_conv(x)

        if self.downsample > 1:
            input = F.avg_pool2d(input, kernel_size=self.downsample)
            bp = F.avg_pool2d(bp, kernel_size=self.downsample)

        return input + bp

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
    
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 6 * 6 * 512)
        self.final = nn.Conv2d(64, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResBlockGenerator(512, 256, stride=2),
            ResBlockGenerator(256, 128, stride=2),
            ResBlockGenerator(128, 64, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        features = self.dense(z).view(-1, 512, 6, 6)
        fake = self.model(features)

        if not self.training:
            fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
            fake = fake.to(torch.uint8)
        return fake

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(channels, 64, stride=2),
                ResBlockDiscriminator(64, 128, stride=2),
                ResBlockDiscriminator(128, 256, stride=2),
                ResBlockDiscriminator(256, 512, stride=2),
                ResBlockDiscriminator(512, 1024, stride=1),
                nn.ReLU(),
            )
        self.fc = nn.Linear(1024, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        x = self.model(x)
        features = torch.sum(x, dim=(2,3)) # gloobal sum pooling

        return self.fc(features)
    
class DiscriminatorStrided(nn.Module):
    def __init__(self, enable_conditional=False):
        super().__init__()
        n_classes = 10 if enable_conditional else 0
        self.initial_down = SpectralNorm(nn.Conv2d(3, 32, 4, 2, padding=0)) # padding間違えた
        self.block1 = FirstResBlockDiscriminator(32, 64, 2)
        self.block2 = ResBlockDiscriminator(64, 128, 2)
        self.block3 = ResBlockDiscriminator(128, 256, 2)
        self.block4 = ResBlockDiscriminator(256, 512, 1)
        self.dense = nn.Linear(512, 1)
        if n_classes > 0:
            self.sn_embedding = nn.Embedding(n_classes, 512)
        else:
            self.sn_embedding = None

    def forward(self, inputs, y=None):
        x = self.initial_down(inputs)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.relu(x)

        features = torch.sum(x, dim=(2,3)) # gloobal sum pooling
        x = self.dense(features)
        if self.sn_embedding is not None:
            x = self.sn_embedding(features, x, y)
        return x

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--num_total_steps', type=int, default=100000)
parser.add_argument('--metrics_step', type=int, default=5000)

args = parser.parse_args()

ds_transform = transforms.Compose([
            transforms.Resize(size=(48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

print("Loading Dataset...")
loader = load_stl_unlabeled(args.batch_size, ds_transform, args.workers)
print("Dataset Loaded!")
register_dataset(image_size=48)

Z_dim = 128
#number of updates to discriminator for every update to generator 
disc_iters = 5

discriminator = Discriminator().cuda()
generator = Generator(Z_dim).cuda()

d_params = count_parameters(discriminator)
g_params = count_parameters(generator)
print("Parameters on Discriminator: ", d_params, " \nParameters on Generator: ", g_params)

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=2e-4, betas=(0.5,0.999)) #(0.5,0.999)
optim_gen  = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5,0.999)) #(0.5,0.999)

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.999)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.999)

# scheduler_g = torch.optim.lr_scheduler.LambdaLR(optim_gen, lambda step: 1. - step / args.num_total_steps)
# scheduler_d = torch.optim.lr_scheduler.LambdaLR(optim_disc, lambda step: 1. - step / args.num_total_steps)

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
    #for batch_idx, (data, target) in enumerate(loader):
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

        if next_step % args.metrics_step == 0: 
            pbar.close()
            generator.eval()

            print('Evaluating the generator...')

            # compute and log generative metrics
            metrics = torch_fidelity.calculate_metrics(
                input1=torch_fidelity.GenerativeModelModuleWrapper(generator, Z_dim, 'normal', 0),
                input1_model_num_samples=5000,
                input2= 'stl-10-48',
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