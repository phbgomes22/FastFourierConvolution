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


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

parser.add_argument('--model', type=str, default='resnet')

args = parser.parse_args()

def l2normalize(v, eps=1e-4):
	return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
	def __init__(self, module, name='weight', power_iterations=1):
		super(SpectralNorm, self).__init__()
		self.module = module
		self.name = name
		self.power_iterations = power_iterations
		if not self._made_params():
			self._make_params()

	def _update_u_v(self):
		u = getattr(self.module, self.name + "_u")
		v = getattr(self.module, self.name + "_v")
		w = getattr(self.module, self.name + "_bar")

		height = w.data.shape[0]
		_w = w.view(height, -1)
		for _ in range(self.power_iterations):
			v = l2normalize(torch.matmul(_w.t(), u))
			u = l2normalize(torch.matmul(_w, v))

		sigma = u.dot((_w).mv(v))
		setattr(self.module, self.name, w / sigma.expand_as(w))

	def _made_params(self):
		try:
			getattr(self.module, self.name + "_u")
			getattr(self.module, self.name + "_v")
			getattr(self.module, self.name + "_bar")
			return True
		except AttributeError:
			return False

	def _make_params(self):
		w = getattr(self.module, self.name)

		height = w.data.shape[0]
		width = w.view(height, -1).data.shape[1]

		u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
		v = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
		u.data = l2normalize(u.data)
		v.data = l2normalize(v.data)
		w_bar = nn.Parameter(w.data)

		del self.module._parameters[self.name]
		self.module.register_parameter(self.name + "_u", u)
		self.module.register_parameter(self.name + "_v", v)
		self.module.register_parameter(self.name + "_bar", w_bar)

	def forward(self, *args):
		self._update_u_v()
		return self.module.forward(*args)

class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes, eps=1e-4, momentum=0.1):
    super().__init__()
    self.num_features = num_features
    self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum)
    self.gamma_embed = nn.Linear(num_classes, num_features, bias=False)
    self.beta_embed = nn.Linear(num_classes, num_features, bias=False)

  def forward(self, x, y):
    out = self.bn(x)
    gamma = self.gamma_embed(y) + 1
    beta = self.beta_embed(y)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out


def init_xavier_uniform(layer):
    if hasattr(layer, "weight"):
        torch.nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if hasattr(layer.bias, "data"):       
            layer.bias.data.fill_(0)

## Generator block

class DeconvBNRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding=0, n_classes=0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding=padding)
        if n_classes == 0:
            self.bn = nn.BatchNorm2d(out_ch)
        else:
            self.bn = ConditionalBatchNorm2d(out_ch, n_classes)
        self.relu = nn.ReLU(True)

        self.conv.apply(init_xavier_uniform)

    def forward(self, inputs, label_onehots=None):
        x = self.conv(inputs)
        if label_onehots is not None:
            x = self.bn(x, label_onehots)
        else:
            x = self.bn(x)
        return self.relu(x)

class GeneratorResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsampling, n_classes=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.upsampling = upsampling
        if n_classes == 0:
            self.bn1 = nn.BatchNorm2d(in_ch)
            self.bn2 = nn.BatchNorm2d(out_ch)
        else:
            print("HAS CLASSES")
            self.bn1 = ConditionalBatchNorm2d(in_ch, n_classes)
            self.bn2 = ConditionalBatchNorm2d(out_ch, n_classes)
        if in_ch != out_ch or upsampling > 1:
            self.shortcut_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        else:
            self.shortcut_conv = None

        self.conv1.apply(init_xavier_uniform)
        self.conv2.apply(init_xavier_uniform)

    def forward(self, inputs, label_onehots=None):
        # main
        if label_onehots is not None:
            print("Not None")
            x = self.bn1(inputs, label_onehots)
        else:
            x = self.bn1(inputs)
        x = F.relu(x)

        if self.upsampling > 1:
            x = F.interpolate(x, scale_factor=self.upsampling)
        x = self.conv1(x)

        if label_onehots is not None:
            x = self.bn2(x, label_onehots)
        else:
            x = self.bn2(x)
        x = F.relu(x)

        x = self.conv2(x)

        # short cut
        if self.upsampling > 1:
            shortcut = F.interpolate(inputs, scale_factor=self.upsampling)
        else:
            shortcut = inputs
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
        # residual add
        return x + shortcut
        
## Discriminator Block
class ConvSNLRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding=0, lrelu_slope=0.1):
        super().__init__()
        self.conv = SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel, stride, padding=padding))
        self.lrelu = nn.LeakyReLU(lrelu_slope, True)
        
        self.conv.apply(init_xavier_uniform)

    def forward(self, inputs):
        return self.lrelu(self.conv(inputs))

class SNEmbedding(nn.Module):
    def __init__(self, n_classes, out_dims):
        super().__init__()
        self.linear = SpectralNorm(nn.Linear(n_classes, out_dims, bias=False))

        self.linear.apply(init_xavier_uniform)

    def forward(self, base_features, output_logits, label_onehots):
        wy = self.linear(label_onehots)
        weighted = torch.sum(base_features * wy, dim=1, keepdim=True)
        return output_logits + weighted

class DiscriminatorSNResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsampling):
        super().__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        self.downsampling = downsampling
        if in_ch != out_ch or downsampling > 1:
            self.shortcut_conv = SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0))
        else:
            self.shortcut_conv = None

        self.conv1.apply(init_xavier_uniform)
        self.conv2.apply(init_xavier_uniform)

    def forward(self, inputs):
        x = F.relu(inputs)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        # short cut
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(inputs)
        else:
            shortcut = inputs
        if self.downsampling > 1:
            x = F.avg_pool2d(x, kernel_size=self.downsampling)
            shortcut = F.avg_pool2d(shortcut, kernel_size=self.downsampling)
        # residual add
        return x + shortcut 

class Generator(nn.Module):
    def __init__(self, enable_conditional=False):
        super().__init__()
        print(enable_conditional)
        n_classes = 10 if enable_conditional else 0
        self.dense = nn.Linear(128, 4 * 4 * 256)
        self.block1 = GeneratorResidualBlock(256, 256, 2, n_classes=n_classes)
        self.block2 = GeneratorResidualBlock(256, 256, 2, n_classes=n_classes)
        self.block3 = GeneratorResidualBlock(256, 256, 2, n_classes=n_classes)
        self.bn_out = ConditionalBatchNorm2d(256, n_classes) if enable_conditional else nn.BatchNorm2d(256)
        self.out = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, inputs, y=None):
        x = self.dense(inputs).view(inputs.size(0), 256, 4, 4)
        x = self.block3(self.block2(self.block1(x, y), y), y)
        x = self.bn_out(x, y) if y is not None else self.bn_out(x)

        x = self.out(x)
        if not self.training:
            x = (255 * (x.clamp(-1, 1) * 0.5 + 0.5))
            x = x.to(torch.uint8)

        return x
    

class Discriminator(nn.Module):
    def __init__(self, enable_conditional=False):
        super().__init__()
        n_classes = 10 if enable_conditional else 0
        self.block1 = DiscriminatorSNResidualBlock(3, 128, 2)
        self.block2 = DiscriminatorSNResidualBlock(128, 128, 2)
        self.block3 = DiscriminatorSNResidualBlock(128, 128, 1)
        self.block4 = DiscriminatorSNResidualBlock(128, 128, 1)
        self.dense = nn.Linear(128, 1)
        if n_classes > 0:
            self.sn_embedding = SNEmbedding(n_classes, 128)
        else:
            self.sn_embedding = None

    def forward(self, inputs, y=None):
        inputs = self.block1(inputs)
        inputs = self.block2(inputs)
        inputs = self.block3(inputs)
        x = self.block4(inputs)

        x = F.relu(x)
        features = torch.sum(x, dim=(2,3)) # global sum pooling
        x = self.dense(features)
        if self.sn_embedding is not None:
            x = self.sn_embedding(features, x, y)
        return x



dir = os.getcwd()
dataset_dir = os.path.join(dir, 'dataset')
logs_dir = os.path.join(dir, 'logs_resnet')

loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(dataset_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

Z_dim = 128
#number of updates to discriminator for every update to generator 
disc_iters = 5

# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training
# if args.model == 'resnet':
discriminator = Discriminator().cuda()
generator = Generator().cuda()
# else:
#     discriminator = model.Discriminator().cuda()
#     generator = model.Generator(Z_dim).cuda()

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
            # if args.loss == 'hinge':
            disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()
            # elif args.loss == 'wasserstein':
            #     disc_loss = -discriminator(data).mean() + discriminator(generator(z)).mean()
            # else:
            #     disc_loss = nn.BCEWithLogitsLoss()(discriminator(data), Variable(torch.ones(args.batch_size, 1).cuda())) + \
            #         nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.zeros(args.batch_size, 1).cuda()))
            disc_loss.backward()
            optim_disc.step()

        z = Variable(torch.randn(args.batch_size, Z_dim).cuda())

        # update generator
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        # if args.loss == 'hinge' or args.loss == 'wasserstein':
        gen_loss = -discriminator(generator(z)).mean()
        # else:
        #     gen_loss = nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.ones(args.batch_size, 1).cuda()))
        gen_loss.backward()
        optim_gen.step()

        if batch_idx % 100 == 0:
            print('disc loss', disc_loss.cpu().item(), 'gen loss', gen_loss.cpu().item())
    scheduler_d.step()
    scheduler_g.step()

fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())

def evaluate(epoch):
    print("evaluating...")
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

    img_path = os.path.join(logs_dir, '{}.png'.format(str(epoch).zfill(3)))
    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)

# os.makedirs(args.checkpoint_dir, exist_ok=True)

for epoch in range(2000):
    train(epoch)
    evaluate(epoch)
    # torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
    # torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))