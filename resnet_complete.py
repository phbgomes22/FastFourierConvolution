from models import *

import argparse
import os

import PIL
import torch
import torchvision
import tqdm

from torch.utils import tensorboard

import torch_fidelity


channels = 3
GEN_SIZE=256
DISC_SIZE=128


# class ResBlockGenerator(nn.Module):

#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResBlockGenerator, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
#         nn.init.xavier_uniform(self.conv1.weight.data, 1.)
#         nn.init.xavier_uniform(self.conv2.weight.data, 1.)

#         self.model = nn.Sequential(
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2),
#             self.conv1,
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             self.conv2
#             )
#         self.bypass = nn.Sequential()
#         if stride != 1:
#             self.bypass = nn.Upsample(scale_factor=2)

#     def forward(self, x):
#         return self.model(x) + self.bypass(x)


# class ResBlockDiscriminator(nn.Module):

#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResBlockDiscriminator, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
#         nn.init.xavier_uniform(self.conv1.weight.data, 1.)
#         nn.init.xavier_uniform(self.conv2.weight.data, 1.)

#         if stride == 1:
#             self.model = nn.Sequential(
#                 nn.ReLU(),
#                 spectral_norm(self.conv1),
#                 nn.ReLU(),
#                 spectral_norm(self.conv2)
#                 )
#         else:
#             self.model = nn.Sequential(
#                 nn.ReLU(),
#                 spectral_norm(self.conv1),
#                 nn.ReLU(),
#                 spectral_norm(self.conv2),
#                 nn.AvgPool2d(2, stride=stride, padding=0)
#                 )
#         self.bypass = nn.Sequential()
#         if stride != 1:

#             self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
#             nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

#             self.bypass = nn.Sequential(
#                 spectral_norm(self.bypass_conv),
#                 nn.AvgPool2d(2, stride=stride, padding=0)
#             )
#             # if in_channels == out_channels:
#             #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
#             # else:
#             #     self.bypass = nn.Sequential(
#             #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
#             #         nn.AvgPool2d(2, stride=stride, padding=0)
#             #     )


#     def forward(self, x):
#         return self.model(x) + self.bypass(x)


# # special ResBlock just for the first layer of the discriminator
# class FirstResBlockDiscriminator(nn.Module):

#     def __init__(self, in_channels, out_channels, stride=1):
#         super(FirstResBlockDiscriminator, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
#         self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
#         nn.init.xavier_uniform(self.conv1.weight.data, 1.)
#         nn.init.xavier_uniform(self.conv2.weight.data, 1.)
#         nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

#         # we don't want to apply ReLU activation to raw image before convolution transformation.
#         self.model = nn.Sequential(
#             spectral_norm(self.conv1),
#             nn.ReLU(),
#             spectral_norm(self.conv2),
#             nn.AvgPool2d(2)
#             )
#         self.bypass = nn.Sequential(
#             nn.AvgPool2d(2),
#             spectral_norm(self.bypass_conv),
#         )

#     def forward(self, x):
#         return self.model(x) + self.bypass(x)


# class Generator(nn.Module):
#     def __init__(self, z_size):
#         super(Generator, self).__init__()
#         self.z_dim = z_size

#         self.dense = nn.Linear(self.z_dim, 4 * 4 * GEN_SIZE)
#         self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
#         nn.init.xavier_uniform(self.dense.weight.data, 1.)
#         nn.init.xavier_uniform(self.final.weight.data, 1.)

#         self.model = nn.Sequential(
#             ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
#             ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
#             ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2),
#             nn.BatchNorm2d(GEN_SIZE),
#             nn.ReLU(),
#             self.final,
#             nn.Tanh())

#     def forward(self, z):
#         fake = self.model(self.dense(z).view(-1, GEN_SIZE, 4, 4))

#         if not self.training:
#             fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
#             fake = fake.to(torch.uint8)
#         return fake


# class Discriminator(nn.Module):
#     def __init__(self, sn: bool):
#         super(Discriminator, self).__init__()

#         self.model = nn.Sequential(
#                 FirstResBlockDiscriminator(channels, DISC_SIZE, stride=2),
#                 ResBlockDiscriminator(DISC_SIZE, DISC_SIZE, stride=2),
#                 ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
#                 ResBlockDiscriminator(DISC_SIZE, DISC_SIZE),
#                 nn.ReLU(),
#                 nn.AvgPool2d(8),
#             )
#         self.fc = nn.Linear(DISC_SIZE, 1)
#         nn.init.xavier_uniform(self.fc.weight.data, 1.)
#         self.fc = spectral_norm(self.fc)

#     def forward(self, x):
#         return self.fc(self.model(x).view(-1,DISC_SIZE))

from torch.nn import Parameter
# https://github.com/ajbrock/BigGAN-PyTorch/blob/master/TFHub/biggan_v1.py

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

		u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
		v = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
		u.data = l2normalize(u.data)
		v.data = l2normalize(v.data)
		w_bar = Parameter(w.data)

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
        x = self.block4(self.block3(self.block2(self.block1(inputs))))
        x = F.relu(x)
        features = torch.sum(x, dim=(2,3)) # global sum pooling
        x = self.dense(features)
        if self.sn_embedding is not None:
            x = self.sn_embedding(features, x, y)
        return x



def hinge_loss_dis(fake, real):
    assert fake.dim() == 2 and fake.shape[1] == 1 and real.shape == fake.shape, f'{fake.shape} {real.shape}'
    loss = torch.nn.functional.relu(1.0 - real).mean() + \
           torch.nn.functional.relu(1.0 + fake).mean()
    return loss


def hinge_loss_gen(fake):
    assert fake.dim() == 2 and fake.shape[1] == 1
    loss = -fake.mean()
    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args):
    # set up dataset loader
    os.makedirs(args.dir_dataset, exist_ok=True)
    ds_transform = torchvision.transforms.Compose(
        [
          #  transforms.Resize(image_size),
          #  transforms.CenterCrop(image_size),
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
   # ds_instance = torchvision.datasets.STL10(args.dir_dataset, split="train", download=True, transform=ds_transform)
    ds_instance = torchvision.datasets.CIFAR10(args.dir_dataset, train=True, download=True, transform=ds_transform)
    loader = torch.utils.data.DataLoader(
        ds_instance, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=8, pin_memory=True
    )
    loader_iter = iter(loader)

    # reinterpret command line inputs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 10 if args.conditional else 0  # unconditional
    leading_metric, last_best_metric, metric_greater_cmp = {
        'ISC': (torch_fidelity.KEY_METRIC_ISC_MEAN, 0.0, float.__gt__),
        'FID': (torch_fidelity.KEY_METRIC_FID, float('inf'), float.__lt__),
        'KID': (torch_fidelity.KEY_METRIC_KID_MEAN, float('inf'), float.__lt__),
        'PPL': (torch_fidelity.KEY_METRIC_PPL_MEAN, float('inf'), float.__lt__),
    }[args.leading_metric]

    # create Generator and Discriminator models
    G = Generator(enable_conditional=False).to(device).train()
    #Generator(z_size=args.z_size).to(device).train()
   # G.apply(weights_init)
    params = count_parameters(G)
    print(G)
    
    print("- Parameters on generator: ", params)

    D = Discriminator(enable_conditional=False).to(device).train()
    #Discriminator(sn=True).to(device).train() #LargeF
 #   D.apply(weights_init)
    params = count_parameters(D)
    print("- Parameters on discriminator: ", params)
    print(D)

    # initialize persistent noise for observed samples
    z_vis = torch.randn(64, args.z_size, device=device)
    
    # prepare optimizer and learning rate schedulers (linear decay)
    # optim_G = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # optim_D = torch.optim.AdamW(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lambda step: 1. - step / args.num_total_steps)
    # scheduler_D = torch.optim.lr_scheduler.LambdaLR(optim_D, lambda step: 1. - step / args.num_total_steps)

    # https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/main.py
    optim_D = torch.optim.AdamW(filter(lambda p: p.requires_grad, D.parameters()), lr=args.lr, betas=(0.0,0.9))
    optim_G  = torch.optim.AdamW(filter(lambda p: p.requires_grad, G.parameters()), lr=args.lr, betas=(0.0,0.9))

    # use an exponentially decaying learning rate
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optim_D, gamma=0.99)
    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optim_G, gamma=0.99)

    # initialize logging
    tb = tensorboard.SummaryWriter(log_dir=args.dir_logs)
    pbar = tqdm.tqdm(total=args.num_total_steps, desc='Training', unit='batch')
    os.makedirs(args.dir_logs, exist_ok=True)

    for step in range(args.num_total_steps):
        # read next batch
        try:
            real_img, real_label = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            real_img, real_label = next(loader_iter)
        real_img = real_img.to(device)
        real_label = real_label.to(device)

        # update Generator
        G.requires_grad_(True)
        D.requires_grad_(False)
        z = torch.randn(args.batch_size, args.z_size, device=device)

        optim_D.zero_grad()
        optim_G.zero_grad()
        fake = G(z)
        loss_G = hinge_loss_gen(D(fake))
        loss_G.backward()
        optim_G.step()

        # update Discriminator
        G.requires_grad_(False)
        D.requires_grad_(True)
        for d_iter in range(args.num_dis_updates):
            z = torch.randn(args.batch_size, args.z_size, device=device)
            optim_D.zero_grad()
            optim_G.zero_grad()
            fake = G(z)
            loss_D = hinge_loss_dis(D(fake), D(real_img))
            loss_D.backward()
            optim_D.step()

        # log
        if (step + 1) % 10 == 0:
            step_info = {'loss_G': loss_G.cpu().item(), 'loss_D': loss_D.cpu().item()}
            pbar.set_postfix(step_info)
            for k, v in step_info.items():
                tb.add_scalar(f'loss/{k}', v, global_step=step)
            tb.add_scalar(f'LR/lr', scheduler_G.get_last_lr()[0], global_step=step)
        pbar.update(1)

        # decay LR
        scheduler_G.step()
        scheduler_D.step()

        # check if it is validation time
        next_step = step + 1
        if next_step % (args.num_epoch_steps) != 0:
            continue
        pbar.close()
        G.eval()
        print('Evaluating the generator...')

        # compute and log generative metrics
        metrics = torch_fidelity.calculate_metrics(
            input1=torch_fidelity.GenerativeModelModuleWrapper(G, args.z_size, args.z_type, num_classes),
            input1_model_num_samples=args.num_samples_for_metrics,
            input2='cifar10-train',
            isc=True,
            fid=True,
            kid=True,
            ppl=False,
            ppl_epsilon=1e-2,
            ppl_sample_similarity_resize=64,
        )
        
        # log metrics
        for k, v in metrics.items():
            tb.add_scalar(f'metrics/{k}', v, global_step=next_step)

        # log observed images
        samples_vis = G(z_vis).detach().cpu()
        samples_vis = torchvision.utils.make_grid(samples_vis).permute(1, 2, 0).numpy()
        tb.add_image('observations', samples_vis, global_step=next_step, dataformats='HWC')
        samples_vis = PIL.Image.fromarray(samples_vis)
        samples_vis.save(os.path.join(args.dir_logs, f'{next_step:06d}.png'))

        # save the generator if it improved
        if metric_greater_cmp(metrics[leading_metric], last_best_metric):
            print(f'Leading metric {args.leading_metric} improved from {last_best_metric} to {metrics[leading_metric]}')

            last_best_metric = metrics[leading_metric]

        # resume training
        if next_step <= args.num_total_steps:
            pbar = tqdm.tqdm(total=args.num_total_steps, initial=next_step, desc='Training', unit='batch')
            G.train()

    tb.close()
    print(f'Training finished; the model with best {args.leading_metric} value ({last_best_metric}) is saved as '
          f'{args.dir_logs}/generator.onnx and {args.dir_logs}/generator.pth')


def main():
    dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_total_steps', type=int, default=100000)
    parser.add_argument('--num_epoch_steps', type=int, default=5000)
    parser.add_argument('--num_dis_updates', type=int, default=1)
    parser.add_argument('--num_samples_for_metrics', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-4) # was 2e-4
    parser.add_argument('--z_size', type=int, default=128, choices=(128,))
    parser.add_argument('--z_type', type=str, default='normal')
    parser.add_argument('--leading_metric', type=str, default='ISC', choices=('ISC', 'FID', 'KID', 'PPL'))
    parser.add_argument('--disable_sn', default=False, action='store_true')
    parser.add_argument('--conditional', default=False, action='store_true')
    parser.add_argument('--dir_dataset', type=str, default=os.path.join(dir, 'dataset_stl10'))
    parser.add_argument('--dir_logs', type=str, default=os.path.join(dir, 'logs_fgan_stl10'))
    args = parser.parse_args()
    print('Configuration:\n' + ('\n'.join([f'{k:>25}: {v}' for k, v in args.__dict__.items()])))
  #  assert not args.conditional, 'Conditional mode not implemented'
    train(args)


main()