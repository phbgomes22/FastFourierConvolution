# https://github.com/toshas/torch-fidelity/blob/master/examples/sngan_cifar10.py

from models import *
from util import *

import argparse
import os

import PIL
import torch
import torchvision
import tqdm

from torch.utils import tensorboard

import torch_fidelity
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init(m):
    '''
    Custom weights initialization called on netG and netD
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


### Implementation 2 - Testing now   

def init_xavier_uniform(layer):
    if hasattr(layer, "weight"):
        torch.nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if hasattr(layer.bias, "data"):       
            layer.bias.data.fill_(0)

class GeneratorResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, n_classes=0):
        super().__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        # if n_classes == 0:
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # else:
        #     self.bn1 = ConditionalBatchNorm2d(in_ch, n_classes)
        #     self.bn2 = ConditionalBatchNorm2d(out_ch, n_classes)

        self.model = nn.Sequential(
            self.bn1,
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            self.bn2,
            nn.ReLU(),
            self.conv2
        )

        if self.stride > 1:
            self.bypass = nn.Upsample(scale_factor=2)#nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        else:
            self.bypass = nn.Identity()


    def forward(self, inputs, label_onehots=None):
        # residual add
        return self.model(inputs) + self.bypass(inputs)
        
## Discriminator Block
class ConvSNLRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding=0, lrelu_slope=0.1):
        super().__init__()
        sn_fn = torch.nn.utils.spectral_norm

        self.conv = sn_fn(nn.Conv2d(in_ch, out_ch, kernel, stride, padding=padding))
        self.lrelu = nn.LeakyReLU(lrelu_slope, True)
        
        self.conv.apply(init_xavier_uniform)

    def forward(self, inputs):
        return self.lrelu(self.conv(inputs))


class SNEmbedding(nn.Module):
    def __init__(self, n_classes, out_dims):
        super().__init__()
        sn_fn = torch.nn.utils.spectral_norm

        self.linear = sn_fn(nn.Linear(n_classes, out_dims, bias=False))

        self.linear.apply(init_xavier_uniform)

    def forward(self, base_features, output_logits, label_onehots):
        wy = self.linear(label_onehots)
        weighted = torch.sum(base_features * wy, dim=1, keepdim=True)
        return output_logits + weighted
    
    
class FirstDiscriminatorSNResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        sn_fn = torch.nn.utils.spectral_norm

        self.conv1 = sn_fn(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        self.conv2 = sn_fn(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        self.shortcut_conv = sn_fn(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0))

        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.shortcut_conv.weight.data, np.sqrt(2))

        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.AvgPool2d(2)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.shortcut_conv
        )

    def forward(self, inputs):
        return self.model(inputs) + self.bypass(inputs)


class DiscriminatorSNResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        sn_fn = torch.nn.utils.spectral_norm

        self.conv1 = sn_fn(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        self.conv2 = sn_fn(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.stride = stride
        if self.stride > 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.conv1,
                nn.ReLU(),
                self.conv2,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

            self.shortcut_conv = sn_fn(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0))
            nn.init.xavier_uniform_(self.shortcut_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                self.shortcut_conv,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.conv1,
                nn.ReLU(),
                self.conv2
            )
            self.bypass = nn.Identity()


    def forward(self, inputs):
        return self.model(inputs) + self.bypass(inputs)


class Generator(nn.Module):
    def __init__(self, z_dim, enable_conditional=False):
        super().__init__()
        n_classes = 10 if enable_conditional else 0
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * 256)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)

        self.block1 = GeneratorResidualBlock(256, 256, 2, n_classes=n_classes)
        self.block2 = GeneratorResidualBlock(256, 256, 2, n_classes=n_classes)
        self.block3 = GeneratorResidualBlock(256, 256, 2, n_classes=n_classes)
        self.bn_out = ConditionalBatchNorm2d(256, n_classes) if enable_conditional else nn.BatchNorm2d(256)

        self.final = nn.Conv2d(256, 3, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)
        
        self.out = nn.Sequential(
            nn.ReLU(True),
            self.final,
            nn.Tanh()
        )

    def forward(self, inputs, y=None):
        x = self.dense(inputs).view(inputs.size(0), -1, 4, 4)

        x = self.block3(self.block2(self.block1(x, y), y), y)
        x = self.bn_out(x, y) if y is not None else self.bn_out(x)
        fake = self.out(x)
        if not self.training:
            fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
            fake = fake.to(torch.uint8)

        return fake

class Discriminator(nn.Module):
    def __init__(self, enable_conditional=False):
        super().__init__()
        n_classes = 10 if enable_conditional else 0
        
        sn_fn = torch.nn.utils.spectral_norm


        self.model = nn.Sequential(
            FirstDiscriminatorSNResidualBlock(3, 128, 2),
            DiscriminatorSNResidualBlock(128, 128, 2),
            DiscriminatorSNResidualBlock(128, 128, 1),
            DiscriminatorSNResidualBlock(128, 128, 1),
            nn.ReLU(),
            nn.AvgPool2d(8)
        )


        self.dense = sn_fn(nn.Linear(128, 1))
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)

        # if n_classes > 0:
        #     self.sn_embedding = SNEmbedding(n_classes, 128)
        # else:
        #     self.sn_embedding = None

    def forward(self, inputs, y=None):
        x = self.model(inputs).view(-1, 128)
       # features = torch.sum(x, dim=(2,3)) # global sum pooling
        x = self.dense(x)
        # if self.sn_embedding is not None:
        #     x = self.sn_embedding(features, x, y)
        return x
 

'''
### Implementation 1 - Test - didn't work

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

        sn_fn = torch.nn.utils.spectral_norm

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                sn_fn(self.conv1),
                nn.ReLU(),
                sn_fn(self.conv2)
            )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                sn_fn(self.conv1),
                nn.ReLU(),
                sn_fn(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        self.bypass = nn.Sequential()

        if stride != 1:
            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                sn_fn(self.bypass_conv),
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

        sn_fn = torch.nn.utils.spectral_norm

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            sn_fn(self.conv1),
            nn.ReLU(),
            sn_fn(self.conv2),
            nn.AvgPool2d(2)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            sn_fn(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class GeneratorCIFAR10(nn.Module):
    def __init__(self, z_dim):
        super(GeneratorCIFAR10, self).__init__()
        self.ngf = 256
        self.z_dim = z_dim
        self.mg = 4
        self.dense = nn.Linear(self.z_dim, self.mg * self.mg * self.ngf)
        self.final = nn.Conv2d(self.ngf, 3, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResBlockGenerator(self.ngf, self.ngf, stride=2),
            ResBlockGenerator(self.ngf, self.ngf, stride=2),
            ResBlockGenerator(self.ngf, self.ngf, stride=2),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(),
            self.final,
            nn.Tanh()
        )

    def forward(self, z):
        input = self.dense(z)
        input = input.reshape(input.size(0), -1, self.mg, self.mg)
        fake = self.model(input)
        if not self.training:
            fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
            fake = fake.to(torch.uint8)

        return fake

class DiscriminatorCIFAR10(nn.Module):
    def __init__(self):
        super(DiscriminatorCIFAR10, self).__init__()

        self.ndf = 128
        sn_fn = torch.nn.utils.spectral_norm

        self.model = nn.Sequential(
                FirstResBlockDiscriminator(3, self.ndf, stride=2),
                ResBlockDiscriminator(self.ndf, self.ndf, stride=2),
                ResBlockDiscriminator(self.ndf, self.ndf),
                ResBlockDiscriminator(self.ndf, self.ndf),
                nn.ReLU(),
                nn.AvgPool2d(8),
            )
        
        self.fc = nn.Linear(self.ndf, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = sn_fn(self.fc)

    def forward(self, x):
        return self.fc(self.model(x).view(-1, self.ndf))

'''

def hinge_loss_dis(fake, real):
   # fake = fake.squeeze(-1).squeeze(-1)
  #  real = real.squeeze(-1).squeeze(-1)
    assert fake.dim() == 2 and fake.shape[1] == 1 and real.shape == fake.shape, f'{fake.shape} {real.shape}'
    loss = torch.nn.functional.relu(1.0 - real).mean() + \
           torch.nn.functional.relu(1.0 + fake).mean()
    return loss

def hinge_loss_real(real):
    loss = torch.nn.functional.relu(1.0 - real).mean()
    return loss

def hinge_loss_fake(fake):
    return torch.nn.functional.relu(1.0 + fake).mean()

def hinge_loss_gen(fake):
   # fake = fake.squeeze(-1).squeeze(-1)
    assert fake.dim() == 2 and fake.shape[1] == 1
    loss = -fake.mean()
    return loss


def train(args):
    # set up dataset loader
    dir = os.getcwd()
    dir_dataset_name = 'dataset_' + str(args.dataset)
    dir_dataset = os.path.join(dir, dir_dataset_name)
    os.makedirs(dir_dataset, exist_ok=True)
    image_size = 32 if args.dataset == 'cifar10' else 48
    ds_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(image_size, image_size)),
           # torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    if args.dataset == 'cifar10':
        ds_instance = torchvision.datasets.CIFAR10(dir_dataset, train=True, download=True, transform=ds_transform)
        mg = 4
        input2_dataset = args.dataset + '-train'
        loader = torch.utils.data.DataLoader(
            ds_instance, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
        )
    else:
       # ds_instance = torchvision.datasets.STL10(dir_dataset, split='train', download=True, transform=ds_transform)
        mg = 6
        register_dataset(image_size=image_size)
        input2_dataset = 'stl-10-48'
        loader = load_stl(args.batch_size, ds_transform)

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
    G = Generator(z_dim=args.z_size).to(device).train()
    G.apply(weights_init)
    params = count_parameters(G)
    print(G)
    
    print("- Parameters on generator: ", params)

    D = Discriminator().to(device).train() 
    D.apply(weights_init)
    params = count_parameters(D)
    print("- Parameters on discriminator: ", params)
    print(D)

    # initialize persistent noise for observed samples
    z_vis = torch.randn(64, args.z_size, device=device)
    
    # prepare optimizer and learning rate schedulers (linear decay)
    optim_G = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.5, 0.999)) #0.999
    optim_D = torch.optim.AdamW(D.parameters(), lr=args.lr, betas=(0.5, 0.999)) #0.999
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lambda step: 1. - step / args.num_total_steps)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optim_D, lambda step: 1. - step / args.num_total_steps)

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
            output_dg = D(fake)
            output_dreal = D(real_img)
            ## - hinge loss with criterion
            ## - update hinge loss

            hg_loss_real = hinge_loss_real(output_dreal)
            hg_loss_fake = hinge_loss_fake(output_dg)
            # testing Adaptative Weight Loss Method
            # loss_D = aw_method().aw_loss(Dloss_real= hg_loss_real, Dloss_fake= hg_loss_fake, Dis_opt=optim_D, 
                                # Dis_Net=D, real_validity=output_dreal, fake_validity=output_dg)
            loss_D = hinge_loss_dis(output_dg, output_dreal)
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
            input2= input2_dataset,
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
    parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10', 'stl10'))
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--z_size', type=int, default=128, choices=(128,))
    parser.add_argument('--z_type', type=str, default='normal')
    parser.add_argument('--leading_metric', type=str, default='ISC', choices=('ISC', 'FID', 'KID', 'PPL'))
    parser.add_argument('--disable_sn', default=False, action='store_true')
    parser.add_argument('--conditional', default=False, action='store_true')
    parser.add_argument('--dir_logs', type=str, default=os.path.join(dir, 'logs_fgan'))
    args = parser.parse_args()
    print('Configuration:\n' + ('\n'.join([f'{k:>25}: {v}' for k, v in args.__dict__.items()])))
  #  assert not args.conditional, 'Conditional mode not implemented'
    train(args)


main()