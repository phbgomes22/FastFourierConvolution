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



class FGenerator(FFCModel):
    # Adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    def __init__(self, z_size, mg: int = 4):
        super(FGenerator, self).__init__()
        self.z_size = z_size
        self.ngf = 64
        ratio_g = 0.25
        self.mg = mg

        sn_fn = torch.nn.utils.spectral_norm 
       # self.noise_to_feature = sn_fn(nn.Linear(z_size, (self.mg * self.mg) * self.ngf*8))
        self.noise_to_feature = nn.Sequential(
            sn_fn(nn.Linear(z_size, (self.mg * self.mg) * self.ngf*4))
      #      nn.BatchNorm1d((self.mg * self.mg) * self.ngf*8)
        )

        self.conv2 = FFC_BN_ACT(self.ngf*4, self.ngf*4, 4, 0.0, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=nn.BatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True)
        self.lcl_noise2 = NoiseInjection(int(self.ngf*4*(1-ratio_g)))
        self.glb_noise2 = NoiseInjection(int(self.ngf*4*(ratio_g)))
        
        self.conv3 = FFC_BN_ACT(self.ngf*4, self.ngf*2, 4, ratio_g, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=nn.BatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True)
        self.lcl_noise3 = NoiseInjection(int(self.ngf*2*(1-ratio_g)))
        self.glb_noise3 = NoiseInjection(int(self.ngf*2*(ratio_g)))
        
        self.conv4 = FFC_BN_ACT(self.ngf*2, self.ngf*2, 4, ratio_g, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=nn.BatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True)
        self.lcl_noise4 = NoiseInjection(int(self.ngf*(1-ratio_g)))
        self.glb_noise4 = NoiseInjection(int(self.ngf*(ratio_g)))

        self.conv5 = FFC_BN_ACT(self.ngf*2, self.ngf, 4, ratio_g, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=nn.BatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True)
        self.lcl_noise5 = NoiseInjection(int(self.ngf*(1-ratio_g)))
        self.glb_noise5 = NoiseInjection(int(self.ngf*(ratio_g)))


        self.conv6 = FFC_BN_ACT(self.ngf, self.ngf, 4, ratio_g, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=nn.BatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True)
        self.lcl_noise6 = NoiseInjection(int(self.ngf*(1-ratio_g)))
        self.glb_noise6 = NoiseInjection(int(self.ngf*(ratio_g)))
        
        self.conv7 = FFC_BN_ACT(self.ngf, 3, 3, ratio_g, 0.0, stride=1, padding=1, activation_layer=nn.Tanh, 
                       norm_layer=nn.Identity, upsampling=False, uses_noise=True, uses_sn=True)

    def forward(self, z):
        
        fake = self.noise_to_feature(z)
      
        fake = fake.reshape(fake.size(0), -1, self.mg, self.mg)

        fake = self.conv2(fake)
        if self.training:
            fake = self.lcl_noise2(fake[0]), self.glb_noise2(fake[1]) 
        
        fake = self.conv3(fake)
        if self.training:
            fake = self.lcl_noise3(fake[0]), self.glb_noise3(fake[1])
        
        fake = self.conv4(fake)
        if self.training:
            fake = self.lcl_noise4(fake[0]), self.glb_noise4(fake[1]) 

        fake = self.conv5(fake)
        if self.training:
            fake = self.lcl_noise5(fake[0]), self.glb_noise5(fake[1]) 

        fake = self.conv6(fake)
        if self.training:
            fake = self.lcl_noise6(fake[0]), self.glb_noise6(fake[1]) 

        fake = self.conv7(fake)
        fake = self.resizer(fake)

        if not self.training:
            min_val = float(fake.min())
            max_val = float(fake.max())
            fake = (255 * (fake.clamp(min_val, max_val) * 0.5 + 0.5))
            # fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
            fake = fake.to(torch.uint8)
        return fake

class DBlock(nn.Module):
    """
    Residual block for discriminator.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        downsample (bool): If True, downsamples the input feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.        
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 downsample=False,
                 spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        self.spectral_norm = spectral_norm

        # Build the layers
        self.c1 = nn.utils.spectral_norm(nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1, 1))
        self.c2 = nn.utils.spectral_norm(nn.Conv2d(self.hidden_channels, self.out_channels, 3, 1,
                            1))

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))

        # Shortcut layer
        if self.learnable_sc:
            if self.spectral_norm:
                self.c_sc = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
            else:
                self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self.c_sc(x)
            return F.avg_pool2d(x, 2) if self.downsample else x

        else:
            return x

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        return self._residual(x) + self._shortcut(x)


class DBlockOptimized(nn.Module):
    """
    Optimized residual block for discriminator. This is used as the first residual block,
    where there is a definite downsampling involved. Follows the official SNGAN reference implementation
    in chainer.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.        
    """
    def __init__(self, in_channels, out_channels, spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spectral_norm = spectral_norm

        # Build the layers
        self.c1 = nn.utils.spectral_norm(nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1))
        self.c2 = nn.utils.spectral_norm(nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1))
        self.c_sc = nn.utils.spectral_norm(nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0))

        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        return self.c_sc(F.avg_pool2d(x, 2))

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        return self._residual(x) + self._shortcut(x)


class SNGANDiscriminator128(FFCModel):
    r"""
    ResNet backbone discriminator for SNGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, ndf=1024, **kwargs):
        super().__init__()
        self.ndf = ndf

        # Build layers
        self.block1 = DBlockOptimized(3, self.ndf >> 4)
        self.block2 = DBlock(self.ndf >> 4, self.ndf >> 3, downsample=True)
        self.block3 = DBlock(self.ndf >> 3, self.ndf >> 2, downsample=True)
        self.block4 = DBlock(self.ndf >> 2, self.ndf >> 1, downsample=True)
        self.block5 = DBlock(self.ndf >> 1, self.ndf, downsample=True)
        self.block6 = DBlock(self.ndf, self.ndf, downsample=False)
        self.l7 = nn.utils.spectral_norm(nn.Linear(self.ndf, 1))
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l7.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)

        # Global sum pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)

        return output


class Discriminator(FFCModel):
    # Adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    def __init__(self, sn=True, mg: int = 4):
        super(Discriminator, self).__init__()
        self.mg = mg
        sn_fn = torch.nn.utils.spectral_norm if sn else lambda x: x
        self.conv1 = sn_fn(torch.nn.Conv2d(3, 64, 3, stride=1, padding=(1,1)))
        self.conv2 = sn_fn(torch.nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = sn_fn(torch.nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = sn_fn(torch.nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv5 = sn_fn(torch.nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = sn_fn(torch.nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = sn_fn(torch.nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))
        self.conv8 = sn_fn(torch.nn.Conv2d(512, 512, 4, stride=2, padding=(1,1)))
        self.conv9 = sn_fn(torch.nn.Conv2d(512, 512, 4, stride=2, padding=(1,1)))
        self.fc = sn_fn(torch.nn.Linear(self.mg * self.mg * 512, 1))
    #    self.print_layer = Print(debug=True)
        self.act = torch.nn.LeakyReLU(0.1)

       # self.attn1 = Self_Attn(512, 'relu')

    def forward(self, x):
        m = self.act(self.conv1(x))
        m = self.act(self.conv2(m))
        m = self.act(self.conv3(m))
        m = self.act(self.conv4(m))
        m = self.act(self.conv5(m))
        m = self.act(self.conv6(m))
        m = self.act(self.conv7(m))
        m = self.act(self.conv8(m))
        m = self.act(self.conv9(m))
        output = self.fc(m.view(-1, self.mg * self.mg * 512))
 
        return output


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
    image_size = 128


    mg = 4
    # input2_dataset = 'celeba-64'
    # register_dataset('celeba-64', image_size=image_size)
    loader = load_flowers(batch_size=args.batch_size, image_size=image_size) # , file_path=args.dataset_path

    loader_iter = iter(loader)

    # reinterpret command line inputs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 0 

    # create Generator and Discriminator models
    G = FGenerator(z_size=args.z_size, mg=mg).to(device).train()
    G.apply(weights_init)
    params = count_parameters(G)
    
    print("- Parameters on generator: ", params)
    
    D = SNGANDiscriminator128().to(device).train() 
    D.apply(weights_init)
    params = count_parameters(D)
    print("- Parameters on discriminator: ", params)

    # initialize persistent noise for observed samples
    z_vis = torch.randn(64, args.z_size, device=device)
    
    # prepare optimizer and learning rate schedulers (linear decay)
    optim_G = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_D = torch.optim.AdamW(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lambda step: 1. - step / args.num_total_steps)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optim_D, lambda step: 1. - step / args.num_total_steps)
    # initialize logging
    os.makedirs(args.dir_logs, exist_ok=True)

    ini_step = 0

    ### GET CHECKPOINTS
    if args.checkpoint:
        # Obtain custom or latest checkpoint files
        netG_ckpt_dir = os.path.join(args.dir_logs, 'checkpoints',
                                            'netG')
        
        netG_ckpt_file = get_latest_checkpoint(
            netG_ckpt_dir)  # can be None

        netD_ckpt_dir = os.path.join(args.dir_logs, 'checkpoints',
                                            'netD')
        netD_ckpt_file = get_latest_checkpoint(
            netD_ckpt_dir)

        ### RESTORE CHECKPOINTS

        if netD_ckpt_file and os.path.exists(netD_ckpt_file):
            print("INFO: Restoring checkpoint for D...")
            ini_step = D.restore_checkpoint(
                ckpt_file=netD_ckpt_file, optimizer=optim_D, scheduler=scheduler_D)

        if netG_ckpt_file and os.path.exists(netG_ckpt_file):
            print("INFO: Restoring checkpoint for G...")
            ini_step = G.restore_checkpoint(
                ckpt_file=netG_ckpt_file, optimizer=optim_G, scheduler=scheduler_G)

        print("INFO: Initial Step: ", ini_step)

    tb = tensorboard.SummaryWriter(log_dir=args.dir_logs)
    pbar = tqdm.tqdm(total=args.num_total_steps, initial=ini_step,  desc='Training', unit='batch')
        

    for step in range(ini_step, args.num_total_steps):
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
            loss_D = hinge_loss_dis(output_dg, output_dreal)
            loss_D.backward()
            optim_D.step()

        # log
        if (step + 1) % 10 == 0:
            step_info = {'loss_G': loss_G.cpu().item(), 'loss_D': loss_D.cpu().item(), 'lr': scheduler_G.get_last_lr()[0]}
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


        # log observed images
        samples_vis = G(z_vis).detach().cpu()
        samples_vis = torchvision.utils.make_grid(samples_vis).permute(1, 2, 0).numpy()
        tb.add_image('observations', samples_vis, global_step=next_step, dataformats='HWC')
        samples_vis = PIL.Image.fromarray(samples_vis)
        samples_vis.save(os.path.join(args.dir_logs, f'{next_step:06d}.png'))


        # resume training
        if next_step <= args.num_total_steps:
            pbar = tqdm.tqdm(total=args.num_total_steps, initial=next_step, desc='Training', unit='batch')
            G.train()

            if args.checkpoint:
                G.save_checkpoint(directory = netG_ckpt_dir,
                                        global_step = next_step,
                                        optimizer = optim_G,
                                        scheduler = scheduler_G)

                D.save_checkpoint(directory = netD_ckpt_dir,
                                        global_step = next_step,
                                        optimizer = optim_D,
                                        scheduler = scheduler_D)

    tb.close()
    print(f'Training finished; the model is saved as '
          f'{args.dir_logs}/generator.onnx and {args.dir_logs}/generator.pth')


def main():
    dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_total_steps', type=int, default=100000)
    parser.add_argument('--num_epoch_steps', type=int, default=5000)
    parser.add_argument('--num_dis_updates', type=int, default=1)
    parser.add_argument('--num_samples_for_metrics', type=int, default=10000)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10', 'stl10', 'flowers', 'celeba'))
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--z_size', type=int, default=128, choices=(128,))
    parser.add_argument('--z_type', type=str, default='normal')
    parser.add_argument('--leading_metric', type=str, default='ISC', choices=('ISC', 'FID', 'KID', 'PPL'))
    parser.add_argument('--disable_sn', default=False, action='store_true')
    parser.add_argument('--conditional', default=False, action='store_true')
    parser.add_argument('--dir_logs', type=str, default=os.path.join(dir, 'logs_fgan'))
    parser.add_argument('--checkpoint', default=False, action='store_true')
    parser.add_argument('--dataset_path', type=str, required=False)
    args = parser.parse_args()
    print('Configuration:\n' + ('\n'.join([f'{k:>25}: {v}' for k, v in args.__dict__.items()])))
  #  assert not args.conditional, 'Conditional mode not implemented'
    train(args)


main()