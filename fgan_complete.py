# https://github.com/toshas/torch-fidelity/blob/master/examples/sngan_cifar10.py

from models import *

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
        

class Generator(torch.nn.Module):
    # Adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    def __init__(self, z_size):
        super(Generator, self).__init__()
        self.z_size = z_size

     #   self.print_layer = Print(debug=True)
      
        self.model = torch.nn.Sequential(
          #  GaussianNoise(0.01), #
            torch.nn.ConvTranspose2d(z_size, 512, 4, stride=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
          #  GaussianNoise(0.01), #
            torch.nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1,1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
          #  GaussianNoise(0.01), #
            torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
           # GaussianNoise(0.01), #
            torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
         #   GaussianNoise(0.01), #
            torch.nn.ConvTranspose2d(64, 3, 3, stride=1, padding=(1,1)),
            torch.nn.Tanh()
        )

    def forward(self, z):
        fake = self.model(z.view(-1, self.z_size, 1, 1))
        if not self.training:
            fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
            fake = fake.to(torch.uint8)

        return fake

class FGenerator(FFCModel):
    # Adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    def __init__(self, z_size):
        super(FGenerator, self).__init__()
        self.z_size = z_size
        self.ngf = 64
        ratio_g = 0.5
        self.model = torch.nn.Sequential(
            FFC_BN_ACT(z_size, self.ngf*8, 4, 0.0, ratio_g, stride=1, padding=0, activation_layer=nn.GELU, 
                      norm_layer=nn.BatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True), 
            FFC_BN_ACT(self.ngf*8, self.ngf*4, 4, ratio_g, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=nn.BatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True), 
            FFC_BN_ACT(self.ngf*4, self.ngf*2, 4, ratio_g, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=nn.BatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True), 
            FFC_BN_ACT(self.ngf*2, self.ngf, 4, ratio_g, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=nn.BatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True), 
            FFC_BN_ACT(self.ngf, 3, 3, ratio_g, 0.0, stride=1, padding=1, activation_layer=nn.Tanh, 
                       norm_layer=nn.Identity, upsampling=True, uses_noise=True, uses_sn=True), 
        )
      #  self.print_layer = Print(debug=True)

    def forward(self, z):
        fake = self.model(z.view(-1, self.z_size, 1, 1))
        fake = self.resizer(fake)
        self.print_size(fake)
        if not self.training:
            fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
            fake = fake.to(torch.uint8)
        return fake

class Discriminator(torch.nn.Module):
    # Adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    def __init__(self, sn=True):
        super(Discriminator, self).__init__()
        sn_fn = torch.nn.utils.spectral_norm if sn else lambda x: x
        self.conv1 = sn_fn(torch.nn.Conv2d(3, 64, 3, stride=1, padding=(1,1)))
        self.conv2 = sn_fn(torch.nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = sn_fn(torch.nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = sn_fn(torch.nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv5 = sn_fn(torch.nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = sn_fn(torch.nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = sn_fn(torch.nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))
        self.fc = sn_fn(torch.nn.Linear(4 * 4 * 512, 1))
    #    self.print_layer = Print(debug=True)
        self.act = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        m = self.act(self.conv1(x))
        m = self.act(self.conv2(m))
        m = self.act(self.conv3(m))
        m = self.act(self.conv4(m))
        m = self.act(self.conv5(m))
        m = self.act(self.conv6(m))
        m = self.act(self.conv7(m))
        output = self.fc(m.view(-1, 4 * 4 * 512))
 
        return output
    
class DCGANDiscrimnator(nn.Module):
    def __init__(self, sn: bool):
        super(DCGANDiscrimnator, self).__init__()
        sn_fn = torch.nn.utils.spectral_norm
        self.conv1 = sn_fn(torch.nn.Conv2d(3, 32, 3, stride=1, padding=(1,1)))
        self.conv2 = sn_fn(torch.nn.Conv2d(32, 32, 4, stride=2, padding=(1,1)))
        self.conv3 = sn_fn(torch.nn.Conv2d(32, 64, 3, stride=1, padding=(1,1)))
        self.conv4 = sn_fn(torch.nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv5 = sn_fn(torch.nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv6 = sn_fn(torch.nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv7 = sn_fn(torch.nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))

        self.act = torch.nn.LeakyReLU(0.1)
        self.fc = sn_fn(torch.nn.Linear(4 * 4 * 256, 1))

    def forward(self, x):
        m = self.act(self.conv1(x))
        m = self.act(self.conv2(m))
        m = self.act(self.conv3(m))
        m = self.act(self.conv4(m))
        m = self.act(self.conv5(m))
        m = self.act(self.conv6(m))
        m = self.act(self.conv7(m))
        return self.fc(m.view(-1, 4 * 4 * 256))

class FDiscriminator(FFCModel):
    # Adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    def __init__(self, sn=True):
        super(FDiscriminator, self).__init__()
        sn_fn = torch.nn.utils.spectral_norm if sn else lambda x: x
        # 3, 4, 3, 4, 3, 4, 3
        self.main = torch.nn.Sequential(
            FFC_BN_ACT(in_channels=3, out_channels=32, kernel_size=3,
                ratio_gin=0.0, ratio_gout=0.0, stride=1, padding=1, bias=True, 
                uses_noise=False, uses_sn=True, activation_layer=nn.LeakyReLU),
            FFC_BN_ACT(in_channels=32, out_channels=64, kernel_size=4,
                ratio_gin=0, ratio_gout=0.0, stride=2, padding=1, bias=True, 
                uses_noise=False, uses_sn=True, activation_layer=nn.LeakyReLU),
            FFC_BN_ACT(in_channels=64, out_channels=128, kernel_size=4,
                ratio_gin=0, ratio_gout=0.0, stride=2, padding=1, bias=True, 
                uses_noise=False, uses_sn=True, activation_layer=nn.LeakyReLU),
            FFC_BN_ACT(in_channels=128, out_channels=256, kernel_size=4,
                ratio_gin=0, ratio_gout=0.0, stride=2, padding=1, bias=True, 
                uses_noise=False, uses_sn=True, activation_layer=nn.LeakyReLU),
            # FFC_BN_ACT(in_channels=256, out_channels=1, kernel_size=4,
            #     ratio_gin=0, ratio_gout=0, stride=1, padding=0, bias=False, 
            #     uses_noise=False, uses_sn=True, norm_layer=nn.Identity, 
            #     activation_layer=nn.Sigmoid)
        )

        self.fc = sn_fn(torch.nn.Linear(4 * 4 * 256, 1))
      #  self.print_size = Print(debug=True)
        self.gaus_noise = GaussianNoise(0.05)
        # self.act = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
      #  x = self.gaus_noise(x)
        self.print_size(x)
        m = self.main(x)
        m = self.resizer(m)
       # self.print_size(m)
       # m = m.view(-1, 1)
      #  self.print_size(m)
       
        return self.fc(m.view(-1, 4 * 4 * 256))

class LargeFDiscriminator(FFCModel):
    # Adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    def __init__(self, sn=True):
        super(LargeFDiscriminator, self).__init__()
        sn_fn = torch.nn.utils.spectral_norm if sn else lambda x: x
        # 3, 4, 3, 4, 3, 4, 3
        ratio_g = 0.5 #0.5
        act_func = nn.LeakyReLU
        ndf = 64 # 32
        self.ndf = ndf
        self.main = torch.nn.Sequential(
            FFC_BN_ACT(in_channels=3, out_channels=ndf, kernel_size=3,
                ratio_gin=0.0, ratio_gout=0, stride=1, padding=1, bias=True, 
                uses_noise=False, uses_sn=True, activation_layer=act_func),
            FFC_BN_ACT(in_channels=ndf, out_channels=ndf, kernel_size=4,
                ratio_gin=0, ratio_gout=0, stride=2, padding=1, bias=True, 
                uses_noise=False, uses_sn=True, activation_layer=act_func),
            FFC_BN_ACT(in_channels=ndf, out_channels=ndf*2, kernel_size=3,
                ratio_gin=0, ratio_gout=ratio_g, stride=1, padding=1, bias=True, 
                uses_noise=False, uses_sn=True, activation_layer=act_func),
            FFC_BN_ACT(in_channels=ndf*2, out_channels=ndf*2, kernel_size=4,
                ratio_gin=ratio_g, ratio_gout=ratio_g, stride=2, padding=1, bias=True, 
                uses_noise=False, uses_sn=True, activation_layer=act_func),
            FFC_BN_ACT(in_channels=ndf*2, out_channels=ndf*2, kernel_size=4,
                ratio_gin=ratio_g, ratio_gout=ratio_g, stride=2, padding=1, bias=True, 
                uses_noise=False, uses_sn=True, activation_layer=act_func),
            FFC_BN_ACT(in_channels=ndf*2, out_channels=ndf*4, kernel_size=3,
                ratio_gin=ratio_g, ratio_gout=0, stride=1, padding=1, bias=True, 
                uses_noise=False, uses_sn=True, activation_layer=act_func),
            FFC_BN_ACT(in_channels=ndf*4, out_channels=ndf*4, kernel_size=4,
                ratio_gin=0, ratio_gout=0, stride=2, padding=1, bias=True, 
                uses_noise=False, uses_sn=True, activation_layer=act_func),
            FFC_BN_ACT(in_channels=ndf*4, out_channels=ndf*8, kernel_size=3,
                ratio_gin=0, ratio_gout=0, stride=1, padding=1, bias=True, 
                uses_noise=False, uses_sn=True, activation_layer=act_func),
        )

        self.fc = sn_fn(torch.nn.Linear(4 * 4 * ndf*8, 1))

      #  self.gaus_noise = GaussianNoise(0.01)

    def forward(self, x):
        debug_print("Começando Discriminador...")
      #  x = self.gaus_noise(x)
        # self.print_size(x)
        m = self.main(x)
        m = self.resizer(m)
        # self.print_size(m)
        # debug_print(m.size())
        return self.fc(m.view(-1, 4 * 4 * self.ndf * 8))

def hinge_loss_dis(fake, real):
   # fake = fake.squeeze(-1).squeeze(-1)
  #  real = real.squeeze(-1).squeeze(-1)
    assert fake.dim() == 2 and fake.shape[1] == 1 and real.shape == fake.shape, f'{fake.shape} {real.shape}'
    loss = torch.nn.functional.relu(1.0 - real).mean() + \
           torch.nn.functional.relu(1.0 + fake).mean()
    return loss


def hinge_loss_gen(fake):
   # fake = fake.squeeze(-1).squeeze(-1)
    assert fake.dim() == 2 and fake.shape[1] == 1
    loss = -fake.mean()
    return loss


def train(args):
    # set up dataset loader
    os.makedirs(args.dir_dataset, exist_ok=True)
    ds_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
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
    G = FGenerator(z_size=args.z_size).to(device).train()
   # G.apply(weights_init)
    params = count_parameters(G)
    print(G)
    
    print("- Parameters on generator: ", params)

    D = Discriminator(sn=True).to(device).train() #LargeF
 #   D.apply(weights_init)
    params = count_parameters(D)
    print("- Parameters on discriminator: ", params)
    print(D)

    # initialize persistent noise for observed samples
    z_vis = torch.randn(64, args.z_size, device=device)
    
    # prepare optimizer and learning rate schedulers (linear decay)
    optim_G = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_D = torch.optim.AdamW(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
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
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--z_size', type=int, default=128, choices=(128,))
    parser.add_argument('--z_type', type=str, default='normal')
    parser.add_argument('--leading_metric', type=str, default='ISC', choices=('ISC', 'FID', 'KID', 'PPL'))
    parser.add_argument('--disable_sn', default=False, action='store_true')
    parser.add_argument('--conditional', default=False, action='store_true')
    parser.add_argument('--dir_dataset', type=str, default=os.path.join(dir, 'dataset'))
    parser.add_argument('--dir_logs', type=str, default=os.path.join(dir, 'logs_fgan'))
    args = parser.parse_args()
    print('Configuration:\n' + ('\n'.join([f'{k:>25}: {v}' for k, v in args.__dict__.items()])))
  #  assert not args.conditional, 'Conditional mode not implemented'
    train(args)


main()