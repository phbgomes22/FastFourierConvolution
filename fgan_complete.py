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
    def __init__(self, z_size):
        super(FGenerator, self).__init__()
        self.z_size = z_size
        self.model = torch.nn.Sequential(
            FFC_BN_ACT(z_size, 512, 4, 0.0, 0.5, stride=1, padding=0, activation_layer=nn.GELU, 
                      upsampling=True, uses_noise=True, uses_sn=True), 
            FFC_BN_ACT(512, 256, 4, 0.5, 0.5, stride=2, padding=1, activation_layer=nn.GELU, 
                      upsampling=True, uses_noise=True, uses_sn=True), 
            FFC_BN_ACT(256, 128, 4, 0.5, 0.5, stride=2, padding=1, activation_layer=nn.GELU, 
                      upsampling=True, uses_noise=True, uses_sn=True), 
            FFC_BN_ACT(128, 64, 4, 0.5, 0.5, stride=2, padding=1, activation_layer=nn.GELU, 
                      upsampling=True, uses_noise=True, uses_sn=True), 
            FFC_BN_ACT(64, 3, 4, 0.5, 0.0, stride=2, padding=1, activation_layer=nn.Tanh, 
                       norm_layer=nn.Identity, upsampling=True, uses_noise=True, uses_sn=False), 
        )

    def forward(self, z):
        fake = self.model(z.view(-1, self.z_size, 1, 1))
        fake = self.resizer(fake)
        self.print_size(fake)
        if not self.training:
            fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
            fake = fake.to(torch.uint8)
        return fake


class FDiscriminator(FFCModel):
    # Adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    def __init__(self, sn=True):
        super(FDiscriminator, self).__init__()
        sn_fn = torch.nn.utils.spectral_norm if sn else lambda x: x
        self.main = torch.nn.Sequential(
            FFC_BN_ACT(in_channels=3, out_channels=64, kernel_size=4,
                ratio_gin=0.0, ratio_gout=0.5, stride=2, padding=1, bias=False, 
                uses_noise=False, uses_sn=True, activation_layer=nn.GELU),
            FFC_BN_ACT(in_channels=64, out_channels=128, kernel_size=4,
                ratio_gin=0.5, ratio_gout=0.5, stride=2, padding=1, bias=False, 
                uses_noise=False, uses_sn=True, activation_layer=nn.GELU),
            FFC_BN_ACT(in_channels=128, out_channels=256, kernel_size=4,
                ratio_gin=0.5, ratio_gout=0.5, stride=2, padding=1, bias=False, 
                uses_noise=False, uses_sn=True, activation_layer=nn.GELU),
            FFC_BN_ACT(in_channels=256, out_channels=512, kernel_size=4,
                ratio_gin=0.5, ratio_gout=0.5, stride=2, padding=1, bias=False, 
                uses_noise=False, uses_sn=True, activation_layer=nn.GELU),
            FFC_BN_ACT(in_channels=512, out_channels=1, kernel_size=4,
                ratio_gin=0.5, ratio_gout=0, stride=1, padding=0, bias=False, 
                uses_noise=False, uses_sn=True, norm_layer=nn.Identity, 
                activation_layer=nn.Sigmoid)
        )
        # self.fc = sn_fn(torch.nn.Linear(4 * 4 * 512, 1))
        # self.act = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        debug_print("Começando Discriminador...")
        self.print_size(x)
        m = self.main(x)
        m = self.resizer(m)
        self.print_size(m)
        debug_print(m.size())
        return m.squeeze(-1).squeeze(-1)

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
            torchvision.transforms.Resize(64),
            torchvision.transforms.CenterCrop(64),
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    ds_instance = torchvision.datasets.CIFAR10(args.dir_dataset, train=True, download=True, transform=ds_transform)
    loader = torch.utils.data.DataLoader(
        ds_instance, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=4, pin_memory=True
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
    G.apply(weights_init)

    D = FDiscriminator(sn=True).to(device).train()
    D.apply(weights_init)

    # initialize persistent noise for observed samples
    z_vis = torch.randn(64, args.z_size, device=device)
    
    # prepare optimizer and learning rate schedulers (linear decay)
    optim_G = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    optim_D = torch.optim.AdamW(D.parameters(), lr=args.lr, betas=(0.0, 0.9))
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
        if next_step % (args.num_epoch_steps/2) != 0:
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
    parser.add_argument('--num_dis_updates', type=int, default=5)
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