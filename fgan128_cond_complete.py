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

class FCondGenerator(FFCModel):
    # Adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    def __init__(self, z_size, mg: int = 4, num_classes: int = 10):
        super(FCondGenerator, self).__init__()
        self.z_size = z_size
        self.ngf = 64
        ratio_g = 0.25

        self.mg = mg

        self.conv2 = FFC_BN_ACT(self.ngf*8, self.ngf*4, 4, 0.0, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=nn.BatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True)
        self.lcl_noise2 = NoiseInjection(int(self.ngf*4*(1-ratio_g)))
        self.glb_noise2 = NoiseInjection(int(self.ngf*4*(ratio_g)))
        
        self.conv3 = FFC_BN_ACT(self.ngf*4, self.ngf*2, 4, ratio_g, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=nn.BatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True)
        self.lcl_noise3 = NoiseInjection(int(self.ngf*2*(1-ratio_g)))
        self.glb_noise3 = NoiseInjection(int(self.ngf*2*(ratio_g)))
        
        self.conv4 = FFC_BN_ACT(self.ngf*2, self.ngf, 4, ratio_g, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=nn.BatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True)
        self.lcl_noise4 = NoiseInjection(int(self.ngf*(1-ratio_g)))
        self.glb_noise4 = NoiseInjection(int(self.ngf*(ratio_g)))

        self.conv5 = FFC_BN_ACT(self.ngf, self.ngf, 4, ratio_g, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=nn.BatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True)
        self.lcl_noise5 = NoiseInjection(int(self.ngf*(1-ratio_g)))
        self.glb_noise5 = NoiseInjection(int(self.ngf*(ratio_g)))


        self.conv6 = FFC_BN_ACT(self.ngf, self.ngf, 4, ratio_g, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=nn.BatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True)
        self.lcl_noise6 = NoiseInjection(int(self.ngf*(1-ratio_g)))
        self.glb_noise6 = NoiseInjection(int(self.ngf*(ratio_g)))
        
        self.conv7 = FFC_BN_ACT(self.ngf, 3, 3, ratio_g, 0.0, stride=1, padding=1, activation_layer=nn.Tanh, 
                       norm_layer=nn.Identity, upsampling=False, uses_noise=True, uses_sn=True)
        
        ## == Conditional

        self.label_conv = nn.Sequential(
            nn.ConvTranspose2d(num_classes, self.ngf*4, 4, 1, 0),
            nn.BatchNorm2d(self.ngf*4),
            nn.GELU()
        )

        self.input_conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_size, self.ngf*4, 4, 1, 0), # nn.Linear(z_size, (self.mg * self.mg) * self.ngf*4),#
            nn.BatchNorm2d( self.ngf*4), #(self.mg * self.mg) *
            nn.GELU()
        )

        self.label_embed = nn.Embedding(num_classes, num_classes)

    def forward(self, z, labels):
        ## conditional
        labels = torch.unsqueeze(labels, dim=-1)
        labels = torch.unsqueeze(labels, dim=-1)
        embedding = self.label_embed(labels)
        embedding = embedding.view(labels.shape[0], -1, 1, 1)
        embedding = self.label_conv(embedding)

        z = z.reshape(z.size(0), -1, 1, 1)
        input = self.input_conv(z)
       # input = fake.reshape(input.size(0), -1, self.mg, self.mg)

        input = torch.cat([input, embedding], dim=1)

        ## remainder
        fake = self.conv2(input)
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


class Discriminator(FFCModel):
    # Adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    def __init__(self, sn=True, mg: int = 4, num_classes=10):
        super(Discriminator, self).__init__()
        self.mg = mg
        sn_fn = torch.nn.utils.spectral_norm if sn else lambda x: x

        self.conv1 = sn_fn(torch.nn.Conv2d(3 + 1, 64, 3, stride=1, padding=(1,1)))
        self.conv2 = sn_fn(torch.nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = sn_fn(torch.nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = sn_fn(torch.nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv5 = sn_fn(torch.nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = sn_fn(torch.nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = sn_fn(torch.nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))
        self.conv8 = sn_fn(torch.nn.Conv2d(512, 512, 4, stride=2, padding=(1,1)))
        self.conv9 = sn_fn(torch.nn.Conv2d(512, 512, 4, stride=2, padding=(1,1)))
        self.fc = sn_fn(torch.nn.Linear(self.mg * self.mg * 512, 1))
        self.act = torch.nn.LeakyReLU(0.1)


        ## == Conditional
        self.label_embed = nn.Embedding(num_classes, 128*128)

    def forward(self, x, labels):
        labels = torch.unsqueeze(labels, dim=-1)
        labels = torch.unsqueeze(labels, dim=-1)
        embedding = self.label_embed(labels)
        embedding = embedding.view(labels.shape[0], 1, 128, 128)

        input = torch.cat([x, embedding], dim=1)
        
        m = self.act(self.conv1(input))
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
    assert fake.dim() == 2 and fake.shape[1] == 1 and real.shape == fake.shape, f'{fake.shape} {real.shape}'
    loss = torch.nn.functional.relu(1.0 - real).mean() + \
           torch.nn.functional.relu(1.0 + fake).mean()
    return loss


def hinge_loss_gen(fake):
    assert fake.dim() == 2 and fake.shape[1] == 1
    loss = -fake.mean()
    return loss

#criterion = MulticlassHingeLoss(num_classes=10)

def train(args):
    # set up dataset loader
    dir = os.getcwd()
    dir_dataset_name = 'dataset_' + str(args.dataset)
    dir_dataset = os.path.join(dir, dir_dataset_name)
    os.makedirs(dir_dataset, exist_ok=True)
    image_size = 32 if args.dataset == 'cifar10' or args.dataset == 'svhn' else 48
    ds_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(image_size, image_size)),
         #   torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    
    loader = load_textures(batch_size=args.batch_size, image_size=image_size, file_path=args.dataset_path)
    loader_iter = iter(loader)

    # reinterpret command line inputs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = args.num_classes if args.conditional else 0  # unconditional
    mg = 4

    # create Generator and Discriminator models
    G = FCondGenerator(z_size=args.z_size, mg=mg, num_classes=num_classes).to(device).train()
   # G.apply(weights_init)
    params = count_parameters(G)

    
    print("- Parameters on generator: ", params)

    D = Discriminator(sn=True, mg=mg, num_classes=num_classes).to(device).train() 
    D.apply(weights_init)
    params = count_parameters(D)
    print("- Parameters on discriminator: ", params)


    # initialize persistent noise for observed samples
    z_vis = torch.randn(64, args.z_size, device=device)
    z_label_vis = torch.nn.functional.one_hot( torch.as_tensor( np.repeat(range(num_classes), 8)[:64] ) ).float().to(device)


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

        print(real_img.shape)
        print(real_label.shape)

        # update Generator
        G.requires_grad_(True)
        D.requires_grad_(False)
        z = torch.randn(args.batch_size, args.z_size, device=device)

        optim_D.zero_grad()
        optim_G.zero_grad()
        fake = G(z, real_label)
        output = D(fake, real_label)
        ## - update hinge loss

        loss_G = hinge_loss_gen(output)
        loss_G.backward()
        optim_G.step()

        # update Discriminator
        G.requires_grad_(False)
        D.requires_grad_(True)
        for d_iter in range(args.num_dis_updates):
            z = torch.randn(args.batch_size, args.z_size, device=device)
            optim_D.zero_grad()
            optim_G.zero_grad()
            fake = G(z, real_label)
            output_dg = D(fake, real_label)
            output_dreal = D(real_img, real_label)
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
        samples_vis = G(z_vis, torch.argmax(z_label_vis, dim=1)).detach().cpu()
        samples_vis = torchvision.utils.make_grid(samples_vis).permute(1, 2, 0).numpy()
        tb.add_image('observations', samples_vis, global_step=next_step, dataformats='HWC')
        samples_vis = PIL.Image.fromarray(samples_vis)
        samples_vis.save(os.path.join(args.dir_logs, f'{next_step:06d}.png'))

        # resume training
        if next_step <= args.num_total_steps:
            pbar = tqdm.tqdm(total=args.num_total_steps, initial=next_step, desc='Training', unit='batch')
            G.train()

            if args.checkpoint and next_step > args.num_total_steps//2:
                G.save_checkpoint(directory = netG_ckpt_dir,
                                        global_step = next_step,
                                        optimizer = optim_G,
                                        scheduler = scheduler_G)

                D.save_checkpoint(directory = netD_ckpt_dir,
                                        global_step = next_step,
                                        optimizer = optim_D,
                                        scheduler = scheduler_D)

    tb.close()


def main():
    dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_total_steps', type=int, default=100000)
    parser.add_argument('--num_epoch_steps', type=int, default=5000)
    parser.add_argument('--num_dis_updates', type=int, default=1)
    parser.add_argument('--num_samples_for_metrics', type=int, default=10000)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10', 'stl10', 'svhn'))
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--z_size', type=int, default=128, choices=(128,))
    parser.add_argument('--z_type', type=str, default='normal')
    parser.add_argument('--leading_metric', type=str, default='ISC', choices=('ISC', 'FID', 'KID', 'PPL'))
    parser.add_argument('--disable_sn', default=False, action='store_true')
    parser.add_argument('--conditional', default=False, action='store_true')
    parser.add_argument('--num_classes', type=int, default=0)
    parser.add_argument('--dir_logs', type=str, default=os.path.join(dir, 'logs_fgan_cond'))
    parser.add_argument('--checkpoint', default=False, action='store_true')
    parser.add_argument('--dataset_path', type=str, required=False)
    args = parser.parse_args()
    print('Configuration:\n' + ('\n'.join([f'{k:>25}: {v}' for k, v in args.__dict__.items()])))
    
    train(args)


main()