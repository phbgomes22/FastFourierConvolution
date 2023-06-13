import torch
from config import Config
from util import *
from models import *
from PIL import Image 
import os
import argparse

device = get_device()


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
            nn.Linear(z_size, (self.mg * self.mg) * self.ngf*8),
      #      nn.BatchNorm1d((self.mg * self.mg) * self.ngf*8)
        )

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
        
        self.conv5 = FFC_BN_ACT(self.ngf, 3, 3, ratio_g, 0.0, stride=1, padding=1, activation_layer=nn.Tanh, 
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
        fake = self.resizer(fake)

        if not self.training:
            min_val = float(fake.min())
            max_val = float(fake.max())
            fake = (255 * (fake.clamp(min_val, max_val) * 0.5 + 0.5))
            # fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
            fake = fake.to(torch.uint8)
        return fake


def main():
    ## Reads the parameters send from the user through the terminal call of test.py
    dir = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', type=str, default=os.path.join(dir, 'logs_fgan_cond'))
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--number_samples', type=int, default=1000)
    parser.add_argument('--dir_logs', type=str, default=os.path.join(dir, 'generated_output'))
    args = parser.parse_args()
    test(args)


def test(args):
    nz = 128
    mg = 4 if args.img_size == 32 else 6
   ## Loading generator
    netG = FGenerator(z_size=nz, mg=mg).to(device) 
    netG.restore_checkpoint(ckpt_file=args.checkpoint_file)

    netG.eval()
    count = 0

    noise = torch.randn(args.number_samples, nz, device=device)

    with torch.no_grad():
        fake = netG(noise).detach().cpu()#.numpy()

    for f in fake:
        generated_image = np.transpose(f, (1,2,0))
        # generated_image -= generated_image.min()
        # generated_image /= generated_image.max()
        im = Image.fromarray((generated_image.squeeze(axis=2).numpy() * 255).astype(np.uint8))
        im.save(os.path.join(args.dir_logs, 'image' + str(count) + ".jpg"))
        count+=1

main()