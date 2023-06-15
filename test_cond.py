import torch
from config import Config
from util import *
from models import *
from PIL import Image 
import os
import argparse


device = get_device()

class FCondGenerator(FFCModel):
    # Adapted from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    def __init__(self, z_size, mg: int = 4, num_classes: int = 10):
        super(FCondGenerator, self).__init__()
        self.z_size = z_size
        self.ngf = 64
        ratio_g = 0.25

        self.mg = mg

        self.conv2 = FFC_BN_ACT(self.ngf*8, self.ngf*4, 4, 0.0, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=ConditionalBatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True, num_classes=num_classes)
        self.lcl_noise2 = NoiseInjection(int(self.ngf*4*(1-ratio_g))) # only local receives noise
        self.glb_noise2 = NoiseInjection(int(self.ngf*4*(ratio_g)))

        self.conv3 = FFC_BN_ACT(self.ngf*4, self.ngf*2, 4, ratio_g, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=ConditionalBatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True, num_classes=num_classes)
        self.lcl_noise3 = NoiseInjection(int(self.ngf*2*(1-ratio_g))) # only local receives noise
        self.glb_noise3 = NoiseInjection(int(self.ngf*2*(ratio_g)))
        
        self.conv4 = FFC_BN_ACT(self.ngf*2, self.ngf, 4, ratio_g, ratio_g, stride=2, padding=1, activation_layer=nn.GELU, 
                      norm_layer=ConditionalBatchNorm2d, upsampling=True, uses_noise=True, uses_sn=True, num_classes=num_classes)
        self.lcl_noise4 = NoiseInjection(int(self.ngf*(1-ratio_g))) # only local receives noise
        self.glb_noise4 = NoiseInjection(int(self.ngf*(ratio_g)))
        
        self.conv5 = FFC_BN_ACT(self.ngf, 3, 3, ratio_g, 0.0, stride=1, padding=1, activation_layer=nn.Tanh, 
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
        fake = self.conv2(input, labels)
        if self.training:
            fake = self.lcl_noise2(fake[0]), self.glb_noise2(fake[1])
        
        fake = self.conv3(fake, labels)
        if self.training:
            fake = self.lcl_noise3(fake[0]), self.glb_noise3(fake[1])
        
        fake = self.conv4(fake, labels)
        if self.training:
            fake = self.lcl_noise4(fake[0]), self.glb_noise4(fake[1])

        fake = self.conv5(fake)
        fake = self.resizer(fake)

        if not self.training:
            fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
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
    parser.add_argument('--z_size', type=int, default=128, choices=(128,))
    args = parser.parse_args()
    os.makedirs(args.dir_logs, exist_ok=True)

    test_cond(args)

def test_cond(args):
    number_samples = args.number_samples
    output_dir = args.dir_logs
    mg = 4 if args.img_size == 32 else 6
    num_classes = 10

    nz = 128

   ## Loading generator
    netG = FCondGenerator(z_size=nz, mg=mg, num_classes=num_classes).to(device) 
    netG.restore_checkpoint(ckpt_file=args.checkpoint_file)
    netG.eval()

    count = 0

    ## create noise array
    noise = torch.randn(number_samples, nz, 1, 1, device=device)

    ## create label array
    num_per_class = number_samples // num_classes
    labels = torch.zeros(number_samples, dtype=torch.long)

    for i in range(num_classes):
        labels[i*num_per_class : (i+1)*num_per_class - 1] = i
    labels = labels.to(device)

    ## generate the samples
    with torch.no_grad():
        fake = netG(noise, labels).detach().cpu()#.numpy()

    ## save the samples 
    for f in fake:
        generated_image = np.transpose(f, (1,2,0))
        im = Image.fromarray(generated_image.cpu().numpy().astype(np.uint8)) #.squeeze(axis=2).numpy() * 255
        im.save(output_dir + '/' + 'image' + str(count) + ".jpg")
        count+=1


main()