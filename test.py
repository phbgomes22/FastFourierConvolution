import torch
from config import Config
from util import *
from models import *
from PIL import Image 
import os
import argparse
import torchvision
from torchvision import models, transforms, utils

import torch.nn.functional as F

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
        
        feature_maps = []
        
        fake = self.noise_to_feature(z)
      
        fake = fake.reshape(fake.size(0), -1, self.mg, self.mg)
        feature_maps.append(fake)

        fake = self.conv2(fake)
        if self.training:
            fake = self.lcl_noise2(fake[0]), self.glb_noise2(fake[1]) 
        
        feature_maps.append(fake[0])
        feature_maps.append(fake[1])

        fake = self.conv3(fake)
        if self.training:
            fake = self.lcl_noise3(fake[0]), self.glb_noise3(fake[1])
        
        feature_maps.append(fake[0])
        feature_maps.append(fake[1])

        fake = self.conv4(fake)
        if self.training:
            fake = self.lcl_noise4(fake[0]), self.glb_noise4(fake[1]) 

        feature_maps.append(fake[0])
        feature_maps.append(fake[1])

        fake = self.conv5(fake)
        fake = self.resizer(fake)

        feature_maps.append(fake)

        if not self.training:
            min_val = float(fake.min())
            max_val = float(fake.max())
            fake = (255 * (fake.clamp(min_val, max_val) * 0.5 + 0.5))
            # fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
            fake = fake.to(torch.uint8)
        return fake, feature_maps

transform = transforms.Compose(
        [
            transforms.Resize(size=(48, 48)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

def main():
    ## Reads the parameters send from the user through the terminal call of test.py
    dir = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', type=str, default=os.path.join(dir, 'logs_fgan_cond'))
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--base_image', type=str, default='image.png')
    parser.add_argument('--number_samples', type=int, default=1000)
    parser.add_argument('--dir_logs', type=str, default=os.path.join(dir, 'generated_output'))
    parser.add_argument('--z_size', type=int, default=128, choices=(128,))
    args = parser.parse_args()
    os.makedirs(args.dir_logs, exist_ok=True)

    test(args)



def get_filters(args):
    ## 
    image = Image.open(args.base_image)
    plt.imshow(image)


    nz = 128
    mg = 4 if args.img_size == 32 else 6

    model = FGenerator(z_size=nz, mg=mg).to(device) 
    model.restore_checkpoint(ckpt_file=args.checkpoint_file)


    #we will save the 49 conv layers in this list
    conv_layers = [
        model.noise_to_feature,
        [model.conv2.ffc.convl2g, model.conv2.ffc.convg2l]
    ]

    # we will save the conv layer weights in this list
    model_weights =[]

    # get all the model children as list
    model_children = list(model.children())
    #counter to keep count of the conv layers
    counter = 0


    print(f"Total convolution layers: {counter}")
    print("conv_layers")

    model = model.to(device)

    image = transform(image)
    print(f"Image shape before: {image.shape}")
    image = image.unsqueeze(0)
    print(f"Image shape after: {image.shape}")
    image = image.to(device)

    z = torch.randn(1, args.z_size, device=device)
    
    img, outputs = model(z)

    save_image(img, args.logs, 1, 'example_image')

    #print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)

    
    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i+1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
     #   a.set_title(names[i].split('(')[0], fontsize=30)
    plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')


def save_image(fake, logs, num, name='image'):
    generated_image = np.transpose(fake, (1,2,0))
    # generated_image -= generated_image.min()
    # generated_image /= generated_image.max()
    im = Image.fromarray(generated_image.numpy().astype(np.uint8)) #.squeeze(axis=2).numpy() * 255).astype(np.uint8)
    im.save(os.path.join(logs, name + str(num) + ".png"))

def test(args):
   ## Loading generator

    get_filters(args)

    return
    # nz = 128
    # mg = 4 if args.img_size == 32 else 6

    # model = FGenerator(z_size=nz, mg=mg).to(device) 
    # model.restore_checkpoint(ckpt_file=args.checkpoint_file)
    # netG.eval()
    # count = 0

    # noise = torch.randn(args.number_samples, nz, device=device)

    # with torch.no_grad():
    #     fake = netG(noise).detach().cpu()#.numpy()

    # for f in fake:
    #     save_image(f, args.dir_logs, count)
    #     count+=1
    

main()