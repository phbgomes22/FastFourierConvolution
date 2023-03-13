import torch
from config import Config
from util import *
from models import *
from PIL import Image 
import os


config = Config.shared()
device = get_device()


def main():
    ## Reads the parameters send from the user through the terminal call of test.py
    config.read_test_params()

    test_cond()

def test_cond():
    ngf = config.ngf
    nz = config.nz
    nc = config.nc
    model_path = config.model_path
    number_samples = config.samples
    output_dir = config.sample_output
    num_classes = config.num_classes
    embed_size = config.gen_embed

   ## Loading generator
    netG = None
    if config.FFC_GENERATOR:
        print("Using FFC Generator...")
        netG = FFCCondGenerator(nz=nz, nc=nc, ngf=ngf, num_classes= num_classes, 
                                    embed_size=embed_size, uses_noise=True, training=False).to(device) 
    else:
        print("Using Vanilla Generator...")
        netG = CondCvGenerator(nz=nz, nc=nc, ngf=ngf, 
                        num_classes= num_classes, 
                        embed_size=embed_size).to(device)

    netG.load_state_dict(torch.load(model_path))

    count = 0

    ## create noise array
    noise = torch.randn(number_samples, nz, 1, 1, device=device)

    ## create label array
    num_per_class = number_samples // num_classes
    labels = torch.zeros(number_samples, dtype=torch.long)

    for i in range(num_classes):
        labels[i*num_per_class : (i+1)*num_per_class] = i
    labels = labels.to(device)

    ## generate the samples
    with torch.no_grad():
        fake = netG(noise, labels).detach().cpu()#.numpy()

    ## save the samples 
    for f in fake:
        generated_image = np.transpose(f, (1,2,0))
        generated_image -= generated_image.min()
        generated_image /= generated_image.max()
        im = Image.fromarray((generated_image.squeeze(axis=2).numpy() * 255).astype(np.uint8))
        im.save(output_dir + 'image' + str(count) + ".jpg")
        count+=1


main()