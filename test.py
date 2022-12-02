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

    test()

def test():
    ngf = config.ngf
    nz = config.nz
    nc = config.nc
    model_path = config.model_path

   ## Loading generator
    netG = None
    if config.FFC_GENERATOR:
        netG = FFCGenerator(nz, nc, ngf, debug=config.DEBUG).to(device) 
    else:
        netG = Generator(nz, nc, ngf).to(device)

    netG.load_state_dict(torch.load(model_path))

    count = 0

    noise = torch.randn(10000, nz, 1, 1, device=device)

    with torch.no_grad():
        fake = netG(noise).detach().cpu()#.numpy()

    output_dir = '../generated_samples/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for f in fake:
        generated_image = np.transpose(f, (1,2,0))
        generated_image -= generated_image.min()
        generated_image /= generated_image.max()
        im = Image.fromarray((generated_image.squeeze(axis=2).numpy() * 255).astype(np.uint8))
        im.save(output_dir + 'image/' + str(count) + ".jpg")
        count+=1


main()