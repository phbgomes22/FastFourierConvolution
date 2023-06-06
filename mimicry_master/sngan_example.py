"""
Typical usage example.
"""

import torch
import torch.optim as optim
from torch_mimicry.nets import sngan
from torch_mimicry.training import Trainer
from torch_mimicry.datasets import load_dataset
from torch_mimicry import metrics

import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.transforms.functional as F
import torch_fidelity

class TransformPILtoRGBTensor:
    def __call__(self, img):
        return F.pil_to_tensor(img)

class STL_10(dset.STL10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img
    
    
def register_dataset(image_size):
    transform_dts = transforms.Compose(
        [
            transforms.Resize(image_size),
        #    transforms.CenterCrop(image_size),
            TransformPILtoRGBTensor()
        ]
    )

    torch_fidelity.register_dataset('stl-10-48', lambda root, download: STL_10(root, split='train+unlabeled', transform=transform_dts, download=download))
    

if __name__ == "__main__":
    # Data handling objects
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(root='./datasets', name='stl10_48')
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64,
                                             shuffle=True,
                                             num_workers=8)
    

    register_dataset(image_size=48)

    # Define models and optimizers
    netG = sngan.SNGANGenerator48().to(device)
    netD = sngan.SNGANDiscriminator48().to(device)
    optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
    optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

    # Start training
    trainer = Trainer(netD=netD,
                        netG=netG,
                        optD=optD,
                        optG=optG,
                        n_dis=5,
                        num_steps=100000,
                        lr_decay='linear',
                        print_steps=1000,
                        vis_steps=5000,
                        dataloader=dataloader,
                        log_dir='./log/example',
                        device=device)
    trainer.train()

    # # Evaluate fid
    # metrics.evaluate(metric='fid',
    #                      log_dir='./log/example',
    #                      netG=netG,
    #                      dataset='stl10_48',
    #                      num_real_samples=10000,
    #                      num_fake_samples=10000,
    #                      evaluate_step=30,
    #                      device=device)

    # # Evaluate kid
    # metrics.evaluate(metric='kid',
    #                      log_dir='./log/example',
    #                      netG=netG,
    #                      dataset='stl10_48',
    #                      num_samples=10000,
    #                      evaluate_step=30,
    #                      device=device)

    # # Evaluate inception score
    # metrics.evaluate(metric='inception_score',
    #                      log_dir='./log/example',
    #                      netG=netG,
    #                      num_samples=10000,
    #                      evaluate_step=30,
    #                      device=device)
