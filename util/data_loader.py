import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10, CelebA
from config import *


def get_device():
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    return device 


def load_data():
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    transform = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                transforms.Grayscale(),
                            ])
    # - FOR CIFAR10
    #dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

    # - For CelebA
    dataset = CelebA(root='./celebsa_data', split='train', download=True, transform=transform)

    # - FOR LOCAL IMAGES IN THE GOOGLE DRIVE
    #dataset = dset.ImageFolder(root=dataroot, transform=transform)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    device = get_device()
    # Plot some training images
    try:
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    except OSError:
        print("Cannot load image")

    return dataloader