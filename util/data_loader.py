import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
import scipy
from torchvision.datasets import CIFAR10, CelebA, MNIST, Omniglot
from config import Config, Datasets


def get_device():
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and Config.shared().ngpu > 0) else "cpu")
    return device 


def load_data():
    config = Config.shared()
    image_size = config.image_size
    batch_size = config.batch_size
    workers = config.workers
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset

    list_transforms = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    if config.nc == 1:
        list_transforms.append(transforms.Grayscale())

    transform = transforms.Compose(list_transforms)

    dataset = None 
    if config.dataset_name == Datasets.CIFAR10.value:
        # - FOR CIFAR10
        print("Loading CIFAR10 dataset... ")
        dataset = CIFAR10(root='../cifar10_data', train=True, download=True, transform=transform)
    elif config.dataset_name == Datasets.CELEBA.value:
        # - For CelebA
        print("Loading CelebA dataset... ")
        dataset = CelebA(root='../celeba_data', split='train', download=True, transform=transform)
    elif config.dataset_name == Datasets.MNIST.value:
        # - For MNIST 
        print("Loading MNIST dataset... ")
        dataset = MNIST(root='../mnist_data', train=True, download=True, transform=transform)
    elif config.dataset_name == Datasets.OMNIGLOT.value:
        # - For Omniglot 
        print("Loading OMNIGLOT dataset... ")
        dataset = Omniglot(root='../omniglot_data', download=True, transform=transform)
    elif config.dataset_name == Datasets.LOCAL_DATASET.value:
        # - FOR LOCAL IMAGES IN THE GOOGLE DRIVE
        dataset = dset.ImageFolder(root=config.dataroot, transform=transform)

    print("Will create dataloader...")
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    print("Dataloader created.")

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