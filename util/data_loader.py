'''
Authors: Pedro Gomes 
'''

import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10, CelebA, MNIST, Omniglot, Food101, StanfordCars, SVHN, Flowers102, FashionMNIST
from config import Config, Datasets

from .tar_loader import TarImageFolder


def get_device():
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and Config.shared().ngpu > 0) else "cpu")
    return device 


def load_data():
    ''''
    Based on the configuration setup, this function loads the dataset for the training.
    For this implementation, the training set will always have 64x64 images. The available datasets are defined
    in the `Datasets` file in the module config. 
    The function returns a dataloader in order to iterate over the training samples.
    '''
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
    ]

    if Datasets.is_grayscale(config.dataset_name):
        list_transforms.append(
            transforms.Normalize((0.5,), (0.5,))
        )
    else:
        list_transforms.append(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
        if config.nc == 1:
            list_transforms.append( transforms.Grayscale() )


    transform = transforms.Compose(list_transforms)

    dataset = None 
    if config.dataset_name == Datasets.CIFAR10.value:
        # - FOR CIFAR10
        print("Loading CIFAR10 dataset... ")
        dataset = CIFAR10(root='../cifar10_data', train=True, download=True, transform=transform)
    elif config.dataset_name == Datasets.CELEBA.value:
        # - For CelebA
        print("Loading CelebA dataset... ")
        ## - Trouble loading CelebA from dir?
        ## - https://stackoverflow.com/questions/69755609/dataset-not-found-or-corrupted-you-can-use-download-true-to-download-it
        dataset = CelebA(root='../celeba_data', split='train', download=True, transform=transform)
    elif config.dataset_name == Datasets.MNIST.value:
        # - For MNIST 
        print("Loading MNIST dataset... ")
        dataset = MNIST(root='../mnist_data', train=True, download=True, transform=transform)
    elif config.dataset_name == Datasets.FMNIST.value:
        # - For MNIST 
        print("Loading FashionMNIST dataset... ")
        dataset = FashionMNIST(root='../fashion_mnist_data', train=True, download=True, transform=transform)
    elif config.dataset_name == Datasets.OMNIGLOT.value:
        # - For Omniglot 
        print("Loading OMNIGLOT dataset... ")
        dataset = Omniglot(root='../omniglot_data', download=True, transform=transform)
    elif config.dataset_name == Datasets.FOOD101.value:
        # - For Food101
        print("Loading Food101 dataset... ")
        dataset = Food101(root='../food101_data', download=True, transform=transform)
    elif config.dataset_name == Datasets.CARS.value:
        # - For StanfordCars
        print("Loading StanfordCars dataset... ")
        dataset = StanfordCars(root='../stanfordcars_data', split='train', download=True, transform=transform)
    elif config.dataset_name == Datasets.SVHN.value:
        # - For SVHN
        print("Loading SVHN dataset... ")
        dataset = SVHN(root='../svhn_data', split='train', download=True, transform=transform)
    elif config.dataset_name == Datasets.FLOWERS.value:
        # - For SVHN
        print("Loading Flowers102 dataset... ")
        dataset = Flowers102(root='../flowers102_data', split='train', download=True, transform=transform)
    elif config.dataset_name == Datasets.LOCAL_DATASET.value:
        # - For local images
        print("Loading local dataset... ")
        dataset = dset.ImageFolder(root=config.dataroot, transform=transform)
    elif config.dataset_name == Datasets.LOCAL_TAR.value:
        # - For local tar images
        print("Loading local Tar dataset... ")
        dataset = TarImageFolder(config.dataroot, transform=transform)
    else:
        print("[Error] No dataset selected in data_loader!")
        raise ValueError('[Error] No dataset selected in data_loader!')


    print("Will create dataloader...")

    print("Dataloader created.")

    # device = get_device()

    # model_output = config.model_output
    # # Plot some training images
    # try:
    #     real_batch = next(iter(dataloader))
    #     plt.figure(figsize=(8,8))
    #     plt.axis("off")
    #     plt.title("Training Images")
    #     plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    #     plt.savefig(model_output + "training_set.jpg")
    # except OSError:
    #     print("Cannot load image")

    return dataset, batch_size, workers