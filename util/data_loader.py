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
import torchvision.transforms.functional as F
import torch_fidelity


class TransformPILtoRGBTensor:
    def __call__(self, img):
        return F.pil_to_tensor(img)
    
class Flowers_102(dset.Flowers102):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img

class STL_10(dset.STL10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img
    
class CIFAR_10(dset.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img


def get_device():
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and Config.shared().ngpu > 0) else "cpu")
    return device 

def register_dataset(dataset, image_size):
    transform_dts = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
        #    transforms.CenterCrop(image_size),
            TransformPILtoRGBTensor()
        ]
    )
    if dataset == 'stl-10-48':
        torch_fidelity.register_dataset('stl-10-48', lambda root, download: STL_10(root, split='train+unlabeled', transform=transform_dts, download=download))
    elif dataset == 'flowers-48':
        torch_fidelity.register_dataset('flowers-48', lambda root, download: Flowers_102(root=root, split='train', download=download, transform=transform_dts))
    else:
        torch_fidelity.register_dataset('svhn-32', lambda root, download: SVHN(root, split='train', download=download, transform=transform_dts))
        torch_fidelity.register_dataset('cifar-10-32', lambda root, download: CIFAR_10(root, train=False, download=download, transform=transform_dts))


def load_flowers(batch_size, image_size):

    ds_transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
           # torchvision.transforms.CenterCrop(image_size),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    ds_instance = dset.Flowers102(root='../flowers102_data', split='train', download=True, transform=ds_transform)
    
    aug_transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomHorizontalFlip(1.0),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    aug_instance_hz = dset.Flowers102(root='../flowers102_data', split='train', download=True, transform=aug_transform)

    vert_transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomVerticalFlip(1.0),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    aug_instance_vert = dset.Flowers102(root='../flowers102_data', split='train', download=True, transform=vert_transform)


    crop_transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    aug_instance_crop = dset.Flowers102(root='../flowers102_data', split='train', download=True, transform=crop_transform)

    train_flowers_sets = torch.utils.data.ConcatDataset([ds_instance, aug_instance_hz, aug_instance_vert, aug_instance_crop])

    dataloader = torch.utils.data.DataLoader(train_flowers_sets, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    return dataloader


def load_stl(batch_size, trans):
   
    # train + test (# 13000)
    dataset = dset.STL10(root="./data", split="train+unlabeled", transform=trans, download=True)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    # imgs, labels = [], []
    # for x, y in dataloader:
    #     imgs.append(x)
    #     labels.append(y)
    # dataset = dset.STL10(root="./data", split="test", transform=trans)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    # for x, y in dataloader:
    #     imgs.append(x)
    #     labels.append(y)
    # # as tensor
    # all_imgs = torch.cat(imgs, dim=0)
    # all_labels = torch.cat(labels, dim=0)
    # # as dataset
    # dataset = torch.utils.data.TensorDataset(all_imgs, all_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    return dataloader

def load_data(color_channels: int = -1):
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

    color_channels = config.nc if color_channels == -1 else color_channels

    # Create the dataset

    list_transforms = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]

    if Datasets.is_grayscale(config.dataset_name):
        if color_channels == 3:
            print("Converting Grayscale to RGB...")
            list_transforms.append( transforms.Grayscale(num_output_channels=3) )

        list_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        if color_channels == 1:
            print("Converting RGB to Grayscale...")
            list_transforms.append( transforms.Grayscale() )

        list_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    #list_transforms.append(transforms.ToTensor())

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