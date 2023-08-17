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
from torchvision.datasets import CIFAR10, CelebA, MNIST, Omniglot, Food101, StanfordCars, SVHN, Flowers102, FashionMNIST, SUN397
from config import Config, Datasets

from .tar_loader import TarImageFolder, TarDataset
import torchvision.transforms.functional as F
import torch_fidelity
from natsort import natsorted
import PIL
import os


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

class SVHN_10(dset.SVHN):
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
    elif dataset == 'svhn-32':
        torch_fidelity.register_dataset('svhn-32', lambda root, download: SVHN_10(root, split='train', download=download, transform=transform_dts))
    else:
        torch_fidelity.register_dataset('cifar-10-32', lambda root, download: CIFAR_10(root, train=False, download=download, transform=transform_dts))


def load_cars(batch_size, image_size):

    ds_transform = transforms.Compose (
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    random_crop_transform = transforms.Compose (
        [
            transforms.RandomCrop(size=(image_size, image_size)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    hor_transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomHorizontalFlip(1.0),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    dataset = SUN397(root='../sun_data_ok', split='train', download=True, transform=ds_transform)
    dataset_hor = SUN397(root='../sun_data_hor', split='train', download=True, transform=hor_transform)
    dataset_crop = SUN397(root='../sun_data_crop', split='train', download=True, transform=random_crop_transform)

    train_cars_sets = torch.utils.data.ConcatDataset([dataset, dataset_hor, dataset_crop])

    dataloader = torch.utils.data.DataLoader(train_cars_sets, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    
    print("INFO: Loaded Flowers dataset with ", len(dataloader.dataset), " images!")
    print("INFO: Without Augmentation: ", len(dataset))

    ## Checking images
    try:
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to('cuda')[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.savefig("training_set.jpg")
    except OSError:
        print("Cannot load image")

    return dataloader

def load_flowers(batch_size, image_size):

    ds_transform = transforms.Compose (
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    random_crop_transform = transforms.Compose (

        [   transforms.Resize(size=(image_size*1.25, image_size*1.25)),
            transforms.RandomCrop(size=(image_size, image_size)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    
    aug_transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomHorizontalFlip(1.0),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    vert_transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomVerticalFlip(1.0),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    crop_transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomVerticalFlip(1.0),
            transforms.RandomHorizontalFlip(1.0),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    ds_instance = dset.Flowers102(root='../flowers102_data', split='train', download=True, transform=ds_transform)
    ds_instance_val = dset.Flowers102(root='../flowers102_data', split='val', download=True, transform=ds_transform)
    ds_instance_test = dset.Flowers102(root='../flowers102_data', split='test', download=True, transform=ds_transform)
    train_hz = dset.Flowers102(root='../flowers102_data', split='train', download=True, transform=aug_transform)
    val_hz = dset.Flowers102(root='../flowers102_data', split='val', download=True, transform=aug_transform)
    test_hz = dset.Flowers102(root='../flowers102_data', split='test', download=True, transform=aug_transform)
    train_vert = dset.Flowers102(root='../flowers102_data', split='train', download=True, transform=vert_transform)
    val_vert = dset.Flowers102(root='../flowers102_data', split='val', download=True, transform=vert_transform)
    test_vert = dset.Flowers102(root='../flowers102_data', split='test', download=True, transform=vert_transform)
    train_crop = dset.Flowers102(root='../flowers102_data', split='train', download=True, transform=crop_transform)
    val_crop = dset.Flowers102(root='../flowers102_data', split='val', download=True, transform=crop_transform)
    test_crop = dset.Flowers102(root='../flowers102_data', split='test', download=True, transform=crop_transform)
    train_rcrop = dset.Flowers102(root='../flowers102_data', split='train', download=True, transform=random_crop_transform)
    val_rcrop = dset.Flowers102(root='../flowers102_data', split='val', download=True, transform=random_crop_transform)
    test_rcrop = dset.Flowers102(root='../flowers102_data', split='test', download=True, transform=random_crop_transform)

    train_flowers_sets = torch.utils.data.ConcatDataset([ds_instance, ds_instance_val, ds_instance_test, 
                                                         train_hz, val_hz, test_hz,
                                                         train_vert, val_vert, test_vert,
                                                         train_crop, val_crop, test_crop,
                                                         train_rcrop, val_rcrop, test_rcrop])
                                                 ##        train_color, val_color, test_color])

    dataloader = torch.utils.data.DataLoader(train_flowers_sets, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    
    print("INFO: Loaded Flowers dataset with ", len(dataloader.dataset), " images!")
    print("INFO: Without Augmentation: ", len(ds_instance))

    ## Checking images
    try:
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to('cuda')[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.savefig("training_set.jpg")
    except OSError:
        print("Cannot load image")

    return dataloader


def load_textures(batch_size, image_size, file_path: str = '../celeba_data'):
    img_dir = file_path 

    transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    aug_transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomHorizontalFlip(1.0),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    print("Loading Texture dataset... ")
    texture_dataset = TarDataset(archive=img_dir, transform=transform)#(txt_path=txt_path, img_dir=img_dir, transform=transform)
    texture_dataset_aug = TarDataset(archive=img_dir, transform=aug_transform)
    texture_full = torch.utils.data.ConcatDataset([texture_dataset, texture_dataset_aug])
    print("INFO: Loaded Texture dataset with ", len(texture_full), " images!")
    
    dataloader = torch.utils.data.DataLoader(texture_full, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    
    try:
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to('cuda')[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.savefig("training_set.jpg")
        print("Image shape: ", real_batch[0].to('cuda')[0].shape)
    except OSError:
        print("Cannot load image")


    return dataloader


## Create a custom Dataset class
class CelebADataset(torch.utils.data.Dataset):
  #https://stackoverflow.com/questions/65528568/how-do-i-load-the-celeba-dataset-on-google-colab-using-torch-vision-without-ru
  #
  def __init__(self, root_dir, transform=None):
    """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
    # Read names of images in the root directory
    image_names = os.listdir(root_dir)

    self.root_dir = root_dir
    self.transform = transform 
    self.image_names = natsorted(image_names)

  def __len__(self): 
    return len(self.image_names)

  def __getitem__(self, idx):
    # Get the path to the image 
    img_path = os.path.join(self.root_dir, self.image_names[idx])
    # Load image and convert it to RGB
    img = PIL.Image.open(img_path).convert('RGB')
    # Apply transformations to the image
    if self.transform:
      img = self.transform(img)

    return img


def load_celeba(batch_size: int = 64, image_size:int = 48, file_path: str = '../celeba_data'):
    ## Load the dataset 
    img_dir = file_path 

    transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    aug_transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomHorizontalFlip(1.0),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # - For CelebA
    print("Loading CelebA dataset... ")
    ## - Trouble loading CelebA from dir?
    ## - https://stackoverflow.com/questions/69755609/dataset-not-found-or-corrupted-you-can-use-download-true-to-download-it
    #dataset = CelebA(root=file_path, split='all', download=True, transform=transform)
    
   # celeba_dataset = CelebADataset(img_folder, transform)
    celeba_dataset = TarDataset(archive=img_dir, transform=transform)#(txt_path=txt_path, img_dir=img_dir, transform=transform)
    celeba_dataset_aug = TarDataset(archive=img_dir, transform=aug_transform)
    celeba_full = torch.utils.data.ConcatDataset([celeba_dataset, celeba_dataset_aug])
    print("INFO: Loaded CelebA dataset with ", len(celeba_full), " images!")
    
    dataloader = torch.utils.data.DataLoader(celeba_full, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    
    try:
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to('cuda')[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.savefig("training_set.jpg")
        print("Image shape: ", real_batch[0].to('cuda')[0].shape)
    except OSError:
        print("Cannot load image")


    return dataloader


def load_cond_stl(batch_size, image_size):

    aug_transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomHorizontalFlip(1.0),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    random_crop = transforms.Compose(
        [
            transforms.RandomCrop(size=(60, 60)),
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    crop_transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    color_sharp_transform = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.5, hue=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomAutocontrast(),
            transforms.RandomEqualize(),
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    rand_rot_train = dset.STL10(root="./stl_data/data_stl__trainrandr", split="train", transform=random_crop, download=True)
    rand_rot_test = dset.STL10(root="./stl_data/data_stl_testr", split="test", transform=random_crop, download=True)

    hor_train = dset.STL10(root="./stl_data/data_stl_trainh", split="train", transform=aug_transform, download=True)
    hor_test = dset.STL10(root="./stl_data/data_stl_testh", split="test", transform=aug_transform, download=True)

    vert_train = dset.STL10(root="./stl_data/data_stl_trainv", split="train", transform=color_sharp_transform, download=True)
    vert_test = dset.STL10(root="./stl_data/data_stl_testv", split="test", transform=color_sharp_transform, download=True)

    resize_train = dset.STL10(root="./stl_data/data_stl_trains", split="train", transform=crop_transform, download=True)
    resize_test = dset.STL10(root="./stl_data/data_stl_tests", split="test", transform=crop_transform, download=True)

    stl_set = torch.utils.data.ConcatDataset([ rand_rot_train, rand_rot_test, hor_train, hor_test,
                                               vert_train, vert_test,
                                               resize_train, resize_test])
    
    dataloader = torch.utils.data.DataLoader(stl_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    print("INFO: Loaded STL-10 dataset with ", len(dataloader.dataset), " images!")
    print("INFO: Without Augmentation: ", len(hor_train))

    ## Checking images
    try:
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to('cuda')[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.savefig("training_set.jpg")
    except OSError:
        print("Cannot load image")

    return dataloader


def load_stl(batch_size, trans):
   
    # train + test (# 13000)
    dataset = dset.STL10(root="./data", split="train+unlabeled", transform=trans, download=True)
 
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