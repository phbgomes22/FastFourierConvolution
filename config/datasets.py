'''
Authors: Pedro Gomes 
'''

from enum import Enum

class Datasets(Enum):
    MNIST = "MNIST"
    FMNIST = "FMNIST"
    CIFAR10 = "CIFAR10"
    CELEBA = "CelebA"
    OMNIGLOT = "OMNIGLOT"
    FOOD101 = "FOOD101"
    FLOWERS = "FLOWERS"
    CARS = "CARS"
    SVHN = "SVHN"
    LOCAL_DATASET = "LOCAL_DATASET"
    LOCAL_TAR = "TAR"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_ 

    @classmethod
    def is_grayscale(cls, value):
        return value in [Datasets.OMNIGLOT.value, Datasets.MNIST.value, Datasets.FMNIST.value]