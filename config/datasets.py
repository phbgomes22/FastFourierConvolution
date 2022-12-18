'''
Authors: Pedro Gomes 
'''

from enum import Enum

class Datasets(Enum):
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    CELEBA = "CelebA"
    OMNIGLOT = "OMNIGLOT"
    FOOD101 = "FOOD101"
    LOCAL_DATASET = "LOCAL_DATASET"


    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_ 

    @classmethod
    def is_grayscale(cls, value):
        return value in [Datasets.OMNIGLOT.value, Datasets.MNIST.value]