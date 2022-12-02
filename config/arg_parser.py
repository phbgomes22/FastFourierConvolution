import argparse
import sys
from .datasets import Datasets


def read_options(args = sys.argv[1:]):
    parser = argparse.ArgumentParser(description="The parsing commands list.")
    parser.add_argument("-g", "--generator",
                         help="Choose the type of generator you want the model to use", 
                         choices=["ffc", "vanilla"],
                         default='ffc')

    parser.add_argument("-d", "--discriminator",
                         help="Choose the type of discriminator you want the model to use", 
                         choices=["ffc", "vanilla"],
                         default='vanilla')

    parser.add_argument("--dataset",
                        help="Choose the dataset you will use - default is local dataset in folder ../data/",
                        choices=[Datasets.CIFAR10.value,
                                 Datasets.CELEBA.value, 
                                 Datasets.MNIST.value,
                                 Datasets.OMNIGLOT.value],
                        default=Datasets.LOCAL_DATASET.value)

    parser.add_argument("--data_path",
                        help="The path to the training data",
                        default="../data/")
                         
    parser.add_argument("--debug",
                        help="Choose if running with debug prints or not",
                        choices=[True, False],
                        type=bool,
                        default=False)

    parser.add_argument("--color",
                        help="Choose the color scheme for the images to be generating",
                        choices=["greyscale", "colorized"],
                        default="colorized")

    parser.add_argument("-o", "--output",
                        help="The path to the output to store the trained models",
                        default="../output/")

    opts = parser.parse_args(args)



    return opts