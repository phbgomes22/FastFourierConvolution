import argparse
import sys
from .datasets import Datasets



def read_cond_params(args = sys.argv[1:]):
    '''
    Responsible for setting the possible parameters for `train_cond.py` and returning the parsed values
    '''
    parser = argparse.ArgumentParser(description="The parsing commands list.")

    parser.add_argument("-c", "--nu_classes",
                    help = "Number of classes for conditional tarining",
                    type=int,
                    required=True)

    parser.add_argument("-e", "--epochs",
                    help="Number of iterations for the training",
                    type=int,
                    default=400)


    cond_opts = parser.parse_args(args)

    return cond_opts
    


def read_test_options(args = sys.argv[1:]):
    '''
    Responsible for setting the possible parameters for `test.py` and returning the parsed values
    '''
    parser = argparse.ArgumentParser(description="The parsing commands list.")

    parser.add_argument("--model_path",
                        help="The path to the pre-trained model weights",
                        required=True)

    parser.add_argument("-o", "--output",
                        help="The path for the output files",
                        type=str,
                        default="../generated_samples/")

    parser.add_argument("-n", "--number",
                        help = "Number of samples that the generator will create",
                        type=int,
                        default=1000)

    parser.add_argument("-g", "--generator",
                         help="Choose the type of generator you used when training the model", 
                         choices=["ffc", "vanilla"],
                         default='ffc')

    opts = parser.parse_args(args)

    return opts




def read_train_options(args = sys.argv[1:]):
    '''
    Responsible for setting the possible parameters for `train.py` and returning the parsed values
    '''
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
                        action='store_true')

    parser.add_argument("--color",
                        help="Choose the color scheme for the images to be generating",
                        choices=["greyscale", "colorized"],
                        default="colorized")

    parser.add_argument("-o", "--output",
                        help="The path to the output to store the trained models",
                        type=str,
                        default="../output/")

    parser.add_argument("-e", "--epochs",
                    help="Number of iterations for the training",
                    type=int,
                    default=400)

    parser.add_argument("-b", "--batch_size",
                help="Size of the batch size for the training",
                type=int,
                default=128)

    opts = parser.parse_args(args)



    return opts