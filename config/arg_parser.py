
import argparse
import sys

def read_options(args=sys.argv[1:]):
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
                        help="Choose the dataset you will use",
                        choices=["Celeb", "CIFAR10", "MNIST"],
                        default='MNIST')
                         
    parser.add_argument("--debug",
                        help="Choose if running with debug prints or not",
                        choices=[True, False],
                        type=bool,
                        default=False)

    parser.add_argument("--color",
                        help="Choose the color scheme for the images to be generating",
                        choices=["greyscale", "colorized"],
                        default="colorized")

    opts = parser.parse_args(args)



    return opts