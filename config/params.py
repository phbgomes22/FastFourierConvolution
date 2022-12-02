from .arg_parser import *
import os

class Config:
    _instance = None

    def __init__(self):
        self.some_attribute = None

    @classmethod
    def shared(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    
    ## --- train.py configuration parameters

    # Folder to store models
    model_output = '../output/'

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 512#128

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of channels in the training images. For color images this is 3
    #nc = 3
    nc = 1 # testing grayscale image

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = image_size

    # Size of feature maps in discriminator
    ndf = image_size

    # Number of training epochs
    num_epochs = 100

    # Learning rate for optimizers
    lr = 0.0005#0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # Controls the FFC layer ratio of local and global signals
    gout_factor = 0.5

    DEBUG = False

    FFC_GENERATOR = True

    FFC_DISCRIMINATOR = False

    dataset_name = Datasets.LOCAL_DATASET.value

    dataroot = '../data/'

    ## --- test.py configuration parameters

    ## The path for the weights of the pre-trained model
    model_path = ''

    ## Number of samples that the test function will create
    samples = 1000

    def read_test_params(self):

        opts = read_test_options()

        self.model_path = opts.model_path
        self.samples = opts.number

        self.model_path 


    def read_train_params(self):
        opts = read_train_options()

        self.FFC_GENERATOR = True if opts.generator == 'ffc' else False

        self.FFC_DISCRIMINATOR = True if opts.discriminator == 'ffc' else False

        self.DEBUG = opts.debug

        self.model_output = opts.output
        if not self.model_output.endswith('/'):
            self.model_output += '/'
        ## creates the output dir if it does not exist
        if not os.path.exists(self.model_output):
            os.makedirs(self.model_output)

        assert Datasets.has_value(opts.dataset), "Dataset requested is not a valid dataset"
        self.dataset_name = opts.dataset

        # if the dataset chosen is local, then it must be provided the dataroot for the local dataset
        self.dataroot = opts.data_path

        self.nc = 1 if opts.color == 'greyscale' else 3

