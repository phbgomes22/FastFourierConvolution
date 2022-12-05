from .arg_parser import *
import os

class Config:
    '''
    Class responsible for storing the information on what configurations are the 
    test and training of the models running. 
    It has a set of default parameters, and they can be overrided by the user when running 
    the files with different arguments.
    '''
    _instance = None

    def __init__(self):
        self.some_attribute = None

    @classmethod
    def shared(cls):
        '''
        A shared instance of Config that is shared throughout the different
        models of the project.
        '''
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    
    ## --- train.py configuration parameters

    # Folder to store models
    model_output = '../output/'

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 128

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
    num_epochs = 400

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

    ## The path for the output files
    sample_output = "../generated_samples/"


    def check_and_fill_path(pth: str):
        '''
        Check if path ends with `/`. If not, adds it and return new value.
        Check if given output path already exists. Creates directory if not.
        '''
        new_pth = pth
        if not new_pth.endswith('/'):
            new_pth += '/'
        ## creates the output dir if it does not exist
        if not os.path.exists(new_pth):
            os.makedirs(new_pth)

        return new_pth

    def read_test_params(self):
        '''
        Uses the `arg_parser.py` functions to parse the configuration from the user
        and update the Config shared instance.
        '''
        opts = read_test_options()

        self.model_path = opts.model_path
        self.samples = opts.number
        self.sample_output = self.check_and_fill_path(opts.output)
        self.model_path 


    def read_train_params(self):
        '''
        Uses the `arg_parser.py` functions to parse the configuration from the user
        and update the Config shared instance.
        This function should be called prior to loading the training data or starting the training.
        '''
        opts = read_train_options()

        self.FFC_GENERATOR = True if opts.generator == 'ffc' else False

        self.FFC_DISCRIMINATOR = True if opts.discriminator == 'ffc' else False

        self.DEBUG = opts.debug

        self.model_output = self.check_and_fill_path(opts.output)

        assert Datasets.has_value(opts.dataset), "Dataset requested is not a valid dataset"
        self.dataset_name = opts.dataset

        # if the dataset chosen is local, then it must be provided the dataroot for the local dataset
        self.dataroot = opts.data_path

        self.nc = 1 if opts.color == 'greyscale' else 3

        self.batch_size = opts.batch_size

        self.num_epochs = opts.epochs

