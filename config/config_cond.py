'''
Authors: Pedro Gomes 
'''

from .arg_parser import *
from .params import Config


class ConfigCond(Config): 

    _instance = None

    def __init__(self):
        super(ConfigCond, self).__init__()
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
        
    # number of classes in the dataset
    num_classes = 10

    # generator embedding - hyperparameter
    gen_embed = 100


    def read_cond_params(self):
        '''
        Uses the `arg_parser.py` functions to parse the configuration from the user
        and update the ConfigConnd shared instance.
        '''
        ## First, read the train params
        ## They are a subset of the parameters for the condinitional training
        opts = self.read_train_params()

        assert self.FFC_GENERATOR == True, "Only FFC-Generator available for conditional training"

        ## Then, read the extra params for the conditional training
        self.num_classes = opts.num_classes
