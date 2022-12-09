'''
Authors: Pedro Gomes 
'''

from .arg_parser import *
from .params import Config
import os


class ConfigCond(Config): 
    def __init__(self):
        super(ConfigCond, self).__init__()
        
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
        self.read_train_params()

        assert self.FFC_GENERATOR == True, "Only FFC-Generator available for conditional training"

        ## Then, read the extra params for the conditional training
        opts = read_cond_params()

        self.num_classes = opts.num_classes
        self.gen_embed = opts.gen_embed
