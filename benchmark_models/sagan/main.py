
from parameter import *
from trainer import Trainer
# from tester import Tester
from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder
import torchvision
import torch

def main(config):
    # For fast training
    cudnn.benchmark = True

    ds_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(32, 32)),
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    ds_instance = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=ds_transform)
        
    loader = torch.utils.data.DataLoader(
        ds_instance, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, drop_last=True
    )
    # Data loader
    # data_loader = Data_Loader(config.train, config.dataset, config.image_path, config.imsize,
    #                          config.batch_size, shuf=config.train)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)


    if config.train:
        if config.model=='sagan':
            trainer = Trainer(loader, config)
        # elif config.model == 'qgan':
        #     trainer = qgan_trainer(data_loader.loader(), config)
        trainer.train()
    # else:
    #     tester = Tester(data_loader.loader(), config)
    #     tester.test()

if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)