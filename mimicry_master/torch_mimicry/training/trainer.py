"""
Implementation of Trainer object for training GANs.
"""
import os
import re
import time

import torch

from torch_mimicry.training import scheduler, logger, metric_log
from torch_mimicry.utils import common


import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.transforms.functional as F
#import torch_fidelity

import torch_fidelity

class TransformPILtoRGBTensor:
    def __call__(self, img):
        return F.pil_to_tensor(img)

class STL_10(dset.STL10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img
    
    
def register_dataset(image_size):
    transform_dts = transforms.Compose(
        [
            transforms.Resize(image_size),
        #    transforms.CenterCrop(image_size),
            TransformPILtoRGBTensor()
        ]
    )

    torch_fidelity.register_dataset('stl-10-48', lambda root, download: STL_10(root, split='train', transform=transform_dts, download=download))
  

class Trainer:
    """
    Trainer object for constructing the GAN training pipeline.

    Attributes:
        netD (Module): Torch discriminator model.
        netG (Module): Torch generator model.
        optD (Optimizer): Torch optimizer object for discriminator.
        optG (Optimizer): Torch optimizer object for generator.
        dataloader (DataLoader): Torch object for loading data from a dataset object.
        num_steps (int): The number of training iterations.
        n_dis (int): Number of discriminator update steps per generator training step.
        lr_decay (str): The learning rate decay policy to use.
        log_dir (str): The path to storing logging information and checkpoints.
        device (Device): Torch device object to send model/data to.
        logger (Logger): Logger object for visualising training information.
        scheduler (LRScheduler): GAN training specific learning rate scheduler object.
        params (dict): Dictionary of training hyperparameters.
        netD_ckpt_file (str): Custom checkpoint file to restore discriminator from.
        netG_ckpt_file (str): Custom checkpoint file to restore generator from.
        print_steps (int): Number of training steps before printing training info to stdout.
        vis_steps (int): Number of training steps before visualising images with TensorBoard.
        flush_secs (int): Number of seconds before flushing summaries to disk.
        log_steps (int): Number of training steps before writing summaries to TensorBoard.
        save_steps (int): Number of training steps bfeore checkpointing.
    """
    def __init__(self,
                 netD,
                 netG,
                 optD,
                 optG,
                 dataloader,
                 num_steps,
                 log_dir='./log',
                 n_dis=1,
                 lr_decay=None,
                 device=None,
                 netG_ckpt_file=None,
                 netD_ckpt_file=None,
                 print_steps=1,
                 vis_steps=500,
                 log_steps=50,
                 save_steps=5000,
                 flush_secs=30):
        # Input values checks
        ints_to_check = {
            'num_steps': num_steps,
            'n_dis': n_dis,
            'print_steps': print_steps,
            'vis_steps': vis_steps,
            'log_steps': log_steps,
            'save_steps': save_steps,
            'flush_secs': flush_secs
        }
        for name, var in ints_to_check.items():
            if var < 1:
                raise ValueError('{} must be at least 1 but got {}.'.format(
                    name, var))

        self.netD = netD
        self.netG = netG
        self.optD = optD
        self.optG = optG
        self.n_dis = n_dis
        self.lr_decay = lr_decay
        self.dataloader = dataloader
        self.num_steps = num_steps
        self.device = device
        self.log_dir = log_dir
        self.netG_ckpt_file = netG_ckpt_file
        self.netD_ckpt_file = netD_ckpt_file
        self.print_steps = print_steps
        self.vis_steps = vis_steps
        self.log_steps = log_steps
        self.save_steps = save_steps

        register_dataset(image_size=48)  

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Training helper objects
        self.logger = logger.Logger(log_dir=self.log_dir,
                                    num_steps=self.num_steps,
                                    dataset_size=len(self.dataloader),
                                    flush_secs=flush_secs,
                                    device=self.device)

       #  self.scheduler = scheduler.LRScheduler(lr_decay=self.lr_decay,
        #                                        optD=self.optD,
        #                                        optG=self.optG,
        #                                        num_steps=self.num_steps)
        
        # self.scheduler_d = optim.lr_scheduler.ExponentialLR(self.optD, gamma=0.999)
        # self.scheduler_g = optim.lr_scheduler.ExponentialLR(self.optG, gamma=0.999)

        self.scheduler_d = torch.optim.lr_scheduler.LambdaLR(self.optD, lambda step: 1. - step / 2*self.num_steps)
        self.scheduler_g = torch.optim.lr_scheduler.LambdaLR(self.optG, lambda step: 1. - step / 2*self.num_steps)
     
        # Obtain custom or latest checkpoint files
        if self.netG_ckpt_file:
            self.netG_ckpt_dir = os.path.dirname(netG_ckpt_file)
            self.netG_ckpt_file = netG_ckpt_file
        else:
            self.netG_ckpt_dir = os.path.join(self.log_dir, 'checkpoints',
                                              'netG')
            self.netG_ckpt_file = self._get_latest_checkpoint(
               self.netG_ckpt_dir)  # can be None

        if self.netD_ckpt_file:
            self.netD_ckpt_dir = os.path.dirname(netD_ckpt_file)
            self.netD_ckpt_file = netD_ckpt_file
        else:
            self.netD_ckpt_dir = os.path.join(self.log_dir, 'checkpoints',
                                              'netD')
            self.netD_ckpt_file = self._get_latest_checkpoint(
               self.netD_ckpt_dir)

        # Log hyperparameters for experiments
        self.params = {
            'log_dir': self.log_dir,
            'num_steps': self.num_steps,
            'batch_size': self.dataloader.batch_size,
            'n_dis': self.n_dis,
            'lr_decay': self.lr_decay,
            'optD': optD.__repr__(),
            'optG': optG.__repr__(),
            'save_steps': self.save_steps,
        }
        self._log_params(self.params)

        num_params, num_trained_params = self.netG.count_params()
        print("Parameters on generator: ", num_params, " - trainable: ", num_trained_params)
        num_params, num_trained_params = self.netD.count_params()
        print("Parameters on discriminator: ", num_params, " - trainable: ", num_trained_params)

        # Device for hosting model and data
        if not self.device:
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else "cpu")

        # Ensure model and data are in the same device
        for net in [self.netD, self.netG]:
            if net.device != self.device:
                net.to(self.device)

    def _log_params(self, params):
        """
        Takes the argument options to save into a json file.
        """
        params_file = os.path.join(self.log_dir, 'params.json')

        # Check for discrepancy with previous training config.
        if os.path.exists(params_file):
            check = common.load_from_json(params_file)

            if params != check:
                diffs = []
                for k in params:
                    if k in check and params[k] != check[k]:
                        diffs.append('{}: Expected {} but got {}.'.format(
                            k, check[k], params[k]))

                diff_string = '\n'.join(diffs)
                raise ValueError(
                    "Current hyperparameter configuration is different from previously:\n{}"
                    .format(diff_string))

        common.write_to_json(params, params_file)

    def _get_latest_checkpoint(self, ckpt_dir):
        """
        Given a checkpoint dir, finds the checkpoint with the latest training step.
        """
        def _get_step_number(k):
            """
            Helper function to get step number from checkpoint files.
            """
            search = re.search(r'(\d+)_steps', k)

            if search:
                return int(search.groups()[0])
            else:
                return -float('inf')

        if not os.path.exists(ckpt_dir):
            return None

        files = os.listdir(ckpt_dir)
        if len(files) == 0:
            return None

        ckpt_file = max(files, key=lambda x: _get_step_number(x))

        return os.path.join(ckpt_dir, ckpt_file)

    def _fetch_data(self, iter_dataloader):
        """
        Fetches the next set of data and refresh the iterator when it is exhausted.
        Follows python EAFP, so no iterator.hasNext() is used.
        """
        try:
            real_batch = next(iter_dataloader)
        except StopIteration:
            iter_dataloader = iter(self.dataloader)
            real_batch = next(iter_dataloader)

        real_batch = (real_batch[0].to(self.device),
                      real_batch[1].to(self.device))

        return iter_dataloader, real_batch

    def _restore_models_and_step(self):
        """
        Restores model and optimizer checkpoints and ensures global step is in sync.
        """
        global_step_D = global_step_G = 0

        # if self.netD_ckpt_file and os.path.exists(self.netD_ckpt_file):
        #     print("INFO: Restoring checkpoint for D...")
        #     global_step_D = self.netD.restore_checkpoint(
        #         ckpt_file=self.netD_ckpt_file, optimizer=self.optD)

        # if self.netG_ckpt_file and os.path.exists(self.netG_ckpt_file):
        #     print("INFO: Restoring checkpoint for G...")
        #     global_step_G = self.netG.restore_checkpoint(
        #         ckpt_file=self.netG_ckpt_file, optimizer=self.optG)

        if global_step_G != global_step_D:
            raise ValueError('G and D Networks are out of sync.')
        else:
            global_step = global_step_G  # Restores global step

        return global_step

    def _save_model_checkpoints(self, global_step):
        """
        Saves both discriminator and generator checkpoints.
        """
        self.netG.save_checkpoint(directory=self.netG_ckpt_dir,
                                  global_step=global_step,
                                  optimizer=self.optG)

        self.netD.save_checkpoint(directory=self.netD_ckpt_dir,
                                  global_step=global_step,
                                  optimizer=self.optD)

    def train(self):
        """
        Runs the training pipeline with all given parameters in Trainer.
        """
        # Restore models
        global_step = self._restore_models_and_step()
        print("INFO: Starting training from global step {}...".format(
            global_step))

        try:
            start_time = time.time()

            # Iterate through data
            iter_dataloader = iter(self.dataloader)
            while global_step < self.num_steps:
                log_data = metric_log.MetricLog()  # log data for tensorboard

                # -------------------------
                #   One Training Step
                # -------------------------
                # Update n_dis times for D
                for i in range(self.n_dis):
                    iter_dataloader, real_batch = self._fetch_data(
                        iter_dataloader=iter_dataloader)

                    # ------------------------
                    #   Update D Network
                    # -----------------------
                    log_data = self.netD.train_step(real_batch=real_batch,
                                                    netG=self.netG,
                                                    optD=self.optD,
                                                    log_data=log_data,
                                                    global_step=global_step,
                                                    device=self.device)

                    # -----------------------
                    #   Update G Network
                    # -----------------------
                    # Update G, but only once.
                    if i == (self.n_dis - 1):
                        log_data = self.netG.train_step(
                            real_batch=real_batch,
                            netD=self.netD,
                            optG=self.optG,
                            global_step=global_step,
                            log_data=log_data,
                            device=self.device)

                # --------------------------------
                #   Update Training Variables
                # -------------------------------
                global_step += 1

               # log_data = self.scheduler.step(log_data=log_data,
               #                                global_step=global_step)
                self.scheduler_d.step()
                self.scheduler_g.step()
                
                _lr_D = self.scheduler_d.get_last_lr()[0]
                _lr_G = self.scheduler_g.get_last_lr()[0]
              
                log_data.add_metric('lr_D', _lr_D, group='lr', precision=6)
                log_data.add_metric('lr_G', _lr_G, group='lr', precision=6)

                # -------------------------
                #   Logging and Metrics
                # -------------------------
                if global_step % self.log_steps == 0:
                    self.logger.write_summaries(log_data=log_data,
                                                global_step=global_step)

                if global_step % self.print_steps == 0:
                    curr_time = time.time()
                    self.logger.print_log(global_step=global_step,
                                          log_data=log_data,
                                          time_taken=(curr_time - start_time) /
                                          self.print_steps)
                    start_time = curr_time

            #    if global_step % self.vis_steps == 0:
            #        self.logger.vis_images(netG=self.netG,
            #                               global_step=global_step)

                if global_step % self.save_steps == 0:
                    print("INFO: Saving checkpoints...")
                    self._save_model_checkpoints(global_step)

                if (global_step + 1) % 3000 == 0: 
                    self.netG.eval()

                    ### HERE IS WHERE WE ADD THE CODE FOR METRICS
                    metrics = torch_fidelity.calculate_metrics(
                        input1=torch_fidelity.GenerativeModelModuleWrapper(self.netG, self.netG.nz, 'normal', 0),
                        input1_model_num_samples=5000,
                        input2= 'stl-10-48',
                        isc=True,
                        fid=True,
                        kid=True,
                        ppl=False,
                        ppl_epsilon=1e-2,
                        ppl_sample_similarity_resize=64,
                    )
                    
                    self.netG.train()


            print("INFO: Saving final checkpoints...")
            self._save_model_checkpoints(global_step)

        except KeyboardInterrupt:
            print("INFO: Saving checkpoints from keyboard interrupt...")
            self._save_model_checkpoints(global_step)

        finally:
            self.logger.close_writers()

        print("INFO: Training Ended.")
