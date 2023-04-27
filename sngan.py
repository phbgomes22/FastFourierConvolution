import torch
from torch import nn
import torch.nn.functional as F

import argparse
import os

import PIL
import torchvision
import tqdm

from torch.utils import tensorboard

import torch_fidelity

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes, eps=1e-4, momentum=0.1):
    super().__init__()
    self.num_features = num_features
    self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum)
    self.gamma_embed = nn.Linear(num_classes, num_features, bias=False)
    self.beta_embed = nn.Linear(num_classes, num_features, bias=False)

  def forward(self, x, y):
    out = self.bn(x)
    gamma = self.gamma_embed(y) + 1
    beta = self.beta_embed(y)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out

def init_xavier_uniform(layer):
    if hasattr(layer, "weight"):
        torch.nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if hasattr(layer.bias, "data"):       
            layer.bias.data.fill_(0)

## Generator block
class DeconvBNRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding=0, n_classes=0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding=padding)
        if n_classes == 0:
            self.bn = nn.BatchNorm2d(out_ch)
        else:
            self.bn = ConditionalBatchNorm2d(out_ch, n_classes)
        self.relu = nn.ReLU(True)

        self.conv.apply(init_xavier_uniform)

    def forward(self, inputs, label_onehots=None):
        x = self.conv(inputs)
        if label_onehots is not None:
            x = self.bn(x, label_onehots)
        else:
            x = self.bn(x)
        return self.relu(x)

class GeneratorResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsampling, n_classes=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.upsampling = upsampling
        if n_classes == 0:
            self.bn1 = nn.BatchNorm2d(in_ch)
            self.bn2 = nn.BatchNorm2d(out_ch)
        else:
            self.bn1 = ConditionalBatchNorm2d(in_ch, n_classes)
            self.bn2 = ConditionalBatchNorm2d(out_ch, n_classes)
        if in_ch != out_ch or upsampling > 1:
            self.shortcut_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        else:
            self.shortcut_conv = None

        self.conv1.apply(init_xavier_uniform)
        self.conv2.apply(init_xavier_uniform)

    def forward(self, inputs, label_onehots=None):
        # main
        if label_onehots is not None:
            x = self.bn1(inputs, label_onehots)
        else:
            x = self.bn1(inputs)
        x = F.relu(x)

        if self.upsampling > 1:
            x = F.interpolate(x, scale_factor=self.upsampling)
        x = self.conv1(x)

        if label_onehots is not None:
            x = self.bn2(x, label_onehots)
        else:
            x = self.bn2(x)
        x = F.relu(x)

        x = self.conv2(x)

        # short cut
        if self.upsampling > 1:
            shortcut = F.interpolate(inputs, scale_factor=self.upsampling)
        else:
            shortcut = inputs
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
        # residual add
        return x + shortcut
        
## Discriminator Block
class ConvSNLRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding=0, lrelu_slope=0.1):
        super().__init__()
        sn_fn = torch.nn.utils.spectral_norm 
        self.conv = sn_fn(nn.Conv2d(in_ch, out_ch, kernel, stride, padding=padding))
        self.lrelu = nn.LeakyReLU(lrelu_slope, True)
        
        self.conv.apply(init_xavier_uniform)

    def forward(self, inputs):
        return self.lrelu(self.conv(inputs))

class SNEmbedding(nn.Module):
    def __init__(self, n_classes, out_dims):
        super().__init__()
        sn_fn = torch.nn.utils.spectral_norm 
        self.linear = sn_fn(nn.Linear(n_classes, out_dims, bias=False))

        self.linear.apply(init_xavier_uniform)

    def forward(self, base_features, output_logits, label_onehots):
        wy = self.linear(label_onehots)
        weighted = torch.sum(base_features * wy, dim=1, keepdim=True)
        return output_logits + weighted

class DiscriminatorSNResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsampling):
        super().__init__()
        sn_fn = torch.nn.utils.spectral_norm 
        self.conv1 = sn_fn(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        self.conv2 = sn_fn(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        self.downsampling = downsampling
        if in_ch != out_ch or downsampling > 1:
            self.shortcut_conv = sn_fn(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0))
        else:
            self.shortcut_conv = None

        self.conv1.apply(init_xavier_uniform)
        self.conv2.apply(init_xavier_uniform)

    def forward(self, inputs):
        x = F.relu(inputs)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        # short cut
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(inputs)
        else:
            shortcut = inputs
        if self.downsampling > 1:
            x = F.avg_pool2d(x, kernel_size=self.downsampling)
            shortcut = F.avg_pool2d(shortcut, kernel_size=self.downsampling)
        # residual add
        return x + shortcut
    
##
## - Models
## 

class Generator(nn.Module):
    def __init__(self, dataset="cifar", enable_conditional=False):
        super().__init__()
        if dataset in ["cifar", "svhn"]:
            self.mg = 4
        elif dataset == "stl":
            self.mg = 6
        if enable_conditional:
            n_classes = 10
        else:
            n_classes = 0

        self.dense = nn.Linear(128, self.mg * self.mg * 512)
        self.conv1 = DeconvBNRelu(512, 256, 4, 2, padding=1, n_classes=n_classes)
        self.conv2 = DeconvBNRelu(256, 128, 4, 2, padding=1, n_classes=n_classes)
        self.conv3 = DeconvBNRelu(128, 64, 4, 2, padding=1, n_classes=n_classes)
        self.out = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()            
        )

    def forward(self, inputs, label_onehots=None):
        x = self.dense(inputs).view(inputs.size(0), 512, self.mg, self.mg)
        x = self.conv3(self.conv2(self.conv1(x, label_onehots), label_onehots), label_onehots)
        return self.out(x)


class Discriminator(nn.Module):
    def __init__(self, dataset="cifar", enable_conditional=False):
        super().__init__()
        if dataset in ["cifar", "svhn"]:
            self.mg = 4
        elif dataset == "stl":
            self.mg = 6
        if enable_conditional:
            n_classes = 10
        else:
            n_classes = 0

        self.conv1 = self.discriminator_block(3, 64)
        self.conv2 = self.discriminator_block(64, 128)
        self.conv3 = self.discriminator_block(128, 256)
        self.conv4 = ConvSNLRelu(256, 512, 3, 1, padding=1)
        self.dense = nn.Linear(self.mg * self.mg * 512, 1)
        if n_classes > 0:
            self.sn_embedding = SNEmbedding(n_classes, self.mg * self.mg * 512)
        else:
            self.sn_embedding = None

    def discriminator_block(self, in_ch, out_ch):
        return nn.Sequential(
            ConvSNLRelu(in_ch, out_ch, 3, 1, padding=1),
            ConvSNLRelu(out_ch, out_ch, 4, 2, padding=1)
        )

    def forward(self, inputs, label_onehots=None):
        x = self.conv4(self.conv3(self.conv2(self.conv1(inputs))))
        base_feature = x.view(inputs.size(0), -1)
        x = self.dense(base_feature)
        if self.sn_embedding is not None:
            x = self.sn_embedding(base_feature, x, label_onehots)
        return x
    


##
## - Train
##

def hinge_loss_dis(fake, real):
   # fake = fake.squeeze(-1).squeeze(-1)
  #  real = real.squeeze(-1).squeeze(-1)
    assert fake.dim() == 2 and fake.shape[1] == 1 and real.shape == fake.shape, f'{fake.shape} {real.shape}'
    loss = torch.nn.functional.relu(1.0 - real).mean() + \
           torch.nn.functional.relu(1.0 + fake).mean()
    return loss


def hinge_loss_gen(fake):
   # fake = fake.squeeze(-1).squeeze(-1)
    assert fake.dim() == 2 and fake.shape[1] == 1
    loss = -fake.mean()
    return loss


def train(args):
    # set up dataset loader
    dir = os.getcwd()
    dir_dataset_name = 'dataset_' + str(args.dataset)
    dir_dataset = os.path.join(dir, dir_dataset_name)
    os.makedirs(dir_dataset, exist_ok=True)
    image_size = 32 if args.dataset == 'cifar10' else 48
    ds_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    if args.dataset == 'cifar10':
        ds_instance = torchvision.datasets.CIFAR10(dir_dataset, train=True, download=True, transform=ds_transform)
        mg = 4
    else:
        ds_instance = torchvision.datasets.STL10(dir_dataset, split='train', download=True, transform=ds_transform)
        mg = 6

    loader = torch.utils.data.DataLoader(
        ds_instance, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=8, pin_memory=True
    )
    loader_iter = iter(loader)

    # reinterpret command line inputs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 10 if args.conditional else 0  # unconditional
    leading_metric, last_best_metric, metric_greater_cmp = {
        'ISC': (torch_fidelity.KEY_METRIC_ISC_MEAN, 0.0, float.__gt__),
        'FID': (torch_fidelity.KEY_METRIC_FID, float('inf'), float.__lt__),
        'KID': (torch_fidelity.KEY_METRIC_KID_MEAN, float('inf'), float.__lt__),
        'PPL': (torch_fidelity.KEY_METRIC_PPL_MEAN, float('inf'), float.__lt__),
    }[args.leading_metric]

    # create Generator and Discriminator models
    G = Generator().to(device).train()
   # G.apply(weights_init)
    params = count_parameters(G)
    print(G)
    
    print("- Parameters on generator: ", params)

    D = Discriminator().to(device).train() #LargeF
   # D.apply(weights_init)
    params = count_parameters(D)
    print("- Parameters on discriminator: ", params)
    print(D)

    # initialize persistent noise for observed samples
    z_vis = torch.randn(64, args.z_size, device=device)
    
    # prepare optimizer and learning rate schedulers (linear decay)
    optim_G = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_D = torch.optim.AdamW(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lambda step: 1. - step / args.num_total_steps)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optim_D, lambda step: 1. - step / args.num_total_steps)

    # initialize logging
    tb = tensorboard.SummaryWriter(log_dir=args.dir_logs)
    pbar = tqdm.tqdm(total=args.num_total_steps, desc='Training', unit='batch')
    os.makedirs(args.dir_logs, exist_ok=True)

    for step in range(args.num_total_steps):
        # read next batch
        try:
            real_img, real_label = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            real_img, real_label = next(loader_iter)
        real_img = real_img.to(device)
        real_label = real_label.to(device)
        
        # update Generator
        G.requires_grad_(True)
        D.requires_grad_(False)
        z = torch.randn(args.batch_size, args.z_size, device=device)

        optim_D.zero_grad()
        optim_G.zero_grad()
        fake = G(z)
        loss_G = hinge_loss_gen(D(fake))
        loss_G.backward()
        optim_G.step()

        # update Discriminator
        G.requires_grad_(False)
        D.requires_grad_(True)
        for d_iter in range(args.num_dis_updates):
            z = torch.randn(args.batch_size, args.z_size, device=device)
            optim_D.zero_grad()
            optim_G.zero_grad()
            fake = G(z)
            loss_D = hinge_loss_dis(D(fake), D(real_img))
            loss_D.backward()
            optim_D.step()

        # log
        if (step + 1) % 10 == 0:
            step_info = {'loss_G': loss_G.cpu().item(), 'loss_D': loss_D.cpu().item()}
            pbar.set_postfix(step_info)
            for k, v in step_info.items():
                tb.add_scalar(f'loss/{k}', v, global_step=step)
            tb.add_scalar(f'LR/lr', scheduler_G.get_last_lr()[0], global_step=step)
        pbar.update(1)

        # decay LR
        scheduler_G.step()
        scheduler_D.step()

        # check if it is validation time
        next_step = step + 1
        if next_step % (args.num_epoch_steps) != 0:
            continue
        pbar.close()
        G.eval()
        print('Evaluating the generator...')

        # compute and log generative metrics
        metrics = torch_fidelity.calculate_metrics(
            input1=torch_fidelity.GenerativeModelModuleWrapper(G, args.z_size, args.z_type, num_classes),
            input1_model_num_samples=args.num_samples_for_metrics,
            input2= args.dataset + '-train',
            isc=True,
            fid=True,
            kid=True,
            ppl=False,
            ppl_epsilon=1e-2,
            ppl_sample_similarity_resize=64,
        )
        
        # log metrics
        for k, v in metrics.items():
            tb.add_scalar(f'metrics/{k}', v, global_step=next_step)

        # log observed images
        samples_vis = G(z_vis).detach().cpu()
        samples_vis = torchvision.utils.make_grid(samples_vis).permute(1, 2, 0).numpy()
        tb.add_image('observations', samples_vis, global_step=next_step, dataformats='HWC')
        samples_vis = PIL.Image.fromarray(samples_vis)
        samples_vis.save(os.path.join(args.dir_logs, f'{next_step:06d}.png'))

        # save the generator if it improved
        if metric_greater_cmp(metrics[leading_metric], last_best_metric):
            print(f'Leading metric {args.leading_metric} improved from {last_best_metric} to {metrics[leading_metric]}')

            last_best_metric = metrics[leading_metric]

        # resume training
        if next_step <= args.num_total_steps:
            pbar = tqdm.tqdm(total=args.num_total_steps, initial=next_step, desc='Training', unit='batch')
            G.train()

    tb.close()
    print(f'Training finished; the model with best {args.leading_metric} value ({last_best_metric}) is saved as '
          f'{args.dir_logs}/generator.onnx and {args.dir_logs}/generator.pth')


def main():
    dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_total_steps', type=int, default=100000)
    parser.add_argument('--num_epoch_steps', type=int, default=5000)
    parser.add_argument('--num_dis_updates', type=int, default=1)
    parser.add_argument('--num_samples_for_metrics', type=int, default=10000)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10', 'stl10'))
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--z_size', type=int, default=128, choices=(128,))
    parser.add_argument('--z_type', type=str, default='normal')
    parser.add_argument('--leading_metric', type=str, default='ISC', choices=('ISC', 'FID', 'KID', 'PPL'))
    parser.add_argument('--disable_sn', default=False, action='store_true')
    parser.add_argument('--conditional', default=False, action='store_true')
    parser.add_argument('--dir_logs', type=str, default=os.path.join(dir, 'logs_fgan'))
    args = parser.parse_args()
    print('Configuration:\n' + ('\n'.join([f'{k:>25}: {v}' for k, v in args.__dict__.items()])))
  #  assert not args.conditional, 'Conditional mode not implemented'
    train(args)


main()