import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from util import *
from models import *
from torchsummary import summary


device = get_device()

config = Config.shared()

# Initialize BCELoss function
criterion = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init(m):
    '''
    Custom weights initialization called on netG and netD
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        



def get_generator():
    '''
    Creates and returns the Generator model. 
    The parameter -g / --generator provided by the user will control whether the generator
    has fourier convolutions or regular ones. The default value is to use FFC.
    Weight initialization is set by applying the `weights_init` function.
    '''
    ## Getting parameters
    ngf = config.ngf
    ngpu = config.ngpu
    nz = config.nz
    nc = config.nc
    image_size = config.image_size
    num_classes = config.num_classes
    embed_size = config.gen_embed

    
    ## Creating generator
    netG = None
    if config.FFC_GENERATOR:
        netG = FFCCondGenerator(nz=nz, nc=nc, ngf=ngf, num_classes= num_classes, 
                                embed_size=embed_size, uses_noise=True).to(device) 
        
    else:
        netG = CondCvGenerator(nz=nz, nc=nc, ngf=ngf, 
                        num_classes= num_classes, image_size=image_size, 
                        embed_size=embed_size).to(device)
        

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    params = count_parameters(netG)
    print("- Parameters on generator: ", params)
    return netG


def get_discriminator():
    '''
    Creates and returns the Discriminator model. 
    The parameter -d / --discriminator provided by the user will control whether the discriminator
    has fourier convolutions or regular ones. The default is using regular convolutions.
    Weight initialization is set by applying the `weights_init` function.
    '''

    ngpu = config.ngpu
    ndf = config.ndf
    nc = config.nc
    num_epochs = config.num_epochs
    num_classes = config.num_classes
    DEBUG = config.DEBUG

    # Create the Discriminator
    if config.FFC_DISCRIMINATOR:
        netD = FFCCondDiscriminator(nc=nc, ndf=ndf, num_classes=num_classes, num_epochs=num_epochs, uses_sn=False, uses_noise=True).to(device)
    else:
        netD = CondDiscriminator(nc=nc, ndf=ndf, num_classes=num_classes, num_epochs=num_epochs, uses_sn=False, uses_noise=True).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
        
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    summary(netD, [(nc, ndf, ndf), (1,), 10])
    
    params = count_parameters(netD)
    print("- Parameters on discriminator: ", params)
    return netD




def train(netG, netD):
    '''
    Controls the training loop of the GAN.
    '''
    ## parameters
    beta1 = config.beta1
    lr = config.lr
    num_epochs = config.num_epochs
    nz = config.nz
    model_output = config.model_output
    num_classes = config.num_classes
    image_size = config.image_size

    ## Loads data for traning based on the config set by the user
    dataset, batch_size, workers = load_data()

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
                                            #pin_memory=True, persistent_workers=True)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device) # 1, 1,

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    iters = 0

    #
    labels = range(num_classes)
    fixed_labels = torch.nn.functional.one_hot( torch.as_tensor( np.repeat(labels, 8)[:64] ) ).float().to(device)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            ## (0) Conditional Training 
            ## Getting labels
            ############################
            labels = data[1].to(device)
        
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Conditional Training 
            # Forward pass real batch through D alonside one_hot_labels
            output = netD(real_cpu, labels, epoch).view(-1)
    
            # Calculate loss on all-real batch
            errD_real = criterion(output, label.float())
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Conditional Training 
            # Generate fake image batch with G alongside one_hot_labels
            fake = netG(noise, labels)

            label.fill_(fake_label)
            # Conditional Training 
            # Classify all fake batch with D  alongside one_hot_labels
            output = netD(fake.detach(), labels, epoch).view(-1)
            
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label.float())
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Conditional Training 
            # Since we just updated D, perform another forward pass of all-fake batch through D  alongside one_hot_labels
            output = netD(fake, labels, epoch).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label.float())
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            # Output training stats
            if i % 16 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                if i % 32 == 0 and epoch%2 == 0:
                    with torch.no_grad():
                        # Conditional training - sampling
                        fake = netG(fixed_noise, torch.argmax(fixed_labels, dim=1)).detach().cpu()
                    curr_fake = vutils.make_grid(fake, padding=2, normalize=True)
                    # save an image with samples of the generator model output
                    save_grid_images(curr_fake, epoch, i, model_output)

                    # saves the generator model from the current epoch and batch
                    torch.save(netG.state_dict(), model_output + "generator"+ str(epoch) + "_" + str(i))
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            iters += 1
    
    print("Training Loop Ended!")
    save_training_plot(G_losses=G_losses, D_losses=D_losses, epoch=num_epochs, model_output=model_output)


def main():

    ## Reads the parameters send from the user through the terminal call of train.py
    config.read_train_params()

    print("Will create models...")
    ## Creating generator and discriminator
    netG = get_generator()
    print("Generator created!")
    netD = get_discriminator()
    print("Discriminator created!")

    print("Will begin training... ")
    train(netG, netD)

main()