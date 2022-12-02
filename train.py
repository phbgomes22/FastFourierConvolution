import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from util import *
from models import *

# Initialize BCELoss function
criterion = nn.BCELoss()

device = get_device()

config = Config.shared()

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



def get_generator():
    ## Getting parameters
    ngf = config.ngf
    ngpu = config.ngpu
    nz = config.nz
    nc = config.nc

    ## Creating generator
    netG = None
    if config.FFC_GENERATOR:
        netG = FFCGenerator(nz, nc, ngf, debug=config.DEBUG).to(device) 
    else:
        Generator(nz, nc, ngf).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    return netG


def get_discriminator():

    ngpu = config.ngpu
    ndf = config.ndf
    nc = config.nc
    DEBUG = config.DEBUG

    # Create the Discriminator
    netD = None
    if config.FFC_DISCRIMINATOR:
        netD = FFCDiscriminator(nc, ndf, debug=DEBUG).to(device)
    else:
        netD = Discriminator(nc, ndf, ngpu=ngpu).to(device) 

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
        
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    # print(netD)
    return netD


def main():

    ## Reads the parameters send from the user through the terminal call of train.py
    config.read_params()

    ## Loads data for traning based on the config set by the user
    dataloader = load_data()

    print("Will create models...")
    ## Creating generator and discriminator
    netG = get_generator()
    print("Generator created!")
    netD = get_discriminator()
    print("Discriminator created!")

    print("Will begin training... ")
    train(netG, netD, dataloader)



def train(netG, netD, dataloader):
    ## parameters
    beta1 = config.beta1
    lr = config.lr
    num_epochs = config.num_epochs
    nz = config.nz
    model_output = config.model_output

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(int(num_epochs / 2)):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
    
            # Calculate loss on all-real batch
            errD_real = criterion(output, label.float())
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)

            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
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
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label.float())
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            # Output training stats
            if i % 50 == 0 and epoch%4 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, int(num_epochs / 2), i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                curr_fake = vutils.make_grid(fake, padding=2, normalize=True)
                img_list.append(curr_fake)
                image_to_show = np.transpose(curr_fake, (1,2,0))
                plt.figure(figsize=(5,5))
                plt.imshow(image_to_show)
                plt.savefig(model_output + "image" + str(epoch) + "_" + str(i) + ".jpg")
                torch.save(netG.state_dict(), model_output + "generator"+ str(epoch) + "_" + str(i))
                plt.show()
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())


            if len(D_losses) > 1 and abs(D_losses[-1] - D_losses[-2]) > 20.0:
                break
            
            # Check how the generator is doing by saving G's output on fixed_noise
            # if (iters % 250 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                
            iters += 1


main()