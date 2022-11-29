# Folder to store models
model_output = '../model_cifar10_baw/'

# Number of workers for dataloader
workers = 4

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
num_epochs = 5000

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Controls the FFC layer ratio of local and global signals
gout_factor = 0.5

DEBUG = False

FFC_GENERATOR = True

FFC_DISCRIMINATOR = False