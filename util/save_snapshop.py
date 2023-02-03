
#import matplotlib.pyplot as plt
from .data_loader import *
import numpy as np



def save_training_plot(G_losses, D_losses, epoch: int, model_output: str):
    plt.figure(figsize=(20,10))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(model_output + "image" + str(epoch) + "epochs.jpg")
    plt.show()


def save_grid_images(imgs, epoch: int, iter: int, model_output: str):
    image_to_show = np.transpose(imgs, (1,2,0))
    plt.figure(figsize=(10,10))
    plt.imshow(image_to_show)
    # saves the image representing samples from the generator
    plt.savefig(model_output + "image" + str(epoch) + "_" + str(iter) + ".jpg")
    # saves the generator model from the current epoch and batch
    plt.show()