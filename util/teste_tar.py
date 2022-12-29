import torch
import tar_loader
import torchvision.transforms as transforms
import numpy as np
import matplotlib as plt
import torchvision.utils as vutils


def test():

    list_transforms = [
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ]
    transform = transforms.Compose(list_transforms)

    dataset = tar_loader.TarImageFolder('../../celeba_data.tar', transform=transform)

    print("dataset loaded")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                            shuffle=True, num_workers=1)

    print("dataloader created")

    count = 0
    for (image, label) in dataloader:
        print(f"Dimensions of image batch: {image.shape}")
        print(f"Labels in batch: {label}")
        count += 1
        if count == 5: 
            break

    print("Dataloader created.")

    model_output = '.'
    device = "cpu"
    # Plot some training images
    try:
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.savefig(model_output + "training_set.jpg")
    except OSError:
        print("Cannot load image")



test()