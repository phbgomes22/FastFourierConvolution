from models import *
from config import Config
import torch_fidelity
from util import *

config = Config.shared()
device = get_device()

class DropLabelsDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, index):
        item = self.ds[index]
        assert type(item) in (tuple, list)
        returned_item = item[0].to(torch.uint8)
      
        return returned_item

    def __len__(self):
        return len(self.ds)

def main():
    ## Reads the parameters send from the user through the terminal call of test.py
    config.read_metrics_params()

    test()

def test():
    ngf = config.ngf
    nz = config.nz
    nc = config.nc
    model_path = config.model_path
    number_samples = config.samples
    output_dir = config.sample_output
    num_classes = config.num_classes
    embed_size = config.gen_embed

    file_name = ""

    print("Loading dataset...")
    ## Loads data for traning based on the config set by the user
    dataset, _, _ = load_data()

    file_name += config.dataset_name
    metrics_dataset = DropLabelsDataset(dataset)

    print("Calculating metrics...")


    ## Creating generator
    netG = None
    if config.FFC_GENERATOR:
        netG = FFCCondGenerator(nz=nz, nc=nc, ngf=ngf, num_classes= num_classes, 
                                    embed_size=embed_size, uses_noise=True, training=False).to(device) 
        file_name += "_fgan_"
    else:
        netG = CondCvGenerator(nz=nz, nc=nc, ngf=ngf, 
                        num_classes= num_classes, 
                        embed_size=embed_size, training=False).to(device)
        file_name += "_vanilla_"

    ### - DEBUGGING
        if config.DEBUG:
        # create noise array
        noise = torch.randn(2*num_classes, nz, 1, 1, device=device)

        # create label array
        num_per_class = 2
        labels = torch.zeros(2*num_classes, dtype=torch.long)

        for i in range(num_classes):
            labels[i*num_per_class : (i+1)*num_per_class] = i
        labels = labels.to(device)

        with torch.no_grad():
            fake = netG(noise, labels).detach().cpu()#.numpy()

        # save the samples 
        for f in fake:
            generated_image = np.transpose(f, (1,2,0))
            im = Image.fromarray((generated_image.squeeze(axis=2).numpy()))
            im.save("../metrics/" + 'image' + str(count) + ".jpg")
            count+=1
        return

    ### - END DEBUGGING

    ### - Running metrics
    metrics = torch_fidelity.calculate_metrics(
                input1=torch_fidelity.GenerativeModelModuleWrapper(netG, nz, "normal", num_classes),
                input1_model_num_samples=number_samples,
                input2=metrics_dataset,
                isc=True,
                fid=True,
                kid=True,
                ppl=False,
                ppl_epsilon=1e-2,
                ppl_sample_similarity_resize=32,
            )

    print("Storing metrics...")
    with open('../metrics/metrics_' + file_name + '.txt', 'w+') as f:
        for k, v in metrics.items():
            f.write(str(model_path) + " " + str(k) + " " + str(v) + '\n')

    print("Done!")



main()