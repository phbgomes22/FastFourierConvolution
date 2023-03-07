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


    print("Loading dataset...")
    ## Loads data for traning based on the config set by the user
    dataset, _, _ = load_data()

    metrics_dataset = DropLabelsDataset(dataset)

    print("Calculating metrics...")


    ## Creating generator
    netG = None
    
    if config.FFC_GENERATOR:
        netG = FFCCondGenerator(nz=nz, nc=nc, ngf=ngf, num_classes= num_classes, 
                                    embed_size=embed_size, uses_noise=True, training=False).to(device) 
    else:
        netG = CondCvGenerator(nz=nz, nc=nc, ngf=ngf, 
                        num_classes= num_classes, 
                        embed_size=embed_size, training=False).to(device)

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
    with open('../fid' + str(model_path) + str(number_samples) + '.txt', 'w+') as f:
        for k, v in metrics.items():
            f.write('metrics/' + str(k) + "_" + str(v))

    print("Done!")



main()