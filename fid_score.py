from models import *
from config import Config
import torch_fidelity
from util import *

config = Config.shared()
device = get_device()


def main():
    ## Reads the parameters send from the user through the terminal call of test.py
    config.read_metrics_options()

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


    ## Loads data for traning based on the config set by the user
    dataset, batch_size, workers = load_data()

    print("Loading dataset...")
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    print("Calculating metrics...")

    netG = FFCCondGenerator(nz=nz, nc=nc, ngf=ngf, num_classes= num_classes, 
                                    embed_size=embed_size, uses_noise=True).to(device) 


    metrics = torch_fidelity.calculate_metrics(
                input1=torch_fidelity.GenerativeModelModuleWrapper(G, nz, "normal", num_classes),
                input1_model_num_samples=number_samples,
                input2='cifar10-train',
                isc=True,
                fid=True,
                kid=True,
                ppl=True,
                ppl_epsilon=1e-2,
                ppl_sample_similarity_resize=64,
            )

    print("Storing metrics...")
    with open('../fid' + str(model_path) + str(number_samples) + '.txt', 'w+') as f:
        for k, v in metrics.items():
            f.write('metrics/' + str(k) + "_" + str(v))

    print("Done!")



main()