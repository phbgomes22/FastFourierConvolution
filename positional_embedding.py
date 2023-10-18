
import torch
import numpy as np



def main():
    print("Output value")
    FEATURES = 128
    z = torch.randn(64, FEATURES)
    
   # print(z.shape)

    SCALE = 5.5
    B = SCALE * np.random.normal(size=(128, FEATURES//2))
  #  print(B.shape)
    A = torch.ones(B.shape[1])
    
    encoded = (np.pi * z) @ B
    rff_input = torch.cat([A * encoded.cos(),
                           A * encoded.sin()], dim=-1)
   # print(rff_input.shape)


def main2():
    FEATURES = 128
    z = torch.randn(32, 3, 64, 64)
    
   # print(z.shape)

    SCALE = 5.5
    B = SCALE * np.random.normal(size=(128, FEATURES//2))
  #  print(B.shape)
    A = torch.ones(B.shape[1])
    
    encoded = (np.pi * z) @ B
    rff_input = torch.cat([A * encoded.cos(),
                           A * encoded.sin()], dim=-1)

main2()