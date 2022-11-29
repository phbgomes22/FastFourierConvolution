import torch.nn as nn
from util import *
from .spectral_transform import SpectralTransform

'''
The FFC Layer

It represents the module that receives the total signal, splits into local and global signals and returns the complete signal in the end.
This represents the layer of the Fast Fourier Convolution that comes in place of a vanilla convolution layer.

It contains:
    Conv2ds with a kernel_size received as a parameter from the __init__ in `kernel_size`.
    The Spectral Transform module for the processing of the global signal. 
'''
class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        debug_print("****")
        debug_print(kernel_size, padding, stride)
        debug_print("****")

        # calculate the number of input and output channels based on the ratio (alpha) 
        # of the local and global signals 
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        # defines the module as a Conv2d unless the channels input or output are zero
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        # this is the convolution that processes the local signal and contributes 
        # for the formation of the outputted local signal

        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)

        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        # this is the convolution that processes the local signal and contributes 
        # for the formation of the outputted global signal
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)

        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        # this is the convolution that processes the global signal and contributes 
        # for the formation of the outputted local signal
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)

        # defines the module as the Spectral Transform unless the channels output are zero
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform

        # (Fourier)
        # this is the convolution that processes the global signal and contributes (in the spectral domain)
        # for the formation of the outputted global signal 
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)


    # receives the signal as a tuple containing the local signal in the first position
    # and the global signal in the second position
    def forward(self, x):
        # splits the received signal into the local and global signals
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            # creates the output local signal passing the right signals to the right convolutions
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            # creates the output global signal passing the right signals to the right convolutions
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        # returns both signals as a tuple
        return out_xl, out_xg



'''
The FFC Layer

It represents the module that receives the total signal, splits into local and global signals and returns the complete signal in the end.
This represents the layer of the Fast Fourier Convolution that comes in place of a vanilla convolution layer.

It contains:
    Conv2ds with a kernel_size received as a parameter from the __init__ in `kernel_size`.
    The Spectral Transform module for the processing of the global signal. 
'''
### EDITADO PARA FAZER UPSAMPLING COM FFC!!!
class FFCTranspose(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0, 
                 dilation=1, groups=1, bias=False, enable_lfu=True, out_padding=0):
        super(FFCTranspose, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        # calculate the number of input and output channels based on the ratio (alpha) 
        # of the local and global signals 
        in_cg = int(in_channels * ratio_gin)
        in_cl = int(in_channels - in_cg)
        out_cg = int(out_channels * ratio_gout)
        out_cl = int(out_channels - out_cg)
        
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        
        # defines the module as a Conv2d unless the channels input or output are zero
        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.ConvTranspose2d
        # this is the convolution that processes the local signal and contributes 
        # for the formation of the outputted local signal

        debug_print("----")    
        debug_print(in_cg, in_cl,  kernel_size, padding, stride)
        debug_print("----")

        # (in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t=1, padding: _size_2_t=0, 
        # output_padding: _size_2_t=0, groups: int=1, bias: bool=True, dilation: int=1, padding_mode: str='zeros', device=None, dtype=None)
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, output_padding=out_padding, groups=groups, bias=bias, dilation=dilation)

        
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.ConvTranspose2d
        # this is the convolution that processes the local signal and contributes 
        # for the formation of the outputted global signal
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, output_padding=out_padding, groups=groups, bias=bias, dilation=dilation)

       
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.ConvTranspose2d
        # this is the convolution that processes the global signal and contributes 
        # for the formation of the outputted local signal
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, output_padding=out_padding, groups=groups, bias=bias, dilation=dilation)

        # defines the module as the Spectral Transform unless the channels output are zero
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform

        # (Fourier)
        # this is the convolution that processes the global signal and contributes (in the spectral domain)
        # for the formation of the outputted global signal 
       # self.upsample = 

        self.convg2g = nn.Sequential(

           # nn.ConvTranspose2d(in_cg,  out_cg, kernel_size,
           #                   stride, padding, output_padding=out_padding, groups=groups, bias=bias, dilation=dilation),
            #module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu),

            # SpectralTransform (self, in_channels, out_channels, stride=1, groups=1, enable_lfu=False)
            module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu),
            ## [PG] - already tested: inverting Umpsample with Spectral Convolution didnt work
            ## [PG - 22/aug/02] - upsample may be a problem, will test with another conv2d
           # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
            nn.ConvTranspose2d(out_cg,  out_cg*2, kernel_size,
                              stride, padding, output_padding=out_padding, groups=groups, bias=bias, dilation=dilation)
            ## UPSAMPLING DO SPECTRAL COM BILINEAR!!!
        )
        ## -- debugging
        self.print_size = nn.Sequential(Print(debug=DEBUG))
        


    # receives the signal as a tuple containing the local signal in the first position
    # and the global signal in the second position
    def forward(self, x):
        # splits the received signal into the local and global signals
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0


        if self.ratio_gout != 1:
            # creates the output local signal passing the right signals to the right convolutions
            debug_print(".  --- FFC Transp")
            self.print_size(x_l)
            out_xl = self.convl2l(x_l) 
            debug_print(".  --- Conv2l2")

            self.print_size(out_xl)
            out_xl = out_xl + self.convg2l(x_g)
            debug_print(".  --- Convgl2")

            self.print_size(out_xl)
            debug_print(".  --- Fim FFC Transp")

        if self.ratio_gout != 0:
            # creates the output global signal passing the right signals to the right convolutions
            out_xg = self.convl2g(x_l)

            if type(x_g) is tuple:
                out_xg = out_xg + self.convg2g(x_g)
               
        
        # returns both signals as a tuple
        return out_xl, out_xg



'''
Creates a single FFC -> Batch normalization -> Activation Module flow.

This is the class that is put in the models as a blackbox. 
So this is on of the entry point of all code related to the FFC (the other being the FFCSE_block).

It has:
    The FFC layer module.
    Followed by Bach Normalization components for both the local and global signals.
    Followed by an ActivationLayer 
        -   The default activation layer is nn.Identity, so I think we are supposed to change it.
'''
## ADICIONEI PARAMETRO UPSAMPLING
class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 enable_lfu=True, upsampling=False, out_padding=0):
        super(FFC_BN_ACT, self).__init__()

        # Creates the FFC layer, that will process the signal 
        # (divided into local and global and apply the convolutions and Fast Fourier)
        if upsampling:
            self.ffc = FFCTranspose(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, out_padding)
        else:
            self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)

        # create the BatchNormalization layers
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))

        # create the activation function layers
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer

        if lact is nn.Tanh or lact is nn.Sigmoid:
            self.act_l = lact() # was inplace=True
        else:
            self.act_l = lact(inplace=True)

        if gact is nn.Tanh or gact is nn.Sigmoid:
            self.act_g = gact() # was inplace=True
        else:
            self.act_g = gact(inplace=True)

        self.print_size = Print(debug=DEBUG)

    def forward(self, x):
        debug_print(" -- FFC_BN_ACT")
        x_l, x_g = self.ffc(x)
        self.print_size(x_l)
        
        x_l = self.act_l(self.bn_l(x_l))
        self.print_size(x_l)

        x_g = self.act_g(self.bn_g(x_g))
        debug_print(" -- Fim FFC_BN_ACT")
        return x_l, x_g
