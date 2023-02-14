

class FFC_Settings: 
    def __init__(self, bias: bool = False, enable_lfu: bool = True, noise_injection: bool = False, 
                attention: bool = False,  spectral_norm: bool = False):
        self.bias = bias
        self.enable_lfu = enable_lfu
        self.noise_injection = noise_injection
        self.spectral_norm = spectral_norm
        self.attention = attention

        
