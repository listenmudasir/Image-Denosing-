# FILE: network_scunet_faa.py

import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import trunc_normal_, DropPath

# Import original blocks directly from the baseline network file for compatibility
from network_scunet import WMSA, Block, ConvTransBlock

class FrequencyAwareAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(FrequencyAwareAttention, self).__init__()
        reduced_channels = max(8, in_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        original_dtype = x.dtype
        # Force float32 for all internal calculations to ensure stability
        with torch.amp.autocast(device_type='cuda', enabled=False):
            x_float32 = x.float()
            b, c, _, _ = x_float32.shape
            
            fft_x = torch.fft.rfft2(x_float32, norm='ortho')
            fft_mag = torch.abs(fft_x)
            
            pooled_freq = nn.functional.adaptive_avg_pool2d(fft_mag, (1, 1)).view(b, c)
            attention_weights = self.mlp(pooled_freq).view(b, c, 1, 1)

        # Apply attention and cast weights back to the original dtype
        return x * attention_weights.to(original_dtype)

class SCUNetWithFAA(nn.Module):
    def __init__(self, in_nc=3, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=256):
        super(SCUNetWithFAA, self).__init__()
        self.config, self.dim, self.head_dim, self.window_size = config, dim, 32, 8
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        
        self.m_head = nn.Sequential(nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False))
        begin = 0
        self.m_down1 = nn.Sequential(*([ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) for i in range(config[0])] + [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)]))
        begin += config[0]
        self.m_down2 = nn.Sequential(*([ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2) for i in range(config[1])] + [nn.Conv2d(2*dim, 4*dim, 2, 2, 0, bias=False)]))
        begin += config[1]
        self.m_down3 = nn.Sequential(*([ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4) for i in range(config[2])] + [nn.Conv2d(4*dim, 8*dim, 2, 2, 0, bias=False)]))
        begin += config[2]
        self.m_body = nn.Sequential(*[ConvTransBlock(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8) for i in range(config[3])])
        
        self.m_faa = FrequencyAwareAttention(in_channels=8 * dim)

        begin += config[3]
        self.m_up3 = nn.Sequential(*([nn.ConvTranspose2d(8*dim, 4*dim, 2, 2, 0, bias=False)] + [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4) for i in range(config[4])]))
        begin += config[4]
        self.m_up2 = nn.Sequential(*([nn.ConvTranspose2d(4*dim, 2*dim, 2, 2, 0, bias=False)] + [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2) for i in range(config[5])]))
        begin += config[5]
        self.m_up1 = nn.Sequential(*([nn.ConvTranspose2d(2*dim, dim, 2, 2, 0, bias=False)] + [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) for i in range(config[6])]))
        self.m_tail = nn.Sequential(nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False))

    def forward(self, x0, return_attention=False):
        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h/64)*64-h)
        paddingRight = int(np.ceil(w/64)*64-w)
        x0_padded = nn.functional.pad(x0, (0, paddingRight, 0, paddingBottom), 'reflect')
        
        x1 = self.m_head(x0_padded)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        bottleneck_out = self.m_body(x4)
        bottleneck_attended = self.m_faa(bottleneck_out)
        x = self.m_up3(bottleneck_attended + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        x = x[..., :h, :w]
        
        if return_attention:
            # Modify FAA to return weights if you want to use this
            return x, None 
        return x