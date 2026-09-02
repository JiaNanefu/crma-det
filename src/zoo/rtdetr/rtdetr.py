"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'rgfr_lite', ]

    def __init__(self,
                 backbone: nn.Module,
                 encoder,
                 decoder,
                 use_rgfr_lite=False,
                 rgfr_lite=None,
                 multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.use_rgfr_lite = bool(use_rgfr_lite)
        if self.use_rgfr_lite and rgfr_lite is None:
            raise ValueError("use_rgfr_lite=True requires an rgfr_lite module.")
        self.rgfr_lite = rgfr_lite if self.use_rgfr_lite else None
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            # NumPy may return a scalar integer type that PyTorch rejects as an interpolation size.
            sz = int(np.random.choice(self.multi_scale))
            x = F.interpolate(x, size=[sz, sz])
            
        x = self.backbone(x)
        x = self.encoder(x)

        if self.use_rgfr_lite and self.rgfr_lite is not None:
            x = self.rgfr_lite(x)

        x = self.decoder(x, targets)

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
