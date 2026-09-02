import torch
import torch.nn as nn 

# PyTorch 2.x compatibility: use torch.amp instead of torch.cuda.amp
try:
    from torch.amp import GradScaler as TorchGradScaler
except ImportError:
    # Fallback for older PyTorch versions
    from torch.cuda.amp import GradScaler as TorchGradScaler


from src.core import register
import src.misc.dist as dist 


__all__ = ['GradScaler']

GradScaler = register(TorchGradScaler)
