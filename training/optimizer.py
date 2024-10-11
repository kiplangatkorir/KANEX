import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class KANEXOptimizer:
    @staticmethod
    def create_optimizer(model: torch.nn.Module, lr: float = 1e-4, weight_decay: float = 0.01):
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    @staticmethod
    def create_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int, min_lr: float = 1e-5):
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)