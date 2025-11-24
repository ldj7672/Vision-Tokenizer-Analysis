"""
Simple EMA (Exponential Moving Average) implementation for inference.
"""
import torch
from torch import nn
from collections import OrderedDict


class LitEma(nn.Module):
    """Lightweight EMA for model parameters."""
    
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        self.decay = decay
        self.use_num_upates = use_num_upates
        self.num_updates = 0
        
        # Store shadow parameters
        self.shadow_params = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()
    
    def update(self, model):
        """Update EMA parameters."""
        if self.use_num_upates:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        else:
            decay = self.decay
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.shadow_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    def copy_to(self, model):
        """Copy EMA parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                param.data.copy_(self.shadow_params[name])
    
    def store(self, model):
        """Store current model parameters."""
        self.collected_params = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.collected_params[name] = param.data.clone()
    
    def restore(self, model):
        """Restore stored model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.collected_params:
                param.data.copy_(self.collected_params[name])


