import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class KANAttention(nn.Module):
    def __init__(self, dim: int, num_functions: int = 16):
        super().__init__()
        self.dim = dim
        self.num_functions = num_functions
        
        # KAN basis functions
        self.basis_functions = nn.Parameter(torch.randn(num_functions, dim) / math.sqrt(dim))
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input to query, key, value spaces
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Compute KAN attention scores
        kan_q = torch.einsum('bld,fd->blf', q, self.basis_functions)
        kan_k = torch.einsum('bld,fd->blf', k, self.basis_functions)
        
        # Compute attention weights using KAN representation
        attention = torch.einsum('blf,bmf->blm', kan_q, kan_k) / math.sqrt(self.dim)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention, v)
        return output

class KANBlock(nn.Module):
    def __init__(self, dim: int, num_functions: int = 16, ff_dim: int = 1024):
        super().__init__()
        self.attention = KANAttention(dim, num_functions)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class KANEX(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int,
        dim: int = 256,
        num_layers: int = 4,
        num_functions: int = 16,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        super().__init__()
        self.config = {
            'vocab_size': vocab_size,
            'max_seq_length': max_seq_length,
            'dim': dim,
            'num_layers': num_layers,
            'num_functions': num_functions,
            'pad_token_id': pad_token_id,
            'bos_token_id': bos_token_id,
            'eos_token_id': eos_token_id
        }
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_length, dim))
        
        self.layers = nn.ModuleList([
            KANBlock(dim, num_functions) for _ in range(num_layers)
        ])
        
        self.output = nn.Linear(dim, vocab_size)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding[:, :L, :]
        
        for layer in self.layers:
            x = layer(x)
            
        logits = self.output(x)
        return logits
    
    def save_pretrained(self, path: str):
        """Save model weights and config."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, path)
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = 'cpu'):
        """Load model from saved weights."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model.to(device)

