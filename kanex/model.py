import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANEXLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # SiLU (Swish) activation for smooth, efficient computation
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.g = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.g(self.phi(x))

class KANEXAttention(nn.Module):
    def __init__(self, dim, heads=4, head_dim=32, window_size=16):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.to_qkv = KANEXLayer(dim, 3 * heads * head_dim, dim)
        self.to_out = nn.Linear(heads * head_dim, dim)

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, self.head_dim).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        dots = dots.masked_fill_(torch.ones_like(dots).triu_(diagonal=self.window_size) == 1, float('-inf'))
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class KANEXPositionEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, dim // 4))
        self.project = KANEXLayer(dim // 4, dim, dim // 2)

    def forward(self, x):
        b, n, _ = x.shape
        positions = self.project(self.pe[:, :n])
        return x + positions

class KANEX(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.position_encoding = KANEXPositionEncoding(config['d_model'], config['max_seq_length'])
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                KANEXAttention(config['d_model'], config['n_heads']),
                nn.LayerNorm(config['d_model']),
                KANEXLayer(config['d_model'], config['d_model'], config['d_ff']),
                nn.LayerNorm(config['d_model'])
            ) for _ in range(config['n_layers'])
        ])
        
        self.to_logits = nn.Linear(config['d_model'], config['vocab_size'])

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.position_encoding(x)
        
        for layer in self.layers:
            x = x + layer(x)
        
        return self.to_logits(x)

    def generate(self, start_tokens, max_new_tokens, temperature=1.0):
        self.eval()
        device = next(self.parameters()).device
        start_tokens = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)
        generated = start_tokens

        for _ in range(max_new_tokens):
            logits = self(generated)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated.squeeze(0).tolist()

# Example usage
if __name__ == "__main__":
    config = {
        'vocab_size': 256,
        'd_model': 256,
        'n_heads': 4,
        'n_layers': 6,
        'd_ff': 1024,
        'max_seq_length': 512
    }
    
    model = KANEX(config)
    x = torch.randint(0, 256, (2, 32))  # Batch size 2, sequence length 32
    output = model(x)
    print("Output shape:", output.shape)
    
    # Generate some tokens
    start_tokens = [65, 66, 67]  # ASCII for 'ABC'
    generated = model.generate(start_tokens, max_new_tokens=10)
    print("Generated tokens:", generated)