import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANEXLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
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
    def __init__(self, dim, max_len):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class KANEX(nn.Module):
    def __init__(self, vocab_size, dim, depth, heads, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_encoding = KANEXPositionEncoding(dim, max_len)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                KANEXAttention(dim, heads=heads),
                KANEXLayer(dim, dim, dim * 2)
            ]))
        
        self.to_logits = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.position_encoding(x)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.to_logits(x)

    def generate(self, start_tokens, max_len):
        self.eval()
        current_seq = start_tokens

        for _ in range(max_len - start_tokens.size(1)):
            logits = self(current_seq)
            next_token = torch.argmax(logits[:, -1], dim=-1).unsqueeze(1)
            current_seq = torch.cat((current_seq, next_token), dim=1)

        return current_seq