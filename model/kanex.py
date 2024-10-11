import torch
import torch.nn as nn
from .layers import KANLayer  # Import the KANLayer class

class SimpleKANEXModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SimpleKANEXModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.kan_layer = KANLayer(embed_dim, embed_dim)  # Ensure KANLayer is defined here
        self.fc = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.kan_layer(x)
        x = self.fc(x)
        return x

# Example usage
if __name__ == "__main__":
    model = SimpleKANEXModel(vocab_size=27, embed_dim=64)
    input_tokens = torch.tensor([1, 2, 3, 4])  # Example input
    output = model(input_tokens)
    print(output.shape)  # Expected output: (batch_size, vocab_size)
