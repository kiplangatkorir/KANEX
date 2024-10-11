import torch
import torch.nn as nn

class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(KANLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

# Example usage
kan_layer = KANLayer(10, 10)
input_tensor = torch.rand(1, 10)  # Random input
output = kan_layer(input_tensor)
print(output)
