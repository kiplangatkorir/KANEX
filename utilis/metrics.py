import torch
from typing import Dict, Any

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == labels).float().mean().item()
        return {
            'accuracy': accuracy
        }
