import torch
import torch.nn.functional as F
from typing import Optional, List
from ..model import KANEX
from ..data import KANEXTokenizer

class KANEXGenerator:
    def __init__(self, model: KANEX, tokenizer: KANEXTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        self.model.eval()
        input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids)
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                filtered_logits = top_p_filtering(next_token_logits, top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == self.tokenizer.tokenizer.eos_token_id:
                    break
        
        return self.tokenizer.decode(input_ids[0].tolist())

def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits