from dataclasses import dataclass
from typing import Optional

@dataclass
class KANEXConfig:
    vocab_size: int = 32000
    dim: int = 512
    depth: int = 6
    num_heads: int = 8
    max_seq_length: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    min_lr: float = 1e-5
    save_dir: str = "checkpoints"
    device: Optional[str] = None
