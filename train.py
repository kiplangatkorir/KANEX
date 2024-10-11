import torch
from pathlib import Path
from kanex.data import KANEXTokenizer, KANEXDataset
from kanex.model import KANEX
from kanex.training import KANEXTrainer
from kanex.config import KANEXConfig

def main():
    config = KANEXConfig()
    
    tokenizer = KANEXTokenizer(config.vocab_size)
    model = KANEX(
        vocab_size=config.vocab_size,
        dim=config.dim,
        depth=config.depth,
        num_heads=config.num_heads
    )
    
    # Add your data loading logic here
    train_dataset = KANEXDataset(...)
    val_dataset = KANEXDataset(...)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    trainer = KANEXTrainer(model, train_loader, val_loader, vars(config))
    trainer.train()

if __name__ == "__main__":
    main()