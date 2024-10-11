import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import os
from tqdm import tqdm
from model import KANEX

class SimpleTokenizer:
    """Basic tokenizer for demonstration"""
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
    
    def encode(self, text: str) -> List[int]:
        # This is a placeholder - implement real tokenization
        return [self.bos_token_id] + [hash(c) % (self.vocab_size - 3) + 3 for c in text] + [self.eos_token_id]
    
    def decode(self, ids: List[int]) -> str:
        # This is a placeholder - implement real detokenization
        return "".join([chr((id % 26) + 97) for id in ids if id not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]])

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for text in texts:
            tokens = self.tokenizer.encode(text)[:max_length]
            if len(tokens) < max_length:
                tokens = tokens + [self.tokenizer.pad_token_id] * (max_length - len(tokens))
            self.examples.append(tokens)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx])

def train(
    model: KANEX,
    train_texts: List[str],
    val_texts: Optional[List[str]] = None,
    batch_size: int = 8,
    epochs: int = 3,
    learning_rate: float = 3e-4,
    max_grad_norm: float = 1.0,
    save_dir: str = "checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Train the KANEX model"""
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    
    # Initialize tokenizer and datasets
    tokenizer = SimpleTokenizer(vocab_size=model.config['vocab_size'])
    train_dataset = TextDataset(train_texts, tokenizer, model.config['max_seq_length'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_texts:
        val_dataset = TextDataset(val_texts, tokenizer, model.config['max_seq_length'])
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Calculate loss (shift predictions and targets)
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = batch[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        model.save_pretrained(os.path.join(save_dir, f"kanex_epoch_{epoch+1}.pt"))
        
        # Validation
        if val_texts:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    outputs = model(batch)
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = batch[..., 1:].contiguous()
                    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation loss: {avg_val_loss:.4f}")

if __name__ == "__main__":
    # Example usage
    model = KANEX(
        vocab_size=10000,
        max_seq_length=512,
        dim=256,
        num_layers=4
    )
    
    # Example training data
    train_texts = [
        "Hello world",
        "This is an example",
        "Training the KANEX model"
    ]
    
    train(
        model=model,
        train_texts=train_texts,
        batch_size=2,
        epochs=3,
        save_dir="checkpoints"
    )