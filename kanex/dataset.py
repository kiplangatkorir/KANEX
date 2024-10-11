# dataset.py
import torch
from datasets import load_dataset

class YourDataset(torch.utils.data.Dataset):
    def __init__(self, split: str = 'train', max_length: int = 512):
        self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the input text and tokenize it
        text = self.dataset[idx]['text']
        tokens = self.tokenize(text)

        # Filter out empty sequences
        if len(tokens) == 0:
            tokens = torch.zeros(self.max_length, dtype=torch.long)  # Use padding if needed
        else:
            tokens = torch.tensor(tokens)[:self.max_length]  # Truncate if necessary

        # Create labels (for training, this can be a shifted version of the input)
        labels = tokens.clone()

        # Ensure consistent length by padding
        if len(tokens) < self.max_length:
            padding = torch.zeros(self.max_length - len(tokens), dtype=torch.long)  # Create padding
            tokens = torch.cat((tokens, padding))  # Concatenate to ensure consistent length
            labels = torch.cat((labels, padding))  # Also pad labels for consistency

        return {'input_ids': tokens, 'labels': labels}  # For simplicity, use the same tokens as labels

    def tokenize(self, text):
        # Basic tokenization; customize this to fit your tokenizer
        return [ord(char) for char in text]  # Just a simple example
