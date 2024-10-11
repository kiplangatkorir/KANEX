import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from tokenizer import Tokenizer

class WikipediaDataset(Dataset):
    def __init__(self, corpus: List[str], tokenizer: Tokenizer, max_length: int = 512):
        """
        Args:
            corpus (List[str]): List of Wikipedia text data (as sentences or paragraphs).
            tokenizer (Tokenizer): Tokenizer to convert text to token IDs.
            max_length (int): Maximum sequence length for each text sample.
        """
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Token IDs and attention mask for the sample.
        """
        text = self.corpus[idx]
        token_ids = self.tokenizer.encode(text)

        # Pad or truncate to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids += [self.tokenizer.token_to_id["<PAD>"]] * (self.max_length - len(token_ids))

        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [1 if token_id != self.tokenizer.token_to_id["<PAD>"] else 0 for token_id in token_ids]

        return torch.tensor(token_ids), torch.tensor(attention_mask)

def get_dataloader(corpus: List[str], tokenizer: Tokenizer, batch_size: int = 32, max_length: int = 512):
    """
    Returns a DataLoader for the Wikipedia dataset.
    """
    dataset = WikipediaDataset(corpus, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
