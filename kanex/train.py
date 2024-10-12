import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset

from model import KANTextGenerator  
def load_wikipedia_dataset():
    """
    Load and preprocess the Wikipedia dataset from Hugging Face.
    Returns a processed dataset with tokenized text.
    """
    dataset = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)

    def preprocess_data(example):
        return {'text': example['text']}

    dataset = dataset.map(preprocess_data, batched=True)
    return dataset


# Step 2: Create a Custom Dataset Class for Tokenized Sequences
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        """
        Custom dataset class to convert raw text into tokenized sequences.
        texts: List of text data.
        tokenizer: GPT2 tokenizer for tokenizing input text.
        max_length: Maximum token length for input sequences.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].squeeze()
        return input_ids

# Step 3: Initialize the Tokenizer and Dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to the EOS token to avoid mismatch

dataset = load_wikipedia_dataset()
texts = [item['text'] for item in dataset]  # Extract text from the dataset

# Convert the dataset to tokenized sequences
text_dataset = TextDataset(texts, tokenizer, max_length=128)
dataloader = DataLoader(text_dataset, batch_size=32, shuffle=True)

# Step 4: Initialize the Model, Optimizer, and Loss Function
model = KANTextGenerator(vocab_size=len(tokenizer), embed_dim=512, hidden_dim=1024, num_splines=5)
model = model.to('cuda')  # Move model to GPU if available

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Ignore padding in the loss calculation

# Step 5: Training Loop
def train(model, dataloader, optimizer, criterion, epochs=3):
    """
    Train the KANEX text generation model.
    model: KANEX model.
    dataloader: DataLoader object for loading batches of tokenized sequences.
    optimizer: Optimizer for the model (Adam).
    criterion: Loss function (CrossEntropyLoss).
    epochs: Number of training epochs.
    """
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, input_ids in enumerate(dataloader):
            input_ids = input_ids.to('cuda')

            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids[:, :-1])  # Predict the next token for each position
            targets = input_ids[:, 1:]  # Target is the next token in the sequence

            # Compute the loss
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}')

# Step 6: Train the Model
if __name__ == "__main__":
    train(model, dataloader, optimizer, criterion, epochs=3)
