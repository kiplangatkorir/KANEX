import torch
from torch.utils.data import DataLoader
from dataset import YourDataset  # Import your dataset class
from model import KANEX  # Ensure the correct import of your model class

def main():
    # Load your data
    train_dataset = YourDataset(split='train')  # Load your training dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Set parameters
    vocab_size = 256  # Set your vocab size (for example)
    max_seq_length = 512  # Set your max sequence length

    # Create the model and ensure it stays on the CPU
    model = KANEX(vocab_size=vocab_size, max_seq_length=max_seq_length)

    # Define your optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    # Training loop
    model.train()
    num_epochs = 3  # Define your number of epochs
    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = batch['input_ids']  # Adjust this based on your dataset
            labels = batch['labels']  # Adjust this based on your dataset
            
            # Forward pass
            outputs = model(inputs)  # No need to move to CUDA
            
            # Compute loss (define your loss function)
            loss_function = torch.nn.CrossEntropyLoss()  # Example loss function
            loss = loss_function(outputs.view(-1, vocab_size), labels.view(-1))  # Example
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    main()
