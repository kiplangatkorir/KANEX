# test_model.py

import torch
from data.tokenizer import SimpleTokenizer
from model.kanex import SimpleKANEX

def main():
    # Initialize tokenizer and model
    tokenizer = SimpleTokenizer()
    model = SimpleKANEX(tokenizer.vocab_size)

    # Test text generation
    prompt = "Hello"
    start_tokens = torch.tensor(tokenizer.encode(prompt))
    generated_tokens = model.generate(start_tokens, max_len=20)
    generated_text = tokenizer.decode(generated_tokens.tolist())

    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()