import torch
import torch.nn.functional as F
from typing import Optional, List
import argparse
from kanex.model import KANEX
from kanex.train import SimpleTokenizer


def generate(
    model: KANEX,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """Generate text using the KANEX model"""
    model.eval()
    model = model.to(device)
    
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    # Generate tokens
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                values, indices = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < values[:, [-1]]] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Break if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Append next token to input
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    # Decode generated tokens
    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text using KANEX model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to start generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering parameter")
    
    args = parser.parse_args()
    
    # Load model
    model = KANEX.from_pretrained(args.model_path)
    tokenizer = SimpleTokenizer(vocab_size=model.config['vocab_size'])
    
    # Generate text
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()