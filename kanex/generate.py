import torch
import torch.nn.functional as F
from typing import List, Tuple
import argparse
from model import KANEX
from train import SimpleTokenizer

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def generate_response(
    model: KANEX,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """Generate a response using the KANEX model"""
    model.eval()
    model = model.to(device)
    
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            for token_id in set(input_ids[0].tolist()):
                next_token_logits[0, token_id] /= repetition_penalty
            
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    response = tokenizer.decode(input_ids[0].tolist())
    return response.split(prompt, 1)[1].strip()  # Return only the generated part

def chat_loop(
    model: KANEX,
    tokenizer: SimpleTokenizer,
    max_history: int = 5,
    **generation_kwargs
):
    """Interactive chat loop with the KANEX model"""
    print("Welcome to the KANEX chat! Type 'quit' to exit.")
    conversation_history: List[Tuple[str, str]] = []
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        # Construct prompt from conversation history
        prompt = ""
        for turn in conversation_history[-max_history:]:
            prompt += f"Human: {turn[0]}\nAI: {turn[1]}\n"
        prompt += f"Human: {user_input}\nAI:"
        
        response = generate_response(model, tokenizer, prompt, **generation_kwargs)
        print(f"AI: {response}")
        
        # Update conversation history
        conversation_history.append((user_input, response))

def main():
    parser = argparse.ArgumentParser(description="Chat with KANEX model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering parameter")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p filtering parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")
    parser.add_argument("--max_history", type=int, default=5, help="Maximum number of conversation turns to consider")
    
    args = parser.parse_args()
    
    try:
        # Load model
        model = KANEX.from_pretrained(args.model_path)
        tokenizer = SimpleTokenizer(vocab_size=model.config['vocab_size'])
        
        # Start chat loop
        chat_loop(
            model=model,
            tokenizer=tokenizer,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_history=args.max_history
        )
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()