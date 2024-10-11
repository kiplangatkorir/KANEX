class SimpleTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def build_vocab(self):
        chars = list("abcdefghijklmnopqrstuvwxyz ")
        vocab = {char: idx for idx, char in enumerate(chars)}
        return vocab

    def encode(self, text):
        return [self.vocab[char] for char in text.lower() if char in self.vocab]

    def decode(self, tokens):
        return ''.join([self.inv_vocab[token] for token in tokens])

# Example usage
tokenizer = SimpleTokenizer()
tokens = tokenizer.encode("hello world")
print(tokens)
text = tokenizer.decode(tokens)
print(text)
