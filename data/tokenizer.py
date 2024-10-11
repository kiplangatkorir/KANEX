import torch
from collections import Counter
import re

class KANEXTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        self.vocab = None
        self.token_to_id = None
        self.id_to_token = None

    def fit(self, texts):
        words = [word.lower() for text in texts for word in re.findall(r'\w+|\S', text)]
        word_counts = Counter(words)
        sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
        vocab = list(self.special_tokens.keys()) + sorted_words[:(self.vocab_size - len(self.special_tokens))]
        
        self.vocab = vocab
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, text):
        words = re.findall(r'\w+|\S', text.lower())
        return [self.token_to_id.get(word, self.special_tokens['<UNK>']) for word in words]

    def decode(self, ids):
        return ' '.join([self.id_to_token.get(id, '<UNK>') for id in ids])

    def get_vocab_size(self):
        return len(self.vocab) if self.vocab else self.vocab_size

    def save(self, path):
        torch.save({
            'vocab': self.vocab,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'special_tokens': self.special_tokens
        }, path)

    def load(self, path):
        data = torch.load(path)
        self.vocab = data['vocab']
        self.token_to_id = data['token_to_id']
        self.id_to_token = data['id_to_token']
        self.special_tokens = data['special_tokens']