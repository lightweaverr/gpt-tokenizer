"""
Minimal (byte-level) Byte Pair Encoding tokenizer.
Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .Tokenizer import Tokenizer
from .helpers import get_pair_counts, merge

class BasicTokenizerBPE(Tokenizer):
    def __init__(self) -> None:
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        stats = {}
        for i in range(num_merges):
            stats = get_pair_counts(ids)

            pair = max(stats, key=stats.get)

            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode('utf-8', errors='replace') # we use errors='replace' to handle invalid tokens
        return text

    def encode(self, text):
        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_pair_counts(ids)
            # gets the pair with least merges index
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))

            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids