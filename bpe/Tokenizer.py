import unicodedata

class Tokenizer:
    """
    Base class for tokenizers to inherit
    """
    def __init__(self) -> None:
        self.merges = {}    # (int, int) -> int 
        self.pattern = ''   # str
        self.special_tokens = {}    # for special tokens like {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    # fucntions to be overriden in the inherited class
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError 
    
    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (k0, k1), idx in self.merges.items():
            vocab[idx] = vocab[k0] + vocab[k1]
        for stoken, idx in self.special_tokens.items():
            vocab[idx] = stoken.encode('utf-8')
        return vocab 
    
    def _replace_control_characters(self, s: str) -> str:
        # we don't want to print control characters
        # which distort the output (e.g. \n or much worse)
        # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        # http://www.unicode.org/reports/tr44/#GC_Values_Table
        chars = []
        for ch in s:
            if unicodedata.category(ch)[0] != "C":
                chars.append(ch) # this character is ok
            else:
                chars.append(f"\\u{ord(ch):04x}") # escape
        return "".join(chars)
    
    def _render_token(self, t: bytes) -> str:
        # pretty print a token, escaping control characters
        s = t.decode('utf-8', errors='replace')
        s = self._replace_control_characters(s)
        return s

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges
            f.write("bpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = self._render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = self._render_token(self.vocab[idx0])
                    s1 = self._render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()