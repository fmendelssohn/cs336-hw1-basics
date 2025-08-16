import re
from typing import Dict, List, Tuple, Iterable
from tqdm import tqdm
from cs336_basics.utils.io import get_tokenizer_from_vocab_merges_path, GPT2_PRETOKENIZER_PATTERN


def get_byte_pairs(ids: Iterable[int]) -> Iterable[Tuple[int, int]]:
    """Return a set of pairs in int ids."""
    pairs = set()
    for pair in zip(ids, ids[1:]):
        pairs.add(pair)
    return pairs

def update(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Update the ids by merging the pairs."""
    new_ids = []
    i = 0
    while i < len(ids):
        curr_pair = tuple(ids[i:i+2])
        if curr_pair == pair:
            new_ids.append(new_id)
            i += 1
        else:
            new_ids.append(ids[i])
        i += 1
    return new_ids

def _fix_vocab(vocab_int_to_byte: Dict[int, bytes], vocab_byte_to_int: Dict[bytes, int]):
    """Make sure all 256 bytes are in the vocab.""" 
    for i in range(256):
        byte = bytes([i])
        if byte not in vocab_byte_to_int:
            vocab_byte_to_int[byte] = len(vocab_byte_to_int)
            vocab_int_to_byte[len(vocab_byte_to_int)] = byte
    return vocab_int_to_byte, vocab_byte_to_int
    

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab_int_to_byte, self.vocab_byte_to_int = _fix_vocab(vocab, {v: k for k, v in vocab.items()})

        # Reorganize merges into pair -> new token id dict
        self.merges = {}
        for a, b in merges:
            id_pair = (self.vocab_byte_to_int[a], self.vocab_byte_to_int[b])
            self.merges[id_pair] = self.vocab_byte_to_int[a + b] 
        
        # Add special tokens
        self.special_tokens = {}
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.vocab_byte_to_int:
                    self.special_tokens[token] = len(self.vocab_int_to_byte)
                    self.vocab_byte_to_int[token_bytes] = len(self.vocab_byte_to_int)
                    self.vocab_int_to_byte[len(self.vocab_int_to_byte)] = token_bytes  # TODO: Why not len(self.vocab_byte_to_int)
                else:
                    self.special_tokens[token] = self.vocab_byte_to_int[token_bytes]

        return

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None, **kwargs):
        vocab, merges = get_tokenizer_from_vocab_merges_path(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    @property
    def vocab_size(self):
        return len(self.vocab_int_to_byte)

    def _encode_chunk(self, text: str) -> List[int]:
        """Encode text without special tokens."""
        if text in self.special_tokens:
            return [self.special_tokens[text]]
        
        text_chunks = re.findall(GPT2_PRETOKENIZER_PATTERN, text)
        result = []
        for chunk in text_chunks:
            text_bytes = chunk.encode("utf-8")
            ids = [self.vocab_byte_to_int[bytes([b])] for b in chunk.encode('utf-8')]
            while len(ids) > 1:
                pairs = get_byte_pairs(ids)
                top_pair = min(pairs, key=lambda pair: self.merges.get(pair(pair), float('inf')))
                if top_pair not in self.merges:
                    break
                new_id = self.merges[top_pair]
                ids = update(ids, top_pair, new_id)
            result.extend(ids)
        return result
    
    def encode(self, text: str, progress_bar: bool = False) -> List[int]:
        """Encode text into a list of token ids."""
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            special_split_chunk = re.split(special_pattern, text)
        else:
            special_split_chunk = [text]
        ids = []
        for chunk in tqdm(special_split_chunk, disable=not progress_bar, 
                          desc=f"Encoding {len(special_split_chunk)} documents"):
            ids += self._encode_chunk(chunk)
        return ids

    def encode_iterable(self, texts: Iterable[str]) -> Iterable[List[int]]:
        """Encode the texts into a list of token ids; returns an iterator."""
        for text in texts:
            ids = self.encode(text)
            for id in ids:
                yield id

    def decode(self, ids: List[int]) -> str:
        """Decode the token ids into the original text."""
        text_bytes = b''.join([self.vocab_int_to_byte[i] for i in ids])
        return text_bytes.decode("utf-8", errors='replace')
