# uv run pytest tests/test_train_bpe.py
from pickle import bytes_types
import regex as re
from typing import Iterable
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from cs336_basics.utils.io import GPT2_PRETOKENIZER_PATTERN

def _find_pretokens(text: str):
    """Find pretokens in the text."""
    logging.info(f"Pre-tokenizing the text of length {len(text)}")
    return Counter(re.findall(GPT2_PRETOKENIZER_PATTERN, text))

def _read_text_file(input_path: str, num_workers: int, special_tokens: Iterable[str]):
    """
    Read text file and return the text as pretoken frequency table.
    """
    # Read input file into memory all at once
    with open(input_path, "r") as file:
        text = file.read()

    # Remove special tokens
    for token in special_tokens:
        text = text.replace(token, "")
    
    logging.info("Initializing pretoken frequency table")
    if num_workers == 1:
        pretokens = _find_pretokens(text)
    else:
        chunk_size = len(text) // num_workers
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            pretokens = executor.map(_find_pretokens, text_chunks)
        pretokens = sum(pretokens, Counter())

    pretoken_freq = {tuple(bytes([b]) for b in pretoken.encode("utf-8")): freq 
                     for pretoken, freq in pretokens.items()}
    return pretoken_freq


def _update_byte_table(byte_tuple: Iterable[bytes], merge_loc: int):
    """Merge the byte tuple at the merge location."""
    assert len(byte_tuple) > 1
    prefix = byte_tuple[:merge_loc]
    tomerge = byte_tuple[merge_loc:merge_loc+2]
    suffix = byte_tuple[merge_loc+2:]
    new_byte_tuple = prefix + (b"".join(tomerge),) + suffix
    return new_byte_tuple, prefix, suffix


# uv run pytest -k test_train_bpe
def train_bpe(input_path: str, vocab_size: int, special_tokens: Iterable[str], progress_bar: bool = False, num_workers: int = 1):
    """
    Train a byte pair encoding tokenizer on the input text file.

    Args:
        input_path: Path to the input text file.
        vocab_size: Size of the vocabulary.
        special_tokens: List of special tokens to add to the vocabulary.

    Returns:
        Tuple of the learned vocab and merges.
    """
    # Initialize vocab with 256 bytes and special tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256+i] = token.encode('utf-8')

    pretoken_freq = _read_text_file(input_path, num_workers, special_tokens)

    logging.info("Initializing byte-pair frequency table")
    pair_freq = Counter()
    for pretokens, freq in tqdm(pretoken_freq.items(), disable=not progress_bar):
        for i in range(len(pretokens) - 1):
            pair = pretokens[i:i+2]
            pair_freq[pair] += freq

    logging.info("Perform BPE algorithm")
    pre_merge_vocab_size = len(vocab)
    num_merges = vocab_size - pre_merge_vocab_size  # Number of merge iterations
    pbar = tqdm(total=num_merges) if progress_bar else None
    merges = []
    while len(vocab) < vocab_size:
        # Find the most frequent pair, break ties lexicographically
        top_pair = max(pair_freq, key=lambda k: (pair_freq[k], k))
        merges.append(top_pair)

        # Update vocab
        new_id = max(vocab.keys()) + 1
        vocab[new_id] = b"".join(top_pair)

        # Update the pre-token frequency table and pair frequency table
        new_pretoken_freq = {}
        for pretokens, freq in pretoken_freq.items():
            i = 0
            while i < len(pretokens):
                pair = pretokens[i:i+2]
                if pair == top_pair:
                    pretokens, prefix, suffix = _update_byte_table(pretokens, i)

                    # Update pair-frequency table
                    if prefix:
                        pair_freq[(prefix[-1], vocab[new_id])] += freq
                        pair_freq[(prefix[-1], top_pair[0])] -= freq
                    if suffix:
                        pair_freq[(vocab[new_id], suffix[0])] += freq
                        pair_freq[(top_pair[1], suffix[0])] -= freq
                    pair_freq[top_pair] -= freq
                i += 1

            # Update pretoken_freq
            new_pretoken_freq[pretokens] = freq
        pretoken_freq = new_pretoken_freq
        pbar.update(len(vocab) - pre_merge_vocab_size - pbar.n) if progress_bar else None

    pbar.close() if progress_bar else None
    return vocab, merges
