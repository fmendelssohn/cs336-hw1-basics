#!/usr/bin/env python3
"""
Script to train a BPE tokenizer using the existing train_bpe function.
"""

import fire
import json
import os
from pathlib import Path

# Add the parent directory to the path so we can import cs336_basics
import sys
sys.path.append(str(Path(__file__).parent.parent))

from cs336_basics.train_bpe import train_bpe


def main(
    input_file: str = "data/owt_train.txt",
    vocab_size: int = 50000,
    output_dir: str = "tokenizer",
    sample_size: int = 10000000,  # 10MB sample
    special_tokens: list = None
):
    """
    Train a BPE tokenizer on a sample of the input file.
    
    Args:
        input_file: Path to input text file
        vocab_size: Size of vocabulary to learn
        output_dir: Directory to save tokenizer files
        sample_size: Number of characters to sample from the file
        special_tokens: List of special tokens to add
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a temporary sample file
    sample_file = os.path.join(output_dir, "sample.txt")
    
    print(f"Creating sample file from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as infile:
        sample_text = infile.read(sample_size)
    
    with open(sample_file, 'w', encoding='utf-8') as outfile:
        outfile.write(sample_text)
    
    print(f"Sample file created: {len(sample_text):,} characters")
    
    # Train BPE tokenizer
    print(f"Training BPE tokenizer with vocab_size={vocab_size}...")
    vocab, merges = train_bpe(
        input_path=sample_file,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        progress_bar=True,
        num_workers=1
    )
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, "vocab.json")
    vocab_dict = {str(k): v.decode('utf-8', errors='ignore') if isinstance(v, bytes) else v 
                  for k, v in vocab.items()}
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
    
    # Save merges
    merges_path = os.path.join(output_dir, "merges.txt")
    with open(merges_path, 'w', encoding='utf-8') as f:
        for merge in merges:
            f.write(' '.join(b.decode('utf-8', errors='ignore') if isinstance(b, bytes) else str(b) 
                            for b in merge) + '\n')
    
    print(f"Tokenizer saved to {output_dir}/")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    # Clean up sample file
    os.remove(sample_file)
    
    return vocab_path, merges_path


if __name__ == '__main__':
    fire.Fire(main) 