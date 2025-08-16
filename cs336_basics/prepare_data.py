#!/usr/bin/env python3
"""
Utility script to prepare tokenized data for training.
Converts text files to .npy tokenized format suitable for the training script.
Output files are saved as .npy files containing np.int64 arrays that can be loaded with np.load(..., mmap_mode='r').
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Union

import fire
import numpy as np
from cs336_basics.tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def tokenize_file(
    tokenizer: Tokenizer,
    input_path: str,
    output_path: str,
    max_tokens: int = None,
    chunk_size: int = 1000000
) -> int:
    """
    Tokenize a text file and save as .npy array for memmap loading.
    
    Args:
        tokenizer: BPE tokenizer instance
        input_path: Path to input text file
        output_path: Path to output tokenized file (.npy extension will be added)
        max_tokens: Maximum number of tokens to process (None for all)
        chunk_size: Number of characters to process at once
    
    Returns:
        Total number of tokens processed
    """
    logger.info(f"Tokenizing {input_path} -> {output_path}")
    
    # Ensure output path has .npy extension 
    if not output_path.endswith('.npy'):
        output_path = output_path.replace('.txt', '.npy')
    
    total_tokens = 0
    processed_chars = 0
    all_tokens = []
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        while True:
            # Read chunk of text
            chunk = infile.read(chunk_size)
            if not chunk:
                break
            
            # Tokenize chunk
            tokens = tokenizer.encode(chunk)
            all_tokens.extend(tokens)
            
            total_tokens += len(tokens)
            processed_chars += len(chunk)
            
            # Check if we've reached max_tokens
            if max_tokens and total_tokens >= max_tokens:
                logger.info(f"Reached max_tokens limit: {max_tokens}")
                all_tokens = all_tokens[:max_tokens]
                total_tokens = max_tokens
                break
            
            # Log progress
            if total_tokens % 1000000 == 0:
                logger.info(f"Processed {total_tokens:,} tokens, {processed_chars:,} characters")
    
    # Convert to numpy array and save as .npy file
    tokens_array = np.array(all_tokens, dtype=np.int64)
    
    # Save as .npy file that can be loaded with np.load(..., mmap_mode='r')
    np.save(output_path, tokens_array)
    
    logger.info(f"Tokenization complete: {total_tokens:,} tokens saved to {output_path}")
    return total_tokens


def create_vocab_stats(tokenizer: Tokenizer, output_path: str):
    """Create vocabulary statistics file."""
    vocab_size = len(tokenizer.vocab)
    
    stats = {
        'vocab_size': vocab_size,
        'merges_count': len(tokenizer.merges),
        'vocab_sample': dict(list(tokenizer.vocab.items())[:100]),  # First 100 tokens
        'merges_sample': list(tokenizer.merges)[:100]  # First 100 merges
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Vocabulary stats saved to {output_path}")


def main(
    vocab: str,
    merges: str,
    input_path: str,
    output: str,
    max_tokens: int = None,
    chunk_size: int = 1000000,
    create_stats: bool = False
):
    """
    Prepare tokenized data for training.
    
    Args:
        vocab: Path to vocabulary JSON file
        merges: Path to merges TXT file
        input_path: Input text file or directory
        output: Output directory for tokenized files (.npy format)
        max_tokens: Maximum tokens per file
        chunk_size: Characters per chunk
        create_stats: Create vocabulary statistics
    """
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {vocab} and {merges}")
    try:
        tokenizer = Tokenizer.from_files(vocab, merges)
        logger.info(f"Tokenizer loaded: {len(tokenizer.vocab):,} vocab, {len(tokenizer.merges):,} merges")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    # Create output directory
    os.makedirs(output, exist_ok=True)
    
    # Create vocabulary stats if requested
    if create_stats:
        stats_path = os.path.join(output, 'vocab_stats.json')
        create_vocab_stats(tokenizer, stats_path)
    
    # Process input
    input_path_obj = Path(input_path)
    
    if input_path_obj.is_file():
        # Single file
        output_path = os.path.join(output, input_path_obj.stem + '_tokens.npy')
        tokenize_file(tokenizer, str(input_path_obj), output_path, max_tokens, chunk_size)
        
    elif input_path_obj.is_dir():
        # Directory of files
        text_files = list(input_path_obj.glob('*.txt')) + list(input_path_obj.glob('*.md'))
        
        if not text_files:
            raise RuntimeError(f"No .txt or .md files found in {input_path}")
        
        logger.info(f"Found {len(text_files)} text files to process")
        
        for text_file in text_files:
            output_path = os.path.join(output, text_file.stem + '_tokens.npy')
            try:
                tokenize_file(tokenizer, str(text_file), output_path, max_tokens, chunk_size)
            except Exception as e:
                logger.error(f"Failed to process {text_file}: {e}")
                continue
    else:
        raise RuntimeError(f"Input path does not exist: {input_path}")
    
    logger.info("Data preparation complete!")


if __name__ == '__main__':
    fire.Fire(main) 