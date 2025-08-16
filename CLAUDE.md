# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is CS336 Assignment 1, focusing on implementing the fundamentals of transformer language models from scratch. The project includes BPE tokenization, transformer architecture components, training infrastructure, and evaluation systems.

## Key Commands

### Environment and Dependencies
```bash
# Install and manage dependencies
uv run <python_file_path>  # Run any Python file with automatic environment setup

# Run tests
uv run pytest                # Run all tests
uv run pytest tests/test_*.py  # Run specific test files
uv run pytest -v ./tests --junitxml=test_results.xml  # Verbose testing with XML output
```

### Linting and Code Quality
```bash
# Code formatting and linting (configured in pyproject.toml)
ruff check                   # Check code style
ruff format                  # Format code
```

### Training and Data Preparation
```bash
# Train models using configuration files
uv run python scripts/train.py --config configs/small_model.yaml
uv run python scripts/train.py --config configs/gpt2_medium.yaml

# Train with custom parameters
uv run python scripts/train.py --d-model 1024 --num-layers 24 --batch-size 16

# Prepare tokenized data for training
uv run python scripts/prepare_data.py --vocab vocab.json --merges merges.txt --input text_files/ --output data/

# Train BPE tokenizer
uv run python scripts/train_tokenizer.py
```

### Data Setup
```bash
# Download required datasets (run from project root)
mkdir -p data && cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz && gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz && gunzip owt_valid.txt.gz
cd ..
```

### Submission
```bash
./make_submission.sh  # Create submission zip file
```

## Architecture Overview

### Core Implementation Files
- **`cs336_basics/model.py`**: Transformer architecture components (Linear, Embedding, SwiGLU, MultiheadSelfAttention, TransformerBlock, BasicTransformerLM)
- **`cs336_basics/tokenizer.py`**: BPE tokenizer implementation
- **`cs336_basics/train_bpe.py`**: BPE training algorithm
- **`cs336_basics/optimizer.py`**: AdamW optimizer and learning rate scheduling
- **`cs336_basics/data.py`**: Data loading utilities (get_batch function)
- **`cs336_basics/utils/nn_utils.py`**: Neural network utilities (RMSNorm, activations, loss functions, gradient clipping)
- **`cs336_basics/utils/io.py`**: Checkpoint save/load utilities

### Training Infrastructure
- **`scripts/train.py`**: Main training script with comprehensive features (checkpointing, W&B logging, cosine LR scheduling)
- **`scripts/prepare_data.py`**: Data preparation and tokenization utilities
- **`configs/`**: YAML configuration files for different model sizes

### Test Integration
- **`tests/adapters.py`**: Critical adapter functions that connect your implementations to the test suite
- Tests use snapshot testing with reference implementations in `tests/_snapshots/`

## Key Implementation Details

### Model Architecture
- Pre-norm transformer blocks with RMSNorm
- Causal multi-head self-attention with RoPE (Rotary Position Embedding)
- SwiGLU feed-forward networks
- Embedding and linear projections

### Training Features
- Memory-efficient data loading with `np.memmap` for large datasets
- Automatic checkpointing and resume capability
- Cosine learning rate decay with linear warmup
- Gradient clipping for training stability
- W&B integration for experiment tracking

### Tokenizer
- Byte-Pair Encoding (BPE) implementation
- Special token handling
- Pretokenization with regex patterns
- Configurable vocabulary size

## Development Workflow

1. **Implement core functions** in the respective modules (model.py, tokenizer.py, etc.)
2. **Connect to tests** by completing functions in `tests/adapters.py`
3. **Run tests frequently** to verify correctness against reference implementations
4. **Use configuration files** for consistent training setups
5. **Monitor training** with console logs and W&B dashboard

## Configuration Management

Training configurations are managed through YAML files in `configs/`:
- `small_model.yaml`: 12-layer, 768-dimensional model
- `gpt2_medium.yaml`: 24-layer, 1024-dimensional model

Configuration precedence: Command line args > Config file > Defaults

## Data Format

- Training data: Space-separated token IDs or newline-separated IDs
- Tokenization: BPE with configurable special tokens
- Memory management: Automatic `memmap` for large files
- make sure to use uv to manage all dependencies