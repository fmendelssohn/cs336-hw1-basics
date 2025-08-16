# Training Script Usage Guide

This guide explains how to use the comprehensive training script for transformer language models.

## Features

- **Configurable hyperparameters**: All model and training parameters can be customized
- **Memory-efficient data loading**: Uses `np.memmap` for large datasets that don't fit in RAM
- **Automatic checkpointing**: Saves model state periodically and supports resuming training
- **Comprehensive logging**: Console logging and Weights & Biases integration
- **Learning rate scheduling**: Cosine decay with warmup
- **Gradient clipping**: Prevents exploding gradients
- **Validation**: Periodic evaluation on validation data

## Quick Start

### 1. Basic Training

```bash
# Train with default settings
uv run python scripts/train.py

# Train with custom hyperparameters
uv run python scripts/train.py \
    --d-model 1024 \
    --num-layers 24 \
    --batch-size 16 \
    --learning-rate 6e-5
```

### 2. Using Configuration Files

```bash
# Train using a pre-defined configuration
uv run python scripts/train.py --config configs/small_model.yaml

# Train using GPT-2 Medium configuration
uv run python scripts/train.py --config configs/gpt2_medium.yaml
```

### 3. Custom Configuration

Create your own configuration file:

```yaml
# Model hyperparameters
vocab_size: 50257
context_length: 1024
d_model: 768
num_layers: 12
num_heads: 12
d_ff: 3072

# Training hyperparameters
batch_size: 32
learning_rate: 1.0e-4
max_iters: 100000

# Data paths
train_data_path: data/your_data.txt
val_data_path: data/your_val_data.txt
```

## Command Line Arguments

### Model Hyperparameters
- `--vocab_size`: Vocabulary size (default: 50257)
- `--context_length`: Maximum sequence length (default: 1024)
- `--d_model`: Model dimension (default: 768)
- `--num_layers`: Number of transformer layers (default: 12)
- `--num_heads`: Number of attention heads (default: 12)
- `--d_ff`: Feed-forward dimension (default: 3072)
- `--rope_theta`: RoPE theta parameter (default: 10000.0)

### Training Hyperparameters
- `--batch_size`: Batch size (default: 32)
- `--max_iters`: Maximum training iterations (default: 100000)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 0.1)
- `--grad_clip`: Gradient clipping norm (default: 1.0)

### Learning Rate Schedule
- `--warmup_iters`: Warmup iterations (default: 1000)
- `--lr_decay_iters`: Learning rate decay iterations (default: 100000)
- `--min_lr`: Minimum learning rate (default: 1e-5)

### Data and Logging
- `--train_data_path`: Training data file path (default: data/owt_train.txt)
- `--val_data_path`: Validation data file path (default: data/owt_valid.txt)
- `--checkpoint_dir`: Checkpoint directory (default: checkpoints)
- `--checkpoint_interval`: Checkpoint frequency (default: 1000)
- `--eval_interval`: Evaluation frequency (default: 100)
- `--log_interval`: Logging frequency (default: 10)

### Weights & Biases
- `--wandb_project`: W&B project name (default: cs336-transformer)
- `--wandb_name`: W&B run name
- `--wandb_entity`: W&B entity
- `--no_wandb`: Disable W&B logging

### Other
- `--device`: Device to use (auto, cuda, cpu) (default: auto)
- `--seed`: Random seed (default: 42)

## Data Format

The training script expects tokenized text data in the following format:

- **Input**: Text files with one token ID per line, or space-separated token IDs
- **Tokenization**: Use your BPE tokenizer to convert text to token IDs
- **Memory**: Large datasets are automatically loaded using `np.memmap`

### Example Data Preparation

```python
from cs336_basics.tokenizer import Tokenizer

# Load your tokenizer
tokenizer = Tokenizer.from_files("vocab.json", "merges.txt")

# Tokenize text and save as space-separated IDs
with open("text.txt", "r") as f:
    text = f.read()

tokens = tokenizer.encode(text)
with open("data/tokens.txt", "w") as f:
    f.write(" ".join(map(str, tokens)))
```

### Using the Data Preparation Script

For convenience, you can use the provided data preparation script:

```bash
# Prepare tokenized data for training
uv run python scripts/prepare_data.py \
    --vocab vocab.json \
    --merges merges.txt \
    --input text_files/ \
    --output data/ \
    --create-stats

# Process a single file
uv run python scripts/prepare_data.py \
    --vocab vocab.json \
    --merges merges.txt \
    --input my_text.txt \
    --output data/
```

## Training Process

### 1. Initialization
- Load configuration (file or command line)
- Initialize model, optimizer, and data loaders
- Set up logging and checkpointing

### 2. Training Loop
- **Forward pass**: Generate logits from input tokens
- **Loss computation**: Cross-entropy loss with targets
- **Backward pass**: Compute gradients
- **Gradient clipping**: Prevent exploding gradients
- **Optimizer step**: Update model parameters
- **Learning rate scheduling**: Cosine decay with warmup

### 3. Monitoring
- **Console logging**: Training loss, learning rate, tokens/sec
- **W&B logging**: All metrics and hyperparameters
- **Validation**: Periodic evaluation on validation data
- **Checkpointing**: Save model state for resuming

### 4. Checkpointing
- **Periodic saves**: Every `checkpoint_interval` iterations
- **Latest checkpoint**: Always available as `latest.pt`
- **Resume training**: Automatically loads latest checkpoint

## Memory Management

### Large Datasets
- Uses `np.memmap` for datasets larger than available RAM
- Only loads data chunks as needed
- Automatically handles device transfers

### GPU Memory
- Automatic device detection and transfer
- Pin memory for faster CPU→GPU transfers
- Non-blocking transfers for better throughput

## Example Training Runs

### Small Model (12 layers, 768d)
```bash
uv run python scripts/train.py \
    --config configs/small_model.yaml \
    --max_iters 50000 \
    --wandb_name "small-model-test"
```

### Medium Model (24 layers, 1024d)
```bash
uv run python scripts/train.py \
    --config configs/gpt2_medium.yaml \
    --batch_size 8 \
    --grad_clip 0.5
```

### Custom Model
```bash
uv run python scripts/train.py \
    --d-model 512 \
    --num-layers 6 \
    --num-heads 8 \
    --batch-size 64 \
    --learning-rate 2e-4 \
    --max-iters 25000 \
    --train-data-path "data/my_data.txt" \
    --no-wandb
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Reduce context length
   - Use gradient accumulation (not yet implemented)

2. **Training Diverges**
   - Reduce learning rate
   - Increase gradient clipping
   - Check data quality

3. **Slow Training**
   - Use GPU if available
   - Reduce evaluation frequency
   - Use larger batch sizes

4. **Checkpoint Issues**
   - Ensure checkpoint directory exists
   - Check disk space
   - Verify file permissions

### Debug Mode

```bash
# Run with debug logging
uv run python scripts/train.py --log-interval 1 --eval-interval 10
```

## Integration with Existing Code

The training script uses all the modules you've implemented:

- **Model**: `BasicTransformerLM` with your transformer implementation
- **Optimizer**: `AdamW` with cosine learning rate scheduling
- **Loss**: `cross_entropy` from your utils
- **Gradient clipping**: `clip_gradients` from your utils
- **Checkpointing**: `save_checkpoint` and `load_checkpoint` from your utils
- **Data loading**: Enhanced version of your `get_batch` function

## Next Steps

- **Gradient accumulation**: For larger effective batch sizes
- **Mixed precision**: FP16 training for memory efficiency
- **Distributed training**: Multi-GPU support
- **Data parallelism**: Shard data across multiple processes
- **Model parallelism**: Split large models across devices

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify your data format and paths
3. Ensure all dependencies are installed
4. Check GPU memory and availability 