import logging
import os
import pprint
import time
from os.path import join as pjoin
from typing import Dict

import fire
import numpy as np
from numpy import random as rand
import torch
import wandb
import yaml

from cs336_basics.data import MemmapDataLoader
from cs336_basics.model import BasicTransformerLM
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_basics.utils.nn_utils import cross_entropy, clip_gradients, save_checkpoint, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """
    """
    def __init__(self, **kwargs):
        # Model hyperparameters
        self.vocab_size = kwargs.get('vocab_size', 50257)
        self.context_length = kwargs.get('context_length', 1024)
        self.d_model = kwargs.get('d_model', 768)
        self.num_layers = kwargs.get('num_layers', 12)
        self.num_heads = kwargs.get('num_heads', 12)
        self.d_ff = kwargs.get('d_ff', 3072)
        self.rope_theta = kwargs.get('rope_theta', 10000.0)
        
        # Training hyperparameters
        self.batch_size = kwargs.get('batch_size', 32)
        self.max_iters = kwargs.get('max_iters', 100000)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 0.1)
        self.betas = kwargs.get('betas', (0.9, 0.95))
        self.eps = kwargs.get('eps', 1e-8)
        self.grad_clip = kwargs.get('grad_clip', 1.0)
        
        # Learning rate schedule
        self.warmup_iters = kwargs.get('warmup_iters', 1000)
        self.lr_decay_iters = kwargs.get('lr_decay_iters', 100000)
        self.min_lr = kwargs.get('min_lr', 1e-5)
        
        # Data and logging
        self.train_data_path = kwargs.get('train_data_path', 'data/owt_train.txt')
        self.val_data_path = kwargs.get('val_data_path', 'data/owt_valid.txt')
        self.ckpt_dir = kwargs.get('ckpt_dir', 'checkpoints')
        self.ckpt_interval = kwargs.get('ckpt_interval', 1000)
        self.eval_interval = kwargs.get('eval_interval', 100)
        self.log_interval = kwargs.get('log_interval', 10)
        
        # Weights & Biases
        self.wandb_project = kwargs.get('wandb_project', 'cs336-transformer')
        self.wandb_name = kwargs.get('wandb_name', None)
        self.wandb_entity = kwargs.get('wandb_entity', None)
        
        # Device
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Random seed
        self.seed = kwargs.get('seed', 42)

    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load config from YAML file."""
        with open(config_path, mode='r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def save(self, config_path: str):
        """Save config to YAML file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, mode='w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, indent=2)

    def __str__(self) -> str:
        return yaml.dump(self.__dict__, default_flow_style=False, indent=2)
    

def evaluate_model(
    model: torch.nn.Module, 
    val_loader: MemmapDataLoader, 
    config: Config, 
    num_eval_batches: int = 10,
) -> Dict[str, float]:
    """
    Evaluate model on validation data.
    """
    model.eval()
    total_loss = 0.
    total_tokens = 0

    with torch.no_grad():
        for _ in range(num_eval_batches):
            try: 
                x, y = val_loader.get_batch(config.batch_size, config.context_length, config.device)
                logits = model(x)
                loss = cross_entropy(logits, y)

                # Count tokens and accumulate loss
                num_tokens = x.numel()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
            except Exception as e:
                logger.warning(f"Error during evaluate_model: {e}")
                continue
            
    # Compute metrics
    loss_per_token = total_loss/total_tokens if total_tokens > 0 else np.Inf
    perplexity = np.exp(loss_per_token)
    return dict(val_loss=loss_per_token, val_perplexity=perplexity)


def train_model(config: Config):
    """
    Main training loop.
    """
    # Set random seed
    torch.manual_seed(config.seed)
    rand.seed(config.seed)

    # Initialize W&B
    if config.wandb_project:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            entity=config.wandb_entity,
            # config=config.__dict__,
            # resume=True,
        )

    # Create checkpoint directory
    os.makedirs(config.ckpt_dir, exist_ok=True)

    # Initialize data loaders
    logger.info("Initializing data loaders...")
    train_loader = MemmapDataLoader(config.train_data_path)
    val_loader = MemmapDataLoader(config.val_data_path)

    # Initialize model
    logger.info("Initializing model...")
    model = BasicTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    ).to(config.device)

    # model = torch.compile(model)

    # Initialize optimizer
    logger.info("Initializing optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    ).to(config.device)

    # Load checkpoint if exists
    start_iter = 0
    latest_ckpt_path_txt = pjoin(config.ckpt_dir, "latest_ckpt_path.txt") 
    if os.path.exists(latest_ckpt_path_txt):
        # Read latest ckpt_path from file
        with open(latest_ckpt_path_txt, mode='r') as f:
            ckpt_path = f.readline()
        logger.info(f"Loading checkpoint from {ckpt_path}")
        start_iter = load_checkpoint(ckpt_path, model, optimizer)
        logger.info(f"Resuming from iteration {start_iter}")

    # Training loop
    logger.info("Start training...")
    model.train()

    # Track running metrics; reset after every config.log_interval iterations
    running_loss = 0.
    running_tokens = 0
    start_time = time.time()

    for t in range(start_iter, config.max_iters):
        # Get learning rate for iter
        lr = get_cosine_lr(
            t, 
            config.learning_rate,
            config.min_lr,
            config.warmup_iters,
            config.lr_decay_iters,
        )

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get batch
        try:
            x, y = train_loader.get_batch(
                config.batch_size, 
                config.context_length, 
                config.device,
            )
        except Exception as e:
            logger.warning(f"Error getting batch for iteration {t}: {e}")
            continue

        # Forward pass
        logits = model(x)
        loss = cross_entropy(logits, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip > 0:
            clip_gradients(model.parameters(), config.grad_clip)
        optimizer.step()

        # Update running metrics
        num_tokens = x.numel()
        running_loss += loss.item() * num_tokens
        running_tokens += num_tokens

        # Logging
        if t % config.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = running_tokens/elapsed if elapsed > 0 else 0
            loss_per_token = running_loss/running_tokens if running_tokens > 0 else 0

            log_data = dict(
                iter=t,
                lr=lr,
                train_loss=loss_per_token,
                tokens_per_sec=tokens_per_sec,
                elapsed=elapsed,
            )
            logger.info(pprint.pformat(log_data))  # indent=4, compact=True
            if wandb is not None:
                wandb.log(log_data)

            # Reset running metrics
            running_loss = 0.
            running_tokens = 0
            start_time = time.time()

        # Evaluation
        if t % config.eval_interval == 0:
            logger.info("Running evaluation...")
            eval_metrics = evaluate_model(model, val_loader, config)

            logger.info(pprint.pformat(eval_metrics))
            if wandb.run is not None:
                wandb.log(dict(iter=t, **eval_metrics))
            
            model.train()  # Back to training mode

        # Checkpointing
        if t % config.ckpt_interval == 0:
            ckpt_path = pjoin(config.ckpt_dir, f"ckpt_{t:06d}.pt")
            save_checkpoint(model, optimizer, t, ckpt_path)

            # Update latest ckpt path
            with open(latest_ckpt_path_txt, mode='w') as f:
                f.write(ckpt_path)

            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Save final checkpoint
    final_ckpt_path = pjoin(config.ckpt_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, t, final_ckpt_path)
    logger.info(f"Training complete; final checkpoint saved: {final_ckpt_path}")
            
    # Close W&B
    if wandb.run is not None:
        wandb.finish()
        

def main(
    config_path: str = None,
    vocab_size: int = 50257,
    context_length: int = 1024,
    d_model: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    d_ff: int = 3072,
    rope_theta: float = 10000.0,
    batch_size: int = 32,
    max_iters: int = 100000,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    warmup_iters: int = 1000,
    lr_decay_iters: int = 100000,
    min_lr: float = 1e-5,
    train_data_path: str = 'data/owt_train.txt',
    val_data_path: str = 'data/owt_valid.txt',
    ckpt_dir: str = 'checkpoints',
    ckpt_interval: int = 1000,
    eval_interval: int = 100,
    log_interval: int = 10,
    wandb_project: str = 'cs336_hw1_basics',
    wandb_name: str = None,
    wandb_entity: str = None,
    no_wandb: bool = False,
    device: str = 'auto',
    seed: int = 42,
):
    """
    Main training function for the transformer language model.
    
    Args:
        config_path: Path to config YAML file
        vocab_size: Vocabulary size
        context_length: Context length
        d_model: Model dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        rope_theta: RoPE theta parameter
        batch_size: Batch size
        max_iters: Maximum iterations
        learning_rate: Learning rate
        weight_decay: Weight decay
        grad_clip: Gradient clipping norm
        warmup_iters: Warmup iterations
        lr_decay_iters: LR decay iterations
        min_lr: Minimum learning rate
        train_data_path: Training data path
        val_data_path: Validation data path
        ckpt_dir: Checkpoint directory
        ckpt_interval: Checkpoint interval
        eval_interval: Evaluation interval
        log_interval: Logging interval
        wandb_project: W&B project name
        wandb_name: W&B run name
        wandb_entity: W&B entity
        no_wandb: Disable W&B logging
        device: Device to use ('auto', 'cuda', or 'cpu')
        seed: Random seed
    """
    # Load config from file if provided
    if config_path:
        config_obj = Config.from_file(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        config_dict = {
            'vocab_size': vocab_size,
            'context_length': context_length,
            'd_model': d_model,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'rope_theta': rope_theta,
            'batch_size': batch_size,
            'max_iters': max_iters,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'grad_clip': grad_clip,
            'warmup_iters': warmup_iters,
            'lr_decay_iters': lr_decay_iters,
            'min_lr': min_lr,
            'train_data_path': train_data_path,
            'val_data_path': val_data_path,
            'ckpt_dir': ckpt_dir,
            'ckpt_interval': ckpt_interval,
            'eval_interval': eval_interval,
            'log_interval': log_interval,
            'wandb_project': wandb_project,
            'wandb_name': wandb_name,
            'wandb_entity': wandb_entity,
            'device': device,
            'seed': seed,
        }
        if device == 'auto':
            config_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        if wandb:
            config_dict['wandb_project'] = None
        config_obj = Config(**config_dict)

    # Save config
    config_path = pjoin(config_obj.ckpt_dir, 'config.yaml')
    config_obj.save(config_path)
    logger.info("Training config:")
    logger.info(str(config_obj))

    # Start training:
    try:
        train_model(config_obj)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error("Training failed with error: %s", e)
        raise


if __name__ == "__main__":
    fire.Fire(main)