from __future__ import annotations

import numpy as np
import numpy.typing as npt
import os
from sympy.printing.python import kw
import torch
from torch import Tensor
from jaxtyping import Float, Int


# uv run pytest -k test_get_batch
def get_batch(
    dataset: npt.NDArray | str, batch_size: int, context_length: int, device: str,
) -> tuple[Tensor, Tensor]:
    """
    Data loader.

    Args:
        dataset: Integer array containing token IDs.

    Returns:
        x: (batch_size, context_length), token IDs for sampled input sequences.
        y: (batch_size, context_length), token IDs for corresponding next-token targets.
    """
    if isinstance(dataset, str):  # Load memap from path
        dataset = np.load(dataset, mmap_mode='r')

    starting_idxs = torch.randint(len(dataset) - context_length, (batch_size,))
    
    x = torch.stack([
        torch.from_numpy(dataset[i : i + context_length].astype(np.int64))
        for i in starting_idxs
    ])

    y = torch.stack([
        torch.from_numpy(dataset[i+1 : i+1 + context_length].astype(np.int64))
        for i in starting_idxs
    ])

    if (hasattr(device, 'type') and device.type == "cuda") or "cuda" in str(device):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


class MemmapDataLoader():
    """
    Memory-efficient data loader using .npy files with memory mapping.
    """
    def __init__(self, dataset_path: str, logger = None) -> None:
        self.dataset_path = dataset_path
        self.dataset = None
        self._open_dataset()
        self.logger = logger
        return

    def _open_dataset(self):
        """
        Open the memory-mapped .npy dataset.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        self.dataset = np.load(self.dataset_path, mmap_mode='r')
        if self.logger is not None:
            self.logger.info(f"Loaded dataset: {len(self.dataset):,} tokens")
        return 

    def get_batch(self, **kwargs):
        return get_batch(self.dataset, **kwargs)

    def __del__(self):
        """
        Clean up memmap when done.
        """
        if hasattr(self.dataset) and self.dataset is not None:
            del self.dataset
