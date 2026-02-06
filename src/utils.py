from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import torch
import yaml

def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(device_cfg: str = "auto") -> str:
    if device_cfg != "auto":
        return device_cfg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def split_indices(n: int, train_ratio: float = 0.8, seed: int = 42):
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    cut = int(train_ratio * n)
    return idx[:cut], idx[cut:]