import os
import random

import numpy as np
import torch

from pathlib import Path


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dir_for_file_if_needed(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    non_trainable_params = total_parameters - trainable
    print("=" * 60)
    print("Parameters summary")
    print(f"Trainable parameters     | {trainable:.1e}")
    print(f"Non-trainable parameters | {non_trainable_params:.1e}")
    print(f"Total parameters         | {total_parameters:.1e}")
    print("=" * 60)
