import os
from pathlib import Path

import torch
from torch import nn


def load_everything(model: nn.Module, optimizer, scheduler, load_optim, load_path):
    checkpoint = torch.load(load_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if load_optim:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    return model, optimizer, scheduler


def get_name(path: str, name: str, dir_create: bool) -> str:
    try:
        dirs = os.listdir(path)
    except:
        Path(path).mkdir(exist_ok=True, parents=True)
        dirs = []
    if name not in dirs:
        save_name = name
        if dir_create:
            Path(os.path.join(path, name)).mkdir(parents=True)
    else:
        i = 1
        while True:
            save_name = f"{name}_{i}"
            if save_name not in dirs:
                if dir_create:
                    Path(os.path.join(path, save_name)).mkdir(parents=True)
                break
            i += 1
    return save_name


