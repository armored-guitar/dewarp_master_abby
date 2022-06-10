import os
from pathlib import Path

import torch
from torch import nn


def load_everything(model: nn.Module, optimizer, scheduler, load_optim, load_path, device):
    checkpoint = torch.load(load_path, map_location=device)
    res = model.load_state_dict(checkpoint["model"])
    print(res)
    if hasattr(model.encoder, "segformer"):
        pos_usage = []
        for module in model.encoder.segformer.segformer.encoder.patch_embeddings:
            pos_usage.append(module.use_pos_encoding)
        print("pos_usage after loading:", pos_usage)
    if load_optim:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        print(optimizer.param_groups[0]['lr'])
        print(scheduler.state_dict())
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


