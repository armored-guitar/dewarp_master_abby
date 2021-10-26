import os

import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from omegaconf import OmegaConf
import yaml

from libs.utils.checkpoint import get_name, load_everything
from libs.training.steps import train_epoch, val_epoch
from libs.training.utils import get_scheduler, get_optimizer
from libs.utils.utils import seed_everything, create_dir_for_file_if_needed, count_params


def train(opt: DictConfig, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader):
    """
    Train whole model
    :param opt: config for training process
    :param model: model to train
    :param train_dl: dataloader for training data
    :param val_dl: dataloader for validation data
    """
    seed_everything(42)
    start_epoch = opt.get("start_epoch", 0)
    end_epoch = opt["n_epoch"]

    optimizer = get_optimizer(opt["optimizer"], model)
    scheduler_config = opt.get("scheduler", None)
    scheduler = get_scheduler(scheduler_config, optimizer) if scheduler_config is not None else None

    if start_epoch != 0:
        name = opt["continue_name"]
        path = os.path.join(opt["checkpoint_dir"], name, f"{start_epoch-1}.pth")
        model, optimizer, scheduler = load_everything(model, optimizer, scheduler, opt["load_optim"], path)
        config_name = get_name(os.path.join(opt["base_dir"], opt["checkpoint_dir"]), "config", False)
    else:
        name = get_name(os.path.join(opt["base_dir"], opt["checkpoint_dir"]), opt["name"], True)
        config_name = "config"

    with open(os.path.join(opt["base_dir"], opt["checkpoint_dir"], name, f"{config_name}.yaml"), 'w') as f:
        yaml.dump(OmegaConf.to_container(opt), f, default_flow_style=False, sort_keys=False)

    writer = SummaryWriter(os.path.join(opt["base_dir"], opt["logs_dir"], name))
    writer.add_text(name, str(OmegaConf.to_object(opt)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    count_params(model)
    
    model.to(device)
    epoch_pbar = tqdm(range(start_epoch, end_epoch), desc="training epoch progress")

    for epoch in epoch_pbar:

        train_epoch(model, train_dl, optimizer, epoch, writer)
        val_log_dict = val_epoch(model, val_dl, epoch, writer)
        epoch_pbar.set_postfix(val_log_dict)

        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_log_dict["step_value"])
        elif scheduler is not None:
            scheduler.step()

        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            **val_log_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }, os.path.join(opt["base_dir"], opt["checkpoint_dir"], name, f"{epoch}.pth")
        )
