import os

import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from omegaconf import OmegaConf
import yaml

from libs.utils.checkpoint import get_name, load_everything
from libs.training.steps import train_epoch, val_epoch, classification_epoch
from libs.training.utils import get_scheduler, get_optimizer
from libs.utils.utils import seed_everything, create_dir_for_file_if_needed, count_params


def train(opt: DictConfig, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader):
    """
    Train whole model
    :param training_opt: config for the whole script
    :param model: model to train
    :param train_dl: dataloader for training data
    :param val_dl: dataloader for validation data
    """
    training_opt = opt["training"]
    seed_everything(42)
    start_epoch = training_opt.get("start_epoch", 0)
    end_epoch = training_opt["n_epoch"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    freeze_encoder_until = training_opt.get("freeze_encoder_until", 0)

    optimizer = get_optimizer(training_opt["optimizer"], model)
    dtype = opt["model"].get("dtype", None)
    if dtype == "half":
        optimizer.eps = 1e-5
    scheduler_config = training_opt.get("scheduler", None)
    # scheduler_config = None
    scheduler = get_scheduler(scheduler_config, optimizer) if scheduler_config is not None else None

    mixed = training_opt.get("mixed_precision", False)

    if mixed:
        print("using mixed precision")
    eval_every = training_opt.get("eval_every", None)
    if start_epoch != 0:
        print("load_model")
        name = training_opt["continue_name"]
        path = os.path.join(training_opt["base_dir"], training_opt["checkpoint_dir"], name, f"{start_epoch - 1}.pth")
        model, optimizer, scheduler = load_everything(model, optimizer, scheduler, training_opt["load_optim"], path,
                                                      device)
        config_name = get_name(os.path.join(training_opt["base_dir"], training_opt["checkpoint_dir"]), "config", False)
        actual_lr = optimizer.param_groups[0]["lr"]
        for g in optimizer.param_groups:
            g['lr'] = actual_lr/10
    else:
        actual_lr = None
        name = get_name(os.path.join(training_opt["base_dir"], training_opt["checkpoint_dir"]), training_opt["name"], True)
        config_name = "config"

    with open(os.path.join(training_opt["base_dir"], training_opt["checkpoint_dir"], name, f"{config_name}.yaml"), 'w') as f:
        yaml.dump(OmegaConf.to_container(opt), f, default_flow_style=False, sort_keys=False)

    writer = SummaryWriter(os.path.join(training_opt["base_dir"], training_opt["logs_dir"], name))
    writer.add_text(name, str(OmegaConf.to_object(opt)))

    count_params(model)

    epoch_pbar = tqdm(range(start_epoch, end_epoch), desc="training epoch progress")

    if (freeze_encoder_until != 0) and (start_epoch < freeze_encoder_until):
        print("freeze encoder")
        for param in model.encoder.parameters():
            param.requires_grad = False

    for epoch in epoch_pbar:
        if freeze_encoder_until != 0 and epoch == freeze_encoder_until:
            for param in model.encoder.parameters():
                param.requires_grad = True
        if eval_every is None:
            train_epoch(model, train_dl, optimizer, epoch, writer, mixed, scheduler, actual_lr)
            val_log_dict = val_epoch(model, val_dl, epoch, writer, mixed)
            actual_lr = None
            epoch_pbar.set_postfix(val_log_dict)

            if (scheduler is not None) and isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_log_dict["step_value"])
            elif scheduler is not None and not isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()
                print("scheduler step")

            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                **val_log_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": "none" if scheduler is None else scheduler.state_dict()
            }, os.path.join(training_opt["base_dir"], training_opt["checkpoint_dir"], name, f"{epoch}.pth")
            )
        else:
            classification_epoch(model, train_dl, val_dl, optimizer, epoch, writer, training_opt, eval_every, name)
