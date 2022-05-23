import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict

from torch.cuda.amp import autocast, GradScaler

from libs.postprocessing.dewarping import create_mapping
from libs.utils.utils import create_dir_for_file_if_needed


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int,
                writer: SummaryWriter, mixed: bool = False, scheduler=None):
    """
    Do one training epoch
    :param model: model
    :param train_loader: dataloader with train examples
    :param optimizer: optimizer
    :param epoch: current epoch
    :param writer: tensorboard writer
    """
    model.train()

    train_epoch_pbar = tqdm(train_loader, desc=f"training epoch {epoch}", leave=False)

    if mixed:
        scaler = GradScaler()
        for batch_idx, batch_data in enumerate(train_epoch_pbar):
            for param in model.parameters():
                param.grad = None
            with autocast():
                loss, log_dict = model.get_loss(batch_data)
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()

            train_epoch_pbar.set_postfix(log_dict)
            for name, value in log_dict.items():
                writer.add_scalar(f"step_{name}", value, batch_idx + len(train_loader) * epoch)
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step()
                writer.add_scalar("lr_step", optimizer.param_groups[0]['lr'], batch_idx + len(train_loader) * epoch)
    else:
        for batch_idx, batch_data in enumerate(train_epoch_pbar):
            for param in model.parameters():
                param.grad = None
            loss, log_dict = model.get_loss(batch_data)
            loss.backward()
            optimizer.step()
            train_epoch_pbar.set_postfix(log_dict)
            for name, value in log_dict.items():
                writer.add_scalar(f"step_{name}", value, batch_idx + len(train_loader) * epoch)
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step()
                writer.add_scalar("lr_step", optimizer.param_groups[0]['lr'], batch_idx + len(train_loader) * epoch)


def val_epoch(model: nn.Module, val_loader: DataLoader, epoch: int, writer: SummaryWriter,
              mixed: bool = False) -> Dict[str, float]:
    """
    Do one validation epoch
    :param model: model
    :param val_loader: dataloader with validation examples
    :param epoch: current epoch
    :param writer: tensorboard
    :return dict of losses and metrics and their values
    """
    model.eval()
    epoch_results = defaultdict(int)
    if mixed:
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(val_loader, desc=f"validation after epoch {epoch}",
                                                        leave=False)):
                with autocast():
                    loss, log_dict = model.get_loss(batch_data)
                for name, value in log_dict.items():
                    epoch_results[f"val_{name}"] += value
    else:
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(val_loader, desc=f"validation after epoch {epoch}",
                                                        leave=False)):
                loss, log_dict = model.get_loss(batch_data)
                for name, value in log_dict.items():
                    epoch_results[f"val_{name}"] += value

    for name, value in epoch_results.items():
        writer.add_scalar(name, float(value) / float(len(val_loader)), epoch)
        epoch_results[name] = float(value) / float(len(val_loader))
    return epoch_results


def test_flow_dewarping(model: nn.Module, val_loader: DataLoader, opt: DictConfig):
    """
    Inference for images
    :param model: model to inference
    :param val_loader: loader with examples
    :param opt: inference config
    """
    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    path = opt["dewarped_path"]
    dewarped_only_path = os.path.join(path, "dewarped_only/")
    compared_path = os.path.join(path, "compare/")
    create_dir_for_file_if_needed(dewarped_only_path)
    create_dir_for_file_if_needed(compared_path)
    model.eval()
    model.to(device)
    index = 0
    sigmoid = nn.Sigmoid()
    k = 1
    print(len(val_loader))
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_loader)):
            mod_input, img, img_name = batch_data[0].to(device), batch_data[1], batch_data[-1][0]
            k += 1
            pred_flow, pred_mask = model(mod_input)
            # img = img.data.cpu().numpy().transpose(0, 2, 3, 1)[0]
            pred_mask = sigmoid(pred_mask)
            pred_flow = pred_flow.data.cpu().numpy().transpose(0, 2, 3, 1)[0]
            pred_mask = pred_mask.data.round().int().cpu().numpy()[0]
            pred_mask = pred_mask.squeeze(0)
            pred_mask[pred_mask < 0.5] = 0
            img = img[0].cpu().numpy()
            dewarped = create_mapping(img, pred_flow, pred_mask)
            results = np.zeros((img.shape[0], img.shape[1] * 2 + 20, 3), dtype=np.uint)
            results[:, :img.shape[1], :] = img.astype(np.uint32)
            results[:, img.shape[1] + 20:, :] = dewarped

            cv2.imwrite(os.path.join(compared_path, img_name), results)
            cv2.imwrite(os.path.join(dewarped_only_path, img_name), dewarped)
            index += 1


def label_flow_dewarping(val_loader: DataLoader, dest_path: str):
    """
    Inference for images
    :param dest_path: str
    :type dest_path:
    :param val_loader: loader with examples
    :param opt: inference config
    """
    device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
    create_dir_for_file_if_needed(dest_path)

    index = 0
    print("without sigmoid")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_loader)):
            mod_input, pred_mask, pred_flow, img, name = batch_data[0].to(device), batch_data[1], batch_data[2], \
                                                         batch_data[3], batch_data[4]
            name = name[0]
            pred_flow = pred_flow.data.cpu().numpy().transpose(0, 2, 3, 1)[0]
            img = img.numpy()[0]

            pred_mask = pred_mask.data.round().int().cpu().numpy()[0]
            pred_mask[pred_mask < 0.5] = 0
            dewarped = create_mapping(img, pred_flow, pred_mask)
            results = np.zeros((img.shape[0], img.shape[1] * 2 + 20, 3), dtype=np.uint)
            results[:, :img.shape[1], :] = img.astype(np.uint32)
            results[:, img.shape[1] + 20:, :] = dewarped

            cv2.imwrite(os.path.join(dest_path, f"{name}_gt.jpg"), results)
            cv2.imwrite(os.path.join(dest_path, f"{name}.jpg"), dewarped)
            index += 1
