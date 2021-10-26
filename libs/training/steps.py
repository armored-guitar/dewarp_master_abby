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

from libs.postprocessing.dewarping import create_mapping
from libs.utils.utils import create_dir_for_file_if_needed


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int,
                writer: SummaryWriter):
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

    for batch_idx, batch_data in enumerate(train_epoch_pbar):
        optimizer.zero_grad()
        loss, log_dict = model.get_loss(batch_data)
        loss.backward()
        optimizer.step()
        train_epoch_pbar.set_postfix(log_dict)
        for name, value in log_dict.items():
            writer.add_scalar(f"step_{name}", value, batch_idx + len(train_loader) * epoch)


def val_epoch(model: nn.Module, val_loader: DataLoader, epoch: int, writer: SummaryWriter) -> Dict[str, float]:
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
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_loader, desc=f"validation after epoch {epoch}", leave=False)):
            loss, log_dict = model.get_loss(batch_data)
            for name, value in log_dict.items():
                epoch_results[f"val_{name}"] += value

    for name, value in epoch_results.items():
        writer.add_scalar(name, float(value) / float(len(val_loader)), epoch)
        epoch_results[name] = float(value) / float(len(val_loader))
    return epoch_results


def test_flow_dewarping(model: nn.Module, val_loader: DataLoader, device, opt: DictConfig):
    """
    Inference for images
    :param model: model to inference
    :param val_loader: loader with examples
    :param opt: inference config
    """
    path = opt["inference"]["dewarped_path"]
    create_dir_for_file_if_needed(path)
    model.eval()
    model.to(device)
    index = 0
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_loader)):
            img = batch_data.to(device)
            pred_flow, pred_mask = model(img, is_softmax=True)
            img = img.data.cpu().numpy().transpose(0, 2, 3, 1)[0]
            pred_mask = sigmoid(pred_mask)
            pred_flow = pred_flow.data.cpu().numpy().transpose(0, 2, 3, 1)[0]
            pred_mask = pred_mask.data.round().int().cpu().numpy()[0]
            pred_mask = pred_mask.squeeze(0)
            pred_mask[pred_mask < 0.5] = 0

            dewarped = create_mapping(img, pred_flow, pred_mask)
            results = np.zeros((img.shape[0], img.shape[1] * 2 + 20, 3), dtype=np.uint)
            results[:, :img.shape[1], :] = img.astype(np.uint32)
            results[:, img.shape[1] + 20:, :] = dewarped

            cv2.imwrite(os.path.join(path, f"{index}.jpg"), results)
            index += 1
