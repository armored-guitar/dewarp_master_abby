from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Any, Tuple, Union

from libs.data.dataset import BaselineDewarpDataset


def get_dataset(opt: DictConfig, transforms: Optional[Any]=None, train=True):
    name = opt["type"]
    if name == "baseline":
        return BaselineDewarpDataset(opt["path"], transforms, train)


def get_loader(opt:DictConfig, transforms: Optional[Any]=None, train=True):
    dataset = get_dataset(opt, transforms, train)
    opt = OmegaConf.to_container(opt)
    opt.pop("type")
    opt.pop("path")
    return DataLoader(dataset, **opt)


def get_loaders(opt: DictConfig, mode: str) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    if mode == "train":
        train_dataset = get_loader(opt["train_dataset"], train=True)
        val_dataset = get_loader(opt["val_dataset"], train=True)
        return train_dataset, val_dataset
    elif mode == "val":
        val_dataset = get_loader(opt["val_dataset"], train=False)
        return val_dataset
