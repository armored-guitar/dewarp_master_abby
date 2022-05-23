from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Any, Tuple, Union

from libs.data.dataset import BaselineDewarpDataset


def get_dataset(opt: DictConfig, transforms: Optional[Any]=None, train=True, return_img: bool = False, return_name=False, test=False):
    name = opt["type"]
    if name == "baseline":
        return BaselineDewarpDataset(opt["path"], transforms, train, return_img=return_img, return_name=return_name, test=test)


def get_loader(opt: DictConfig, transforms: Optional[Any]=None, train=True, test=False, return_img: bool = False, return_name=False):
    dataset = get_dataset(opt, transforms, train, return_img, return_name, test)
    opt = OmegaConf.to_container(opt)
    opt.pop("transforms", False)
    opt.pop("type")
    opt.pop("path")
    return DataLoader(dataset, **opt)


def get_loaders(opt: DictConfig, mode: str) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    if mode == "train":
        use_transforms = opt["train_dataset"].get("transforms", False)
        train_dataset = get_loader(opt["train_dataset"], transforms=use_transforms, train=True)
        val_dataset = get_loader(opt["val_dataset"], transforms=use_transforms, train=True)
        return train_dataset, val_dataset
    elif mode == "val":
        use_transforms = opt["train_dataset"].get("transforms", False)
        val_dataset = get_loader(opt["val_dataset"], transforms=use_transforms, train=False, test=True, return_img=True, return_name=True)
        return val_dataset
    elif mode == "labeling":
        use_transforms = opt["val_dataset"].get("transforms", False)
        val_dataset = get_loader(opt["val_dataset"], transforms=use_transforms, train=True, return_img=True)
        return val_dataset
