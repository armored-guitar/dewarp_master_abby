import os

import torch

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from libs.training.steps import test_flow_dewarping
from libs.data.dataset_factory import get_loaders
from libs.modules.model import get_model
from libs.utils.utils import seed_everything
from libs.config import parse_config


def main(opt: DictConfig):
    seed_everything(42)
    model = get_model(opt["model"])

    val_dl = get_loaders(opt["dataset"], "val")

    checkpoint = torch.load(os.path.join(opt["inference"]["path"], f"{opt['inference']['epoch']}.pth"))
    model.load_state_dict(checkpoint['model'])
    test_flow_dewarping(model, val_dl, opt["inference"])


if __name__ == "__main__":
    main(parse_config("inference"))
