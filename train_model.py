from omegaconf import DictConfig

from libs.data.dataset_factory import get_loaders
from libs.modules.model import get_model
from libs.utils.utils import seed_everything
from libs.training.train import train
from libs.config import parse_config


def main(opt: DictConfig):
    seed_everything(42)
    model = get_model(opt["model"])

    train_dl, val_dl = get_loaders(opt["dataset"], "train")

    train(opt["training"], model, train_dl, val_dl)


if __name__ == "__main__":
    main(parse_config("train"))
