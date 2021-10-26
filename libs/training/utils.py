from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.optim import Adam, SGD, Optimizer

from omegaconf import OmegaConf


def base_multiplier(epoch):
    if 10 <= epoch < 20:
        return 0.5
    elif 20 <= epoch < 30:
        return 0.1
    elif 30 <= epoch < 40:
        return 0.05
    else:
        return 0.01


def get_optimizer(opt, model) -> Optimizer:
    opt = OmegaConf.to_container(opt)
    name = opt.pop("name")
    if name == "adam":
        return Adam(model.parameters(), **opt)
    elif name == "sgd":
        return SGD(model.parameters(), **opt)


def get_scheduler(opt, optimizer):
    opt = OmegaConf.to_container(opt)
    name = opt.pop("name")
    if name == "reduce_on_plateu":
        return ReduceLROnPlateau(optimizer, **opt)
    if name == "cosine_warm_restart":
        return CosineAnnealingWarmRestarts(optimizer, **opt)
    if name == "exponential":
        return ExponentialLR(optimizer, **opt)
    if name == "baseline":
        return LambdaLR(optimizer, lr_lambda=base_multiplier, last_epoch=-1)
