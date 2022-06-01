from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.optim import Adam, SGD, Optimizer, AdamW

from omegaconf import OmegaConf
from libs.modules.encoder_decoder import EncoderDecoder
from libs.modules.segformer import SegFormerEncoder, SegFormerDecoderCNN


def base_multiplier(epoch):
    if epoch < 10:
        return 1
    if 10 <= epoch < 20:
        return 0.5
    elif 20 <= epoch < 30:
        return 0.1
    elif 30 <= epoch < 40:
        return 0.05
    else:
        return 0.01


def transformer_multiplier(epoch):
    if epoch < 15:
        return 1
    if 15 <= epoch < 30:
        return 0.5
    elif 30 <= epoch < 45:
        return 0.1
    elif 45 <= epoch < 60:
        return 0.05
    else:
        return 0.01


def get_optimizer(opt, model) -> Optimizer:
    opt = OmegaConf.to_container(opt)
    name = opt.pop("name")
    if isinstance(model, EncoderDecoder) and isinstance(model.encoder, SegFormerEncoder) and isinstance(model.decoder, SegFormerDecoderCNN):
        lr = opt.pop("lr")
        print("set lr different")
        if name == "adam":
            return Adam([{"params": model.encoder.parameters(), "lr": lr[0]},
                         {"params": model.decoder.parameters(), "lr": lr[1]},
                         {"params": model.heads.parameters(), "lr": lr[1]}], **opt)
        elif name == "sgd":
            return SGD([{"params": model.encoder.parameters(), "lr": lr[0]},
                         {"params": model.decoder.parameters(), "lr": lr[1]},
                         {"params": model.heads.parameters(), "lr": lr[1]}], **opt)
        elif name == "adamw":
            return AdamW([{"params": model.encoder.parameters(), "lr": lr[0]},
                         {"params": model.decoder.parameters(), "lr": lr[1]},
                         {"params": model.heads.parameters(), "lr": lr[1]}], **opt)

    if isinstance(opt["lr"], list):
        opt["lr"] = opt["lr"][0]
    if name == "adam":
        return Adam(model.parameters(), **opt)
    if name == "adamw":
        return AdamW(model.parameters(), **opt)
    elif name == "sgd":
        return SGD(model.parameters(), **opt)


def get_scheduler(opt, optimizer):
    opt = OmegaConf.to_container(opt)
    name = opt.pop("name")
    if name == "reduce_on_plateu":
        print("rop")
        return ReduceLROnPlateau(optimizer, **opt)
    if name == "cosine_warm_restart":
        return CosineAnnealingWarmRestarts(optimizer, **opt)
    if name == "exponential":
        return ExponentialLR(optimizer, **opt)
    if name == "baseline":
        return LambdaLR(optimizer, lr_lambda=base_multiplier, last_epoch=-1, verbose=True)
    if name == "transformer":
        print("transformer_multiplier")
        return LambdaLR(optimizer, lr_lambda=transformer_multiplier, last_epoch=-1, verbose=False)
    print("None scheduler")
    return None
