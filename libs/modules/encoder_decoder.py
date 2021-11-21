from omegaconf import OmegaConf
import abc
from typing import Tuple

from libs.modules.baseline import *
from libs.losses.losses import get_loss
from libs.modules.mhsa_bottleneck import BottleStack


def get_decoder(opt):
    name = opt["name"]
    if name == "baseline_decoder":
        return BaselineDecoder(opt)
    raise NotImplementedError

def get_encoder(opt):
    name = opt["name"]
    if name == "baseline_encoder":
        return BaselineEncoder(opt)
    raise NotImplementedError


def get_heads(opt):
    name = opt["name"]
    if name == "baseline_head":
        return BaselineHeads(opt)
    raise NotImplementedError


def get_bottleneck(opt):
    name = opt["name"]
    if name == "baseline_bottleneck":
        return BaselineBottleneck(opt)
    if name == "botnet_bottleneck":
        return BottleStack(opt)
    raise NotImplementedError


class EncoderDecoder(nn.Module, abc.ABC):
    def __init__(self, opt):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.encoder = get_encoder(opt["encoder"])
        self.decoder = get_decoder(opt["decoder"])
        self.heads = get_heads(opt["heads"])
        bottleneck_config = opt.get("bottleneck", None)
        self.loss = get_loss(opt["loss"])
        if bottleneck_config is not None:
            self.bottleneck = get_bottleneck(bottleneck_config)
        else:
            self.bottleneck = None

    def forward(self, x):
        x = self.encoder(x)

        if self.bottleneck is not None:
            bottleneck = self.bottleneck(x[-1])
        else:
            bottleneck = x[-1]
        x = self.decoder(bottleneck, x)
        x = self.heads(x)
        return x

    @abc.abstractmethod
    def prepare_input(self, batch_data: Tuple[torch.Tensor]):
        raise NotImplementedError

    def get_loss(self, batch_data):

        batch_data = self.prepare_input(batch_data)

        batch_output = self.forward(batch_data[0])
        loss, log_dict = self.loss.calculate_loss(batch_data, batch_output)

        return loss, log_dict


class EncoderDecoderMask2dMap(EncoderDecoder):
    def prepare_input(self, batch_data: Tuple[torch.Tensor]):
        output = tuple(el.to(self.dummy_param.device) for el in batch_data)
        return output


def get_encoder_decoder_model(opt):
    opt = OmegaConf.to_container(opt)
    sub_type = opt["sub_type"]
    if sub_type == "mask_2d_map":
        nun_filters = opt["encoder"]["nun_filters"]
        n_classes = opt["encoder"]["n_classes"]

        opt["heads"]["n_classes"] = n_classes
        opt["heads"]["nun_filters"] = nun_filters

        opt["decoder"]["nun_filters"] = nun_filters

        opt["bottleneck"]["nun_filters"] = nun_filters
        return EncoderDecoderMask2dMap(opt)
