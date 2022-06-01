import torch.nn

from libs.modules.encoder_decoder import get_encoder_decoder_model
from libs.modules.classification_model import get_classification_model


def get_model(opt):
    model_type = opt["type"]
    if model_type == "encoder_decoder":
        model_dtype = opt.get("dtype", None)
        if model_dtype is None:
            return get_encoder_decoder_model(opt)
        elif model_dtype == "half":
            model = get_encoder_decoder_model(opt).half()
            for name, layer in model.named_modules():
                if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.GroupNorm) or isinstance(layer, torch.nn.InstanceNorm2d):
                    layer.eps = 1e-4
            model.loss.float()
            return model
    elif model_type == "classification":
        return get_classification_model(opt)
    raise NotImplementedError
