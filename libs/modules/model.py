from libs.modules.encoder_decoder import get_encoder_decoder_model


def get_model(opt):
    model_type = opt["type"]
    if model_type == "encoder_decoder":
        return get_encoder_decoder_model(opt)
    raise NotImplementedError
