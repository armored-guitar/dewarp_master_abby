import torch.nn as nn
import torch
from typing import Tuple
from transformers import SegformerForImageClassification, SegformerConfig

from libs.losses.losses import get_loss


class ClassificationSegformerModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.use_xca = opt["encoder"].get("use_xca", False)
        self.dummy_param = nn.Parameter(torch.empty(0))

        self.use_pos_encoding = opt["encoder"].get("use_pos_encoding", False)
        self.use_first_pos_only = opt["encoder"].get("use_first_pos_only", False)
        self.use_xca = opt["encoder"].get("use_xca", False)
        configuration = SegformerConfig(attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1,
                                        patch_sizes=[7, 3, 3, 3], classifier_dropout_prob=0.1,
                                        num_labels=3, reshape_last_stage=True,
                                        hidden_sizes=[32, 64, 160, 256], depths=[2, 2, 2, 2],
                                        decoder_hidden_size=256)
        self.segformer = SegformerForImageClassification.from_pretrained("nvidia/mit-b0", use_pos_encoding=self.use_pos_encoding, use_first_pos_only=self.use_first_pos_only, use_xca=self.use_xca)

        pos_usage = []
        mult = True
        for i, module in enumerate(self.segformer.segformer.encoder.patch_embeddings):
            module.use_pos_encoding = self.use_pos_encoding & mult
            if i == 0 and self.use_first_pos_only:
                mult = False
        for module in self.segformer.segformer.encoder.patch_embeddings:
            pos_usage.append(module.use_pos_encoding)
        print("pos_usage:", pos_usage)
        self.loss = get_loss(opt["loss"])

    def forward(self, x):
        return self.segformer(x).logits

    def prepare_input(self, batch_data: Tuple[torch.Tensor]):
        output = tuple(el.to(self.dummy_param.device).to(self.dummy_param.dtype) for el in batch_data)
        return output

    def get_loss(self, batch_data):

        batch_data = self.prepare_input(batch_data)

        batch_output = self.forward(batch_data[0])
        loss, log_dict = self.loss.calculate_loss(batch_data, batch_output)

        return loss, log_dict


def get_classification_model(opt):
    return ClassificationSegformerModel(opt)
