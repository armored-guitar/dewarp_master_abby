from torch import nn
import torch
import numpy as np

from libs.modules.blocks import *
from libs.modules.classification_model import ClassificationSegformerModel
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from transformers import SegformerForImageClassification, SegformerModel


class PositionalEncoding2D(nn.Module):
    def __init__(self, size, channels=8):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        x, y = size
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

        pos_x = torch.arange(x).type(self.inv_freq.type())
        pos_y = torch.arange(y).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)[:, :3].transpose(0, 1).unsqueeze(0).unsqueeze(3)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)[:, :3].transpose(0, 1).unsqueeze(0).unsqueeze(2)
        self.register_buffer("emb_x", emb_x)
        self.register_buffer("emb_y", emb_y)

    def forward(self):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        return self.emb_x, self.emb_y


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, size):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(size)

    def forward(self, use_sum=True):
        emb_x, emb_y = self.penc()
        return emb_x + emb_y if use_sum else torch.concat(
            (emb_x.repeat(1, 1, 1, emb_y.shape[3]), emb_y.repeat(1, 1, emb_x.shape[2], 1)), dim=1)


class PreEncoder(nn.Module):
    def __init__(self, size=(1024, 960), use_sum=True, use_concat=False):
        super().__init__()
        self.emb = PositionalEncodingPermute2D(size)
        self.use_concat = use_concat
        if use_concat:
            self.use_sum = use_sum
            self.fusion = nn.Conv2d(6 if use_sum else 9, 3, 3, 1, 1)
        else:
            self.scale = 0.2
            self.use_sum = True
            self.fusion = nn.Identity()
        print("use position encoding scale 0.2")

    def forward(self, x):
        embedding = self.emb(self.use_sum)
        if self.use_concat:
            x = torch.concat((x, embedding), dim=1)
        else:
            x = x + self.scale * embedding
        x = self.fusion(x)
        return x


class UpsamplingDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x, dummy):
        x = self.upsample(x)
        return x


class SegFormerSplitHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, :2], x[:, 2:]


class SegFormerEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.output_segformer = opt.get("output_segformer", 150)
        self.use_pos_encoding = opt.get("use_pos_encoding", False)
        self.use_first_pos_only = opt.get("use_first_pos_only", False)
        self.pretrained = opt.get("pretrained", False)
        self.use_default_decoder = opt.get("use_default_decoder", True)
        self.use_xca = opt.get("use_xca", False)
        configuration = SegformerConfig(attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1,
                                        patch_sizes=[7, 3, 3, 3], classifier_dropout_prob=0.1,
                                        num_labels=self.output_segformer, reshape_last_stage=True,
                                        hidden_sizes=[32, 64, 160, 256], depths=[2, 2, 2, 2],
                                        decoder_hidden_size=256)

        ## TODO test b1/ model
        ## TODO test b0 with larger lr and schedulers
        ## TODO test baseline with transformer_encoder
        # SegformerConfig(num_labels=150, reshape_last_stage=True, hidden_sizes=[64, 128, 320, 512])
        self.segformer = SegformerForSemanticSegmentation(configuration, use_pos_encoding=self.use_pos_encoding,
                                                          use_first_pos_only=self.use_first_pos_only,
                                                          use_xca=self.use_xca)

        if self.pretrained:
            if isinstance(self.pretrained, bool):
                encoder_pretrained = SegformerForImageClassification.from_pretrained("nvidia/mit-b0").segformer
                self.segformer.segformer = encoder_pretrained
                print("loaded mit model")
            elif isinstance(self.pretrained, str):
                checkpoint = torch.load(self.pretrained)["model"]

                encoder_pretrained = ClassificationSegformerModel({"encoder": opt, "loss": {"name": "ce"}})
                encoder_pretrained.load_state_dict(checkpoint, strict=True)
                encoder_pretrained = encoder_pretrained.segformer.segformer
                assert type(self.segformer.segformer) == type(encoder_pretrained)

                self.segformer.segformer = encoder_pretrained

                assert isinstance(self.segformer.segformer, SegformerModel)

            pos_usage = []
            mult = True
            for i, module in enumerate(self.segformer.segformer.encoder.patch_embeddings):
                module.use_pos_encoding = self.use_pos_encoding & mult
                if i == 0 and self.use_first_pos_only:
                    mult = False

            for module in self.segformer.segformer.encoder.patch_embeddings:
                pos_usage.append(module.use_pos_encoding)
            print("pos_usage:", pos_usage)
        if not self.use_default_decoder:
            self.segformer = self.segformer.segformer

    def forward(self, x):
        segformer_output = self.segformer(x, output_hidden_states=True)
        if self.use_default_decoder:
            return list(segformer_output.hidden_states) + [segformer_output.logits]
        else:
            return segformer_output.hidden_states


class SegFormerDecoderCNN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        BatchNorm = get_batch_norm(opt["batch_norm"])
        GN_num = 32
        act_fn = nn.ReLU(inplace=True)
        self.input_dim = opt.get("input_dim", 150)
        self.regess_1 = ConvBlockResidualGN(self.input_dim, self.input_dim, act_fn,
                                            BatchNorm, GN_num=30, is_dropout=True)
        self.trans_1 = transitionUpGN(self.input_dim, 64, act_fn, BatchNorm,
                                      GN_num=GN_num)

        self.regess_2 = ConvBlockResidualGN(64, 64, act_fn,
                                            BatchNorm, GN_num=GN_num, is_dropout=True)
        self.trans_2 = transitionUpGN(64, 32, act_fn, BatchNorm,
                                      GN_num=GN_num)

    def forward(self, x, dummy):
        x = self.regess_1(x)
        x = self.trans_1(x)
        x = self.regess_2(x)
        x = self.trans_2(x)
        return x


class MLPUnetLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1, use_act=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout=0.1)
        self.proj = nn.Conv2d(self.input_size, self.hidden_size, kernel_size=1)
        self.use_act = use_act
        self.act = nn.GELU() if use_act else nn.Identity()

    def forward(self, inp, skip=None):
        # inp = inp.flatten(2).transpose(1, 2)
        if skip is not None:
            # skip = skip.flatten(2).transpose(1, 2)
            inp = torch.cat([inp, skip], dim=1)
        hidden_states = self.proj(self.dropout(self.act(inp)))
        return hidden_states


class MLPUnet4(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.hidden_size = opt.get("hidden_size", 320)
        self.input_sizes = [256, 160, 64, 32]
        self.output_size = opt.get("output_size", 3)
        self.dropout = opt.get("dropout", 0)
        self.use_act = opt.get("use_act", False)
        layers = []
        for i in range(3):
            layers.append(
                MLPUnetLayer(input_size=self.input_sizes[i] if i == 0 else self.input_sizes[i] + self.hidden_size,
                             hidden_size=self.hidden_size, dropout=self.dropout, use_act=self.use_act))
        self.layers = nn.ModuleList(layers)
        self.fuse = nn.Sequential(
            nn.Conv2d(self.hidden_size + self.input_sizes[-1], self.hidden_size, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.Dropout2d(0.1),
            nn.GELU(),
            nn.Conv2d(self.hidden_size, self.output_size, kernel_size=1)
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.final_upsample = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, dummy, hidden_dims):
        hidden_dims = list(reversed(hidden_dims))
        batch_size, _, height, width = hidden_dims[0].shape
        x = self.layers[0](hidden_dims[0])
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, -1, height, width)
        x = self.upsample(x)
        for i, h in enumerate(hidden_dims[1:-1]):
            batch_size, _, height, width = h.shape
            x = self.layers[i + 1](h, x)
            x = x.permute(0, 2, 1)
            x = x.reshape(batch_size, -1, height, width)
            x = self.upsample(x)
        x = self.fuse(torch.cat([x, hidden_dims[-1]], dim=1))
        return self.final_upsample(x)
