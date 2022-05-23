import torch
import torch.nn as nn
from positional_encodings import PositionalEncoding2D


class TransformerBottleneck(nn.Module):
    def __init__(self, opt):
        super().__init__()
        hidden_size = opt.get("hidden_size", 256)
        bottleneck_out_dim = opt.get("bottleneck_out_dim", 512)
        n_heads = opt.get("n_heads", 8)
        num_layers = opt.get("num_layers", 6)
        self.hidden_size = hidden_size
        self.bottleneck_out_dim = bottleneck_out_dim
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.pos_emb = PositionalEncoding2D(hidden_size)
        self.preoutput = nn.Conv2d(hidden_size, bottleneck_out_dim, kernel_size=3, padding=1)

    def forward(self, x):
        _, hidden_size, height, width = x.shape
        # bs, x, y, ch            #bs, ch, x, y
        x = x + self.pos_emb(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x.flatten(2).transpose(1, 2)
        batch_size, seq_len, hidden_size = x.shape
        x = self.transformer_encoder(x)

        x = x.transpose(1, 2).view(batch_size, hidden_size, height, width)
        x = self.preoutput(x)
        return x


class TransformerFullBottleneck(nn.Module):
    def __init__(self, opt):
        super().__init__()
        hidden_size = opt.get("hidden_size", 256)
        bottleneck_out_dim = opt.get("bottleneck_out_dim", 512)
        n_heads = opt.get("n_heads", 8)
        num_layers = opt.get("num_layers", 6)
        self.hidden_size = hidden_size
        self.bottleneck_out_dim = bottleneck_out_dim
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.pos_emb = PositionalEncoding2D(hidden_size)

        self.preoutput = nn.Conv2d(hidden_size, bottleneck_out_dim, kernel_size=3, padding=1)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
        self.learnable = nn.Parameter(torch.randn(1, 32 * 30, hidden_size))
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

    def forward(self, x):
        _, hidden_size, height, width = x.shape
        # bs, x, y, ch            #bs, ch, x, y
        pos_embed_1 = self.pos_emb(x.permute(0, 2, 3, 1))
        x = x + self.pos_emb(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x.flatten(2).transpose(1, 2)
        batch_size, seq_len, hidden_size = x.shape

        x = self.transformer_encoder(x)
        x = self.transformer_decoder(tgt=self.learnable + pos_embed_1.reshape(batch_size, -1, self.hidden_size),
                                     memory=x).transpose(1, 2).view(batch_size, hidden_size, height, width)
        x = self.preoutput(x)
        return x
