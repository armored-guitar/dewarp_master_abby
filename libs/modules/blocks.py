import torch.nn as nn


def get_batch_norm(name):
    if name == "group_norm":
        return nn.GroupNorm


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)


def transitionUpGN(in_channels, out_dim, act_fn, BatchNorm, GN_num=32):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        BatchNorm(GN_num, out_dim) if out_dim > GN_num else nn.InstanceNorm2d(out_dim),
        act_fn,
    )
    return model


def dilation_conv(in_channels, out_dim, stride=1, dilation=4, groups=1):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_dim, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                  groups=groups),
    )
    return model


def dilation_conv_gn_act(in_channels, out_dim, act_fn, BatchNorm, GN_num=32, dilation=4):
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_dim, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
        BatchNorm(GN_num, out_dim),
        act_fn,
    )
    return model


def upsamplingBilinear(scale_factor=2):
    model = nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=scale_factor),
    )
    return model


class ConvBlockResidualGN(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn, BatchNorm, GN_num=32, is_dropout=False):
        super(ConvBlockResidualGN, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = BatchNorm(GN_num, out_channels)
        self.relu = act_fn
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = BatchNorm(GN_num, out_channels)
        self.is_dropout = is_dropout
        self.drop_out = nn.Dropout2d(p=0.2)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        if self.is_dropout:
            out = self.drop_out(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        if self.is_dropout:
            out = self.drop_out(out)
        return out


class ResidualBlock34DilatedV4GN(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm, GN_num=32, stride=1, downsample=None, is_activation=True,
                 is_top=False, is_dropout=False):
        super(ResidualBlock34DilatedV4GN, self).__init__()
        self.stride = stride
        self.is_activation = is_activation
        self.downsample = downsample
        self.is_top = is_top
        if self.stride != 1 or self.is_top:
            self.conv1 = conv3x3(in_channels, out_channels, self.stride)
        else:
            self.conv1 = dilation_conv(in_channels, out_channels, dilation=1)
        self.bn1 = BatchNorm(GN_num, out_channels) if out_channels > GN_num else nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if self.stride != 1 or self.is_top:
            self.conv2 = conv3x3(out_channels, out_channels)
        else:
            self.conv2 = dilation_conv(out_channels, out_channels, dilation=3)
        self.bn2 = BatchNorm(GN_num, out_channels) if out_channels > GN_num else nn.InstanceNorm2d(out_channels)
        if self.stride == 1 and not self.is_top:
            self.conv3 = dilation_conv(out_channels, out_channels, dilation=1)
            self.bn3 = BatchNorm(GN_num, out_channels) if out_channels > GN_num else nn.InstanceNorm2d(out_channels)
        self.is_dropout = is_dropout
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, x):
        residual = x

        out1 = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out1))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        if self.stride == 1 and not self.is_top:
            if self.is_dropout:
                out = self.drop_out(out)
            out = self.bn3(self.conv3(out))
            out += out1
            if self.is_activation:
                out = self.relu(out)

        return out