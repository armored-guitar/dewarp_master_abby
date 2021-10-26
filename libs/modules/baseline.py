import torch
import torch.nn as nn
import torch.nn.init as tinit
from libs.modules.blocks import *


class EncoderTail(nn.Module):
    def __init__(self, num_filter, map_num, BatchNorm, GN_num=[32, 32, 32, 32], block_nums=[3, 4, 6, 3],
                 block=ResidualBlock34DilatedV4GN, stride=[1, 2, 2, 2], dropRate=[0.2, 0.2, 0.2, 0.2],
                 is_sub_dropout=False):
        super(EncoderTail, self).__init__()
        self.in_channels = num_filter * map_num[0]
        self.dropRate = dropRate
        self.stride = stride
        self.is_sub_dropout = is_sub_dropout
        self.drop_out = nn.Dropout(p=dropRate[0])
        self.drop_out_2 = nn.Dropout(p=dropRate[1])
        self.drop_out_3 = nn.Dropout(p=dropRate[2])
        self.drop_out_4 = nn.Dropout(p=dropRate[3])
        self.relu = nn.ReLU(inplace=True)
        self.block_nums = block_nums
        self.layer1 = self.blocklayer(block, num_filter * map_num[0], self.block_nums[0], BatchNorm, GN_num=GN_num[0],
                                      stride=self.stride[0])
        self.layer2 = self.blocklayer(block, num_filter * map_num[1], self.block_nums[1], BatchNorm, GN_num=GN_num[1],
                                      stride=self.stride[1])
        self.layer3 = self.blocklayer(block, num_filter * map_num[2], self.block_nums[2], BatchNorm, GN_num=GN_num[2],
                                      stride=self.stride[2])
        self.layer4 = self.blocklayer(block, num_filter * map_num[3], self.block_nums[3], BatchNorm, GN_num=GN_num[3],
                                      stride=self.stride[3])

    def blocklayer(self, block, out_channels, block_nums, BatchNorm, GN_num=32, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                BatchNorm(GN_num, out_channels))
        layers = []
        layers.append(
            block(self.in_channels, out_channels, BatchNorm, GN_num, stride, downsample, is_top=True, is_dropout=False))
        self.in_channels = out_channels
        for i in range(1, block_nums):
            layers.append(block(out_channels, out_channels, BatchNorm, GN_num, is_activation=True, is_top=False,
                                is_dropout=self.is_sub_dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.layer1(x)

        out2 = self.layer2(out1)

        out3 = self.layer3(out2)

        out4 = self.layer4(out3)
        return out4


class BaselineEncoder(nn.Module):
    def __init__(self, opt):
        super(BaselineEncoder, self).__init__()
        self.in_channels = opt["in_channels"]
        self.num_filter = opt["nun_filters"]
        self.block_nums = opt["block_nums"]
        self.drop_rate = opt["drop_rate"]
        act_fn = nn.ReLU(inplace=True)

        map_num = [1, 2, 4, 8, 16]
        GN_num = self.num_filter * map_num[0]

        BatchNorm = get_batch_norm(opt["batch_norm"])

        self.resnet_head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.num_filter * map_num[0], kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(self.num_filter * map_num[0]),
            act_fn,
            nn.Conv2d(self.num_filter * map_num[0], self.num_filter * map_num[0], kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(self.num_filter * map_num[0]),
            act_fn,
            nn.Conv2d(self.num_filter * map_num[0], self.num_filter * map_num[0], kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(self.num_filter * map_num[0]),
            act_fn,
        )
        self.resnet_down = EncoderTail(self.num_filter, map_num, BatchNorm, GN_num=[GN_num]*len(self.block_nums),
                                                block_nums=self.block_nums, block=ResidualBlock34DilatedV4GN,
                                                dropRate=self.drop_rate, is_sub_dropout=False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                tinit.xavier_normal_(m.weight, gain=0.2)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                tinit.xavier_normal_(m.weight, gain=0.2)

    def forward(self, x):
        resnet_head = self.resnet_head(x)
        resnet_down = self.resnet_down(resnet_head)
        return [resnet_down]


class BaselineBottleneck(nn.Module):
    def __init__(self, opt):
        super(BaselineBottleneck, self).__init__()
        BatchNorm = get_batch_norm(opt["batch_norm"])
        self.num_filter = opt["nun_filters"]
        act_fn = nn.ReLU(inplace=True)

        map_num = [1, 2, 4, 8, 16]
        GN_num = self.num_filter * map_num[0]

        map_num_i = 3
        self.bridge_1 = nn.Sequential(
            dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                 act_fn, BatchNorm, GN_num=GN_num, dilation=1),
        )
        self.bridge_2 = nn.Sequential(
            dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                 act_fn, BatchNorm, GN_num=GN_num, dilation=2),
        )
        self.bridge_3 = nn.Sequential(
            dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                 act_fn, BatchNorm, GN_num=GN_num, dilation=5),
        )
        self.bridge_4 = nn.Sequential(
            dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                 act_fn, BatchNorm, GN_num=GN_num, dilation=8),
            dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                 act_fn, BatchNorm, GN_num=GN_num, dilation=3),
            dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                 act_fn, BatchNorm, GN_num=GN_num, dilation=2),
        )
        self.bridge_5 = nn.Sequential(
            dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                 act_fn, BatchNorm, GN_num=GN_num, dilation=12),
            dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                 act_fn, BatchNorm, GN_num=GN_num, dilation=7),
            dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                 act_fn, BatchNorm, GN_num=GN_num, dilation=4),
        )
        self.bridge_6 = nn.Sequential(
            dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                 act_fn, BatchNorm, GN_num=GN_num, dilation=18),
            dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                 act_fn, BatchNorm, GN_num=GN_num, dilation=12),
            dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
                                 act_fn, BatchNorm, GN_num=GN_num, dilation=6),
        )

        self.bridge_concate = nn.Sequential(
            nn.Conv2d(self.num_filter * map_num[map_num_i] * 6, self.num_filter * map_num[4], kernel_size=1, stride=1,
                      padding=0),
            BatchNorm(GN_num, self.num_filter * map_num[4]),
            act_fn,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                tinit.xavier_normal_(m.weight, gain=0.2)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                tinit.xavier_normal_(m.weight, gain=0.2)

    def forward(self, resnet_down):
        '''bridge'''
        bridge_1 = self.bridge_1(resnet_down)
        bridge_2 = self.bridge_2(resnet_down)
        bridge_3 = self.bridge_3(resnet_down)
        bridge_4 = self.bridge_4(resnet_down)
        bridge_5 = self.bridge_5(resnet_down)
        bridge_6 = self.bridge_6(resnet_down)

        bridge_concate = torch.cat([bridge_1, bridge_2, bridge_3, bridge_4, bridge_5, bridge_6], dim=1)
        bridge = self.bridge_concate(bridge_concate)

        return bridge


class BaselineDecoder(nn.Module):
    def __init__(self, opt):
        super(BaselineDecoder, self).__init__()
        BatchNorm = get_batch_norm(opt["batch_norm"])
        self.num_filter = opt["nun_filters"]
        act_fn = nn.ReLU(inplace=True)

        map_num = [1, 2, 4, 8, 16]
        GN_num = self.num_filter * map_num[0]

        self.regess_4 = ConvBlockResidualGN(self.num_filter * map_num[4], self.num_filter * (map_num[4]), act_fn,
                                            BatchNorm, GN_num=GN_num, is_dropout=False)
        self.trans_4 = transitionUpGN(self.num_filter * (map_num[4]), self.num_filter * map_num[3], act_fn, BatchNorm,
                                      GN_num=GN_num)

        self.regess_3 = ConvBlockResidualGN(self.num_filter * (map_num[3]), self.num_filter * (map_num[3]), act_fn,
                                            BatchNorm, GN_num=GN_num, is_dropout=False)
        self.trans_3 = transitionUpGN(self.num_filter * (map_num[3]), self.num_filter * map_num[2], act_fn, BatchNorm,
                                      GN_num=GN_num)

        self.regess_2 = ConvBlockResidualGN(self.num_filter * (map_num[2]), self.num_filter * (map_num[2]), act_fn,
                                            BatchNorm, GN_num=GN_num, is_dropout=False)
        self.trans_2 = transitionUpGN(self.num_filter * map_num[2], self.num_filter * map_num[1], act_fn, BatchNorm,
                                      GN_num=GN_num)

        self.regess_1 = ConvBlockResidualGN(self.num_filter * (map_num[1]), self.num_filter * (map_num[1]), act_fn,
                                            BatchNorm, GN_num=GN_num, is_dropout=False)
        self.trans_1 = upsamplingBilinear(scale_factor=2)

        self.regess_0 = ConvBlockResidualGN(self.num_filter * (map_num[1]), self.num_filter * (map_num[1]), act_fn,
                                            BatchNorm, GN_num=GN_num, is_dropout=False)
        self.trans_0 = upsamplingBilinear(scale_factor=2)
        self.up = nn.Sequential(
            nn.Conv2d(self.num_filter * map_num[1], self.num_filter * map_num[0], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(self.num_filter * map_num[0]),
            act_fn,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                tinit.xavier_normal_(m.weight, gain=0.2)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                tinit.xavier_normal_(m.weight, gain=0.2)

    def forward(self, bridge, encoder_input=None):

        regess_4 = self.regess_4(bridge)
        trans_4 = self.trans_4(regess_4)

        regess_3 = self.regess_3(trans_4)
        trans_3 = self.trans_3(regess_3)

        regess_2 = self.regess_2(trans_3)
        trans_2 = self.trans_2(regess_2)

        regess_1 = self.regess_1(trans_2)
        trans_1 = self.trans_1(regess_1)

        regess_0 = self.regess_0(trans_1)
        trans_0 = self.trans_0(regess_0)
        up = self.up(trans_0)

        return up


class BaselineHeads(nn.Module):
    def __init__(self, opt):
        super(BaselineHeads, self).__init__()
        self.n_classes = opt["n_classes"]
        self.num_filter = opt["nun_filters"]
        act_fn = nn.ReLU(inplace=True)

        map_num = [1, 2, 4, 8, 16]

        self.out_regress = nn.Sequential(
            nn.Conv2d(self.num_filter * map_num[0], self.num_filter * map_num[0], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(self.num_filter * map_num[0]),
            nn.PReLU(),
            nn.Conv2d(self.num_filter * map_num[0], self.n_classes, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(self.n_classes),
            nn.PReLU(),
            nn.Conv2d(self.n_classes, self.n_classes, kernel_size=3, stride=1, padding=1),
        )

        self.out_classify = nn.Sequential(
            nn.Conv2d(self.num_filter * map_num[0], self.num_filter * map_num[0], kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(self.num_filter * map_num[0]),
            act_fn,
            nn.Dropout2d(p=0.2),
            nn.Conv2d(self.num_filter * map_num[0], self.n_classes, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(self.n_classes),
            act_fn,
            nn.Conv2d(self.n_classes, 1, kernel_size=3, stride=1, padding=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                tinit.xavier_normal_(m.weight, gain=0.2)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                tinit.xavier_normal_(m.weight, gain=0.2)

    def forward(self, up):
        out_regress = self.out_regress(up)
        out_classify = self.out_classify(up)
        return out_regress, out_classify
