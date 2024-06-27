import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torch.nn import init
import numpy as np
import functools
import math


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class SelfAttention(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(SelfAttention, self).__init__()
        
        self.basic = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_layer(in_channel)
        self.weight = nn.Conv2d(in_channel, in_channel*2, kernel_size=3, stride=1, padding=1)
        self.in_c = in_channel

        initialize_weights([self.basic, self.weight], 0.1)

    def forward(self, x):
        basic = F.relu(self.bn1(self.basic(x)), inplace=True)
        weight = self.weight(basic)
        w, b = weight[:, :self.in_c, :, :], weight[:, self.in_c:, :, :]

        return F.relu(w * basic + b, inplace=True)


class CrossAttention(nn.Module):
    def __init__(self, in_channel=128, ratio=8):
        super(CrossAttention, self).__init__()
        self.conv_query = nn.Conv2d(in_channel, in_channel // ratio, kernel_size=1, bias=False)
        self.conv_key = nn.Conv2d(in_channel, in_channel // ratio, kernel_size=1, bias=False)
        self.conv_value = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)

        initialize_weights([self.conv_query, self.conv_key, self.conv_value], 0.1)

    def forward(self, x, y):
        bz, c, h, w = x.shape
        y_q = self.conv_query(y).view(bz, -1, h * w).permute(0, 2, 1)
        y_k = self.conv_key(y).view(bz, -1, h * w)
        mask = torch.bmm(y_q, y_k)  # bz, hw, hw
        mask = torch.softmax(mask, dim=-1)
        x_v = self.conv_value(x).view(bz, c, -1)
        feat = torch.bmm(x_v, mask.permute(0, 2, 1))  # bz, c, hw
        feat = feat.view(bz, c, h, w)

        return feat


class Feature_Exchange_Module(nn.Module):
    def __init__(self, in_channel, CA=True, ratio=8):
        super(Feature_Exchange_Module, self).__init__()
        self.CA = CA
        self.sa1 = SelfAttention(in_channel)
        self.sa2 = SelfAttention(in_channel)
        
        if self.CA:
            self.att1 = CrossAttention(in_channel, ratio=ratio)
            self.att2 = CrossAttention(in_channel, ratio=ratio)

    def forward(self, pos, neg, beta, gamma):
        pos = self.sa1(pos)
        neg = self.sa2(neg)
        
        if self.CA:
            feat_1 = self.att1(pos, neg)
            feat_2 = self.att2(neg, pos)

        pos_out = pos + beta * feat_1
        neg_out = neg + gamma * feat_2

        return pos_out, neg_out


class Feature_fusion_module(nn.Module):
    def __init__(self, in_channel, out_channel, ratio=4):
        super(Feature_fusion_module, self).__init__()
        self.basic_block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.local_att = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // ratio, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel // ratio, out_channel, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel, out_channel // ratio, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel // ratio, out_channel, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.sigmoid = nn.Sigmoid()
        initialize_weights([self.basic_block, self.local_att, self.global_att], 0.1)

    def forward(self, aux, main):
        fusion = torch.cat([aux, main], dim=1)
        fusion = self.basic_block(fusion)
        local_att = self.local_att(fusion)
        global_att = self.global_att(fusion)
        main = main + fusion * self.sigmoid(local_att + global_att)
        
        return main


def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
