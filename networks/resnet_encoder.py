# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from my_modules.LWN import C2f_LWN
from my_modules.ScConv import ScConv
from my_modules.wt_down import res_Down_wt as Down_wt
from my_modules.dsrConv import DSRConv
# 在文件顶部添加导入
from torchvision.models import get_weight, ResNet18_Weights, ResNet50_Weights


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        # loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        if num_layers == 18:
            loaded = torch.load("/home/zb/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
        elif num_layers == 50:
            loaded = torch.load("/home/zb/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth")


        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model

#encoder = networks.ResnetEncoder(18, False)
#features = encoder(input_image)
class ResnetEncoder(nn.Module):
    def __init__(self, num_layers, pretrained, num_input_images=1, use_lwn=False, use_scconv=False, use_WtResnet=False, use_WtFusion=False, use_dsrconv=False,):
        super(ResnetEncoder, self).__init__()

        self.use_WtFusion = use_WtFusion
        if use_WtFusion:
            print("encoder use WtFusion")
            self.down1 = Down_wt(3, 64)
            self.down_fusion1 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=1, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),)

            self.down2 = Down_wt(64, 256)
            self.down_fusion2 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),)

            self.down3 = Down_wt(256, 512)
            self.down_fusion3 = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1, stride=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),)


        self.use_lwn = use_lwn
        if use_lwn:
            print("encoder use c2f_lwn")
            self.c2f_lwn_1 = C2f_LWN(64, 64, shortcut=True)
            self.c2f_lwn_2 = C2f_LWN(256, 256, shortcut=True)
            self.c2f_lwn_3 = C2f_LWN(512, 512, shortcut=True)

        self.use_scconv = use_scconv
        if use_scconv:
            print("encoder use ScConv")
            self.scconv_1 = ScConv(64, shortcut=True)
            self.scconv_2 = ScConv(256, shortcut=True)
            self.scconv_3 = ScConv(512, shortcut=True)

        self.use_dsrconv = use_dsrconv
        if use_dsrconv:
            print("encoder use DSR-Conv")
            self.dsrconv_1 = DSRConv(64, threshold=0.3)
            self.dsrconv_2 = DSRConv(256, threshold=0.3)
            self.dsrconv_3 = DSRConv(512, threshold=0.3)

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}
        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        # 使用小波下采样替换卷积
        if use_WtResnet:
            print("Encoder use WtResnet")
            resnet_wt(self.encoder)
            print("####################################")
            print(self.encoder)

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        if self.use_WtFusion:
            x_down = self.down1(x)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        x1 = self.encoder.relu(x)
        # print("x1", x1.shape)
        if self.use_WtFusion:
            x1 = self.down_fusion1(torch.cat([x_down, x1], 1))
        if self.use_lwn:
            x1 = self.c2f_lwn_1(x1)
        if self.use_scconv:
            x1 = self.scconv_1(x1)
        elif self.use_dsrconv:
            x1 = self.dsrconv_1(x1)
        self.features.append(x1)
        x2 = self.encoder.layer1(self.encoder.maxpool(self.features[-1]))
        # print("x2", x2.shape)
        if self.use_WtFusion:
            x_down = self.down2(x_down)
            x2 = self.down_fusion2(torch.cat([x_down, x2], 1))
        if self.use_lwn:
            x2 = self.c2f_lwn_2(x2)
        if self.use_scconv:
            x2 = self.scconv_2(x2)
        elif self.use_dsrconv:
            x2 = self.dsrconv_2(x2)
        self.features.append(x2)
        x3 = self.encoder.layer2(self.features[-1])
        if self.use_WtFusion:
            x_down = self.down3(x_down)
            x3 = self.down_fusion3(torch.cat([x_down, x3], 1))
        # print("x3", x3.shape)
        if self.use_lwn:
            x3 = self.c2f_lwn_3(x3)
        if self.use_scconv:
            x3 = self.scconv_3(x3)
        elif self.use_dsrconv:
            x3 = self.dsrconv_3(x3)
        self.features.append(x3)

        # self.features.append(self.encoder.layer3(self.features[-1]))
        # self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
class ResnetEncoder2(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder2, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

#######################################################

def resnet_wt(model, verbose=True):
    """
    原地替换 ResNet 中 **所有 stride=2 的卷积**：
    1. 主分支：layer2/3/4 第一个 block 的 conv2（3×3, stride=2）
    2. shortcut 分支：同 block 的 downsample[0]（1×1, stride=2）
    都换成 Down_wt，并把对应 BN/ReLU 置为 Identity。
    """
    backbone = model.encoder if hasattr(model, 'encoder') else model

    for lyr_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(backbone, lyr_name, None)
        if layer is None or len(layer) == 0:
            continue
        first_block = layer[0]
        block_type = type(first_block).__name__
        if block_type not in ('BasicBlock', 'Bottleneck'):
            continue

        # ---------- 主分支 ----------
        conv2 = first_block.conv2
        stride = conv2.stride[0] if isinstance(conv2.stride, tuple) else conv2.stride
        if stride == 2:
            in_ch = conv2.in_channels
            out_ch = conv2.out_channels
            first_block.conv2 = Down_wt(in_ch, out_ch)
            first_block.bn2   = nn.Identity()
            # Bottleneck 还有一个 relu 在主干尾部
            # if hasattr(first_block, 'relu'):
            #     first_block.relu = nn.Identity()
            if verbose:
                print(f"[{lyr_name}] 主分支 conv2 -> Down_wt")

        # ---------- shortcut / downsample ----------
        if first_block.downsample is not None:
            ds_conv = first_block.downsample[0]   # 通常是 1×1 stride=2
            ds_stride = ds_conv.stride[0] if isinstance(ds_conv.stride, tuple) else ds_conv.stride
            if ds_stride == 2:
                in_ch = ds_conv.in_channels
                out_ch = ds_conv.out_channels
                # 用 Down_wt 替换，输出通道保持与原 1×1 一致
                first_block.downsample[0] = Down_wt(in_ch, out_ch)
                # 去掉后面的 BN（ReLU 在 downsample 里通常没有）
                first_block.downsample[1] = nn.Identity()
                if verbose:
                    print(f"[{lyr_name}] downsample conv -> Down_wt")



