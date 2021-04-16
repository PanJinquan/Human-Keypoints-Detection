# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from collections import OrderedDict
from models.nets.resnest.pytorch.resnest import resnest50, resnest101

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class PoseResNet(nn.Module):
    def __init__(self, cfg, num_layers, pretrained=False, **kwargs):
        super(PoseResNet, self).__init__()
        if cfg.MODEL.EXTRA.TARGET_TYPE == 'offset':
            factor = 3
        else:
            factor = 1

        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        if num_layers == 50:
            self.backbone = resnest50(pretrained=pretrained)
        elif num_layers == 101:
            self.backbone = resnest101(pretrained=pretrained)
        self.inplanes = 2048
        model_dict = OrderedDict(self.backbone.named_children())
        model_dict.pop("avgpool")
        model_dict.pop("fc")
        self.backbone = torch.nn.Sequential(model_dict)
        # state_dict2 = model.state_dict()

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS*factor,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
        self.layers_features = {"backbone": 0, "deconv_layers": 0, "final_layer": 0}

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def get_layers_features(self, info=False):
        """
        backbone:torch.Size([32, 2048, 8, 8])
        deconv_layers:torch.Size([32, 256, 64, 64])
        final_layer:torch.Size([32, 16, 64, 64])
        :param info:
        :return:
        """
        if info:
            for k, v in self.layers_features.items():
                print("{}:{}".format(k, v.shape))
        return self.layers_features

    def forward(self, x):
        x = self.backbone(x)  # torch.Size([32, 2048, 8, 6])
        if "backbone" in self.layers_features:
            self.layers_features["backbone"] = x
        x = self.deconv_layers(x)  # torch.Size([32, 2048, 8, 6])->torch.Size([32, 256, 64, 48])
        if "deconv_layers" in self.layers_features:
            self.layers_features["deconv_layers"] = x
        x = self.final_layer(x)  # torch.Size([32, 256, 64, 48])->torch.Size([32, 17, 64, 48])
        if "final_layer" in self.layers_features:
            self.layers_features["final_layer"] = x
        return x

    def _initialize_weights_kaiming_normal(self):
        """
        custom weights initialize
        :return:
        """
        logger.info('=>initialize_weights_kaiming_normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _initialize_weights_normal(self):
        logger.info('=> initialize_weights_normaln')
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                logger.info('=> init {}.weight as 1'.format(name))
                logger.info('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        logger.info('=> init final conv weights from normal distribution')
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=''):
        self._initialize_weights_kaiming_normal()
        # self._initialize_weights()
        if os.path.isfile(pretrained):
            # pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            # logger.error('=> please download it first')
            # raise ValueError('imagenet pretrained model does not exist')


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    pretrained = cfg.MODEL.PRETRAINED
    if cfg.MODEL.PRETRAINED:
        # pretrained = True
        pass
    else:
        # pretrained = False
        pass
    logger.info('=> pretrained:{}'.format(pretrained))
    model = PoseResNet(cfg, num_layers, pretrained=pretrained, **kwargs)
    return model


if __name__ == "__main__":
    """
    Total params: 15,376,464
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    Total memory: 37.12MB
    Total MAdd: 5.81GMAdd
    Total Flops: 1.8GFlops
    Total MemR+W: 112.97MB
    """
    from models.core.config import config
    from utils import torch_tools

    config.MODEL.EXTRA.NUM_LAYERS = 50
    device = "cpu"
    model = get_pose_net(config, is_train=True).to(device)
    input = torch.randn(size=(32, 3, 256, 256))
    input = input.to(device)
    out = model(input)
    out = model(input)
    print("out:{}".format(out.shape))
    # torch_tools.summary_model(model, batch_size=1, input_size=[192, 256])
    model.get_layers_features(info=True)
