"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""
import torch
import os
import torch.nn as nn
import math

__all__ = ['mobilenetv3_large', 'mobilenetv3_small']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfg, cfgs, mode, num_classes=1000, width_mult=1.):
        """
        :param cfgs:
        :param mode:
        :param num_classes:
        :param width_mult:
        """
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        #########################################
        self.layers_features = {"backbone": 0, "deconv_layers": 0, "final_layer": 0}
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        if cfg.MODEL.EXTRA.TARGET_TYPE == 'offset':
            factor = 3
        else:
            factor = 1
        ########################################
        assert mode in ['large', 'small']

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.inplanes = exp_size
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # output_channel = {'large': 1280, 'small': 1024}
        # output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[
        #     mode]
        # self.classifier = nn.Sequential(
        #     nn.Linear(exp_size, output_channel),
        #     h_swish(),
        #     nn.Dropout(0.2),
        #     nn.Linear(output_channel, num_classes),
        # )

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS * factor,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        self._initialize_weights_v2()

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
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def get_layers_features(self, info=False):
        """
        backbone:torch.Size([32, 1280, 8, 8])
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
        x = self.features(x)
        x = self.conv(x)
        if "backbone" in self.layers_features:
            self.layers_features["backbone"] = x
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        x = self.deconv_layers(x)  # torch.Size([32, 2048, 8, 6])->torch.Size([32, 256, 64, 48])
        if "deconv_layers" in self.layers_features:
            self.layers_features["deconv_layers"] = x
        x = self.final_layer(x)  # torch.Size([32, 256, 64, 48])->torch.Size([32, 17, 64, 48])
        if "final_layer" in self.layers_features:
            self.layers_features["final_layer"] = x
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _initialize_weights_v2(self):
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


def mobilenetv3_large(cfg, pretrained, **kwargs):
    """
    Constructs a MobileNetV3-Large model
    :param cfgs:
    :param mode:
    :param num_classes:
    :param width_mult:
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3, 1, 16, 0, 0, 1],
        [3, 4, 24, 0, 0, 2],
        [3, 3, 24, 0, 0, 1],
        [5, 3, 40, 1, 0, 2],
        [5, 3, 40, 1, 0, 1],
        [5, 3, 40, 1, 0, 1],
        [3, 6, 80, 0, 1, 2],
        [3, 2.5, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [5, 6, 160, 1, 1, 2],
        [5, 6, 160, 1, 1, 1],
        [5, 6, 160, 1, 1, 1]
    ]
    # model = MobileNetV3(cfgs, mode='large', **kwargs)
    model = MobileNetV3(cfg, cfgs, mode='large', **kwargs)
    if pretrained:
        file = get_pretrained_url(mode='large', **kwargs)
        model.load_state_dict(torch.load(file), strict=False)
        print("load pretrain model:{}".format(file))
    return model


def mobilenetv3_small(cfg, pretrained, **kwargs):
    """
    Constructs a MobileNetV3-Small model
    :param cfgs:
    :param mode:
    :param num_classes:
    :param width_mult:
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3, 1, 16, 1, 0, 2],
        [3, 4.5, 24, 0, 0, 2],
        [3, 3.67, 24, 0, 0, 1],
        [5, 4, 40, 1, 1, 2],
        [5, 6, 40, 1, 1, 1],
        [5, 6, 40, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 6, 96, 1, 1, 2],
        [5, 6, 96, 1, 1, 1],
        [5, 6, 96, 1, 1, 1],
    ]
    model = MobileNetV3(cfg, cfgs, mode='small', **kwargs)
    if pretrained:
        file = get_pretrained_url(mode='small', **kwargs)
        model.load_state_dict(torch.load(file), strict=False)
        print("load pretrain model:{}".format(file))
    return model


def get_pretrained_url(mode='small', width_mult=1.):
    """
    :param mode:
    :param width_mult:
    :return:
    """
    if mode == 'small' and width_mult == 1.:
        file = "pretrained/mobilenetv3-small-55df8e1f.pth"
    elif mode == 'small' and width_mult == 0.75:
        file = "pretrained/mobilenetv3-small-0.75-86c972c3.pth"
    elif mode == 'large' and width_mult == 1.:
        file = "pretrained/mobilenetv3-large-1cd25616.pth"
    elif mode == 'large' and width_mult == 0.75:
        file = "pretrained/mobilenetv3-large-0.75-9632d2a8.pth"
    else:
        raise Exception("Error:{}:{}".format(mode, width_mult))
    file = os.path.join(os.path.dirname(__file__), file)
    return file


if __name__ == "__main__":
    import numpy as np
    from torchviz import make_dot
    from torchsummary import summary
    from utils import torch_tools

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_classes = 2
    input_size = [112, 112]
    x = torch.randn(size=(batch_size, 3, input_size[0], input_size[1])).to(device)
    print("x.shape:{}".format(x.shape))
    net_type = "mobilenet_v3"
    # model = mobilenetv3_small(pretrained=True, width_mult=1.0).to(device)
    model = mobilenetv3_large(pretrained=True, width_mult=1.0).to(device)
    out = model(x)
    torch.save(model.state_dict(), net_type + ".pth")
    torch_tools.summary_model(model, batch_size=1, input_size=input_size)
