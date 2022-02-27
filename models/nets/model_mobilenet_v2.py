import torch
import torch.nn as nn
import math

# Modified from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py.
# In this version, Relu6 is replaced with Relu to make it ONNX compatible.
# BatchNorm Layer is optional to make it easy do batch norm confusion.
BN_MOMENTUM = 0.1


def conv_bn(inp, oup, stride, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            ReLU(inplace=True)
        )


def conv_1x1_bn(inp, oup, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            ReLU(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_batch_norm=True, onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, cfg, width_mult=1., dropout_ratio=0.2, use_batch_norm=True, onnx_compatible=False):
        super(MobileNetV2, self).__init__()

        #########################################
        self.layers_features = {"backbone": 0, "deconv_layers": 0, "final_layer": 0}
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        if cfg.MODEL.EXTRA.TARGET_TYPE == 'offset':
            factor = 3
        else:
            factor = 1
        ########################################

        block = InvertedResidual
        input_channel = 32
        # last_channel = 1280
        last_channel = 320
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        # assert input_size[0] % 32 == 0
        input_channel = int(input_channel * width_mult)
        # self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.last_channel = int(last_channel * width_mult)
        self.features = [conv_bn(3, input_channel, 2, onnx_compatible=onnx_compatible)]
        self.inplanes = self.last_channel

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                else:
                    self.features.append(block(input_channel, output_channel, 1,
                                               expand_ratio=t, use_batch_norm=use_batch_norm,
                                               onnx_compatible=onnx_compatible))
                input_channel = output_channel
        # building last several layers
        # self.features.append(conv_1x1_bn(input_channel,
        #                                  self.last_channel,
        #                                  use_batch_norm=use_batch_norm,
        #                                  onnx_compatible=onnx_compatible))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=int(cfg.MODEL.NUM_JOINTS * factor),
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(dropout_ratio),
        #     nn.Linear(self.last_channel, out_features),
        # )

        # self._initialize_weights()
        self._initialize_weights_v2()
        # self._initialize_weights_v2()
        # self.bn = nn.BatchNorm1d(out_features)

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
        if "backbone" in self.layers_features:
            self.layers_features["backbone"] = x
        # x = x.mean(3).mean(2)
        # print("x.shape:{}".format(x.shape))
        # x = self.classifier(x)
        # print("x.shape:{}".format(x.shape))
        # x = self.bn(x)
        x = self.deconv_layers(x)  # torch.Size([32, 2048, 8, 6])->torch.Size([32, 256, 64, 48])
        if "deconv_layers" in self.layers_features:
            self.layers_features["deconv_layers"] = x
        x = self.final_layer(x)  # torch.Size([32, 256, 64, 48])->torch.Size([32, 17, 64, 48])
        if "final_layer" in self.layers_features:
            self.layers_features["final_layer"] = x
        return x

    def _initialize_weights_v1(self):
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

    def _initialize_weights(self):
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


def get_pose_net(cfg, is_train, **kwargs):
    model = MobileNetV2(cfg, **kwargs)
    # if is_train and os.path.isfile(cfg.MODEL.PRETRAINED):
    #     print('=> loading pretrained model {}'.format(cfg.MODEL.PRETRAINED))
    #     state_dict = torch.load(cfg.MODEL.PRETRAINED)
    #     model.load_state_dict(state_dict, strict=False)
    # else:
    #     print("no pretrained")
    print("no pretrained")
    return model



if __name__ == "__main__":
    """
    -------------------------------------------------------------------------------------------------------------------------------------------------------------
    Total memory: 49.72MB
    Total MAdd: 267.23MMAdd
    Total Flops: 122.18MFlops
    Total MemR+W: 101.59MB
    -----------------------------------------------------------------------------------------------------------------------------------------------------------
    Total memory: 49.25MB
    Total MAdd: 238.23MMAdd
    Total Flops: 115.5MFlops
    Total MemR+W: 100.38MB
    """
    from models.core.config import config
    from utils import torch_tools

    device = "cuda:0"
    config.MODEL.EXTRA.NUM_DECONV_FILTERS = [32, 32, 32]
    config.MODEL.EXTRA.WIDTH_MULT = 0.5
    model = get_pose_net(config, width_mult=config.MODEL.EXTRA.WIDTH_MULT,is_train=False).to(device)
    model.eval()
    input = torch.randn(size=(1, 3, 288, 288))
    input = input.to(device)
    out = model(input)
    print("out:{}".format(out.shape))
    torch_tools.summary_model(model, batch_size=1, input_size=[256, 256])
    model.get_layers_features(info=True)
