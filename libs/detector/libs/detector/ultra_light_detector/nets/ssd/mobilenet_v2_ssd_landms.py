import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from nets.nn.mobilenet_v2.mobilenet_v2 import MobileNetV2, InvertedResidual
from nets.ssd.ssd_landms import SSD
from torch.nn import Conv2d, Sequential, ModuleList, ReLU


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def create_mobilenetv2_ssd_landms(prior_boxes, num_classes, is_test=False, width_mult=1.0, device="cuda:0"):
    """
    <class 'list'>: [[24, 12, 6, 3], [24, 12, 6, 3]]
    "shrinkage": [8, 16, 32, 64],
    index=7,c=192
    :param prior_boxes:
    :param num_classes:
    :param is_test:
    :param device:
    :return:
    """
    base_net = MobileNetV2(width_mult=width_mult, use_batch_norm=True, onnx_compatible=False)
    base_net_model = base_net.features
    # GraphPath must import from SSD
    # source_layer_indexes = [GraphPath(11, 'conv', 2), 15, ]  # source_layer_indexes = [8, 16, 19]
    source_layer_indexes = [7, 11, 17]
    # channels = [32, 64, 160, base_net.last_channel]
    channels = [int(32 * width_mult), int(64 * width_mult), int(160 * width_mult), base_net.last_channel]
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=channels[3], out_channels=channels[2], kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=channels[2], out_channels=channels[3],
                            kernel_size=3, stride=2, padding=1),
            ReLU()
        )
    ])

    boxes_expand = [len(boxes) * (len(prior_boxes.aspect_ratios)) for boxes in prior_boxes.min_boxes]
    regression_headers = ModuleList([
        SeperableConv2d(in_channels=channels[0],
                        out_channels=boxes_expand[0] * 4,
                        kernel_size=3,
                        padding=1),
        SeperableConv2d(in_channels=channels[1],
                        out_channels=boxes_expand[1] * 4,
                        kernel_size=3,
                        padding=1),
        SeperableConv2d(in_channels=channels[2],
                        out_channels=boxes_expand[2] * 4,
                        kernel_size=3,
                        padding=1),
        Conv2d(in_channels=channels[3],
               out_channels=boxes_expand[3] * 4,
               kernel_size=3,
               padding=1)])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=channels[0],
                        out_channels=boxes_expand[0] * num_classes,
                        kernel_size=3,
                        padding=1),
        SeperableConv2d(in_channels=channels[1],
                        out_channels=boxes_expand[1] * num_classes,
                        kernel_size=3,
                        padding=1),
        SeperableConv2d(in_channels=channels[2],
                        out_channels=boxes_expand[2] * num_classes,
                        kernel_size=3,
                        padding=1),
        Conv2d(in_channels=channels[3],
               out_channels=boxes_expand[3] * num_classes,
               kernel_size=3,
               padding=1)])

    landms_headers = ModuleList([
        SeperableConv2d(in_channels=channels[0],
                        out_channels=boxes_expand[0] * 10,
                        kernel_size=3,
                        padding=1),
        SeperableConv2d(in_channels=channels[1],
                        out_channels=boxes_expand[1] * 10,
                        kernel_size=3,
                        padding=1),
        SeperableConv2d(in_channels=channels[2],
                        out_channels=boxes_expand[2] * 10,
                        kernel_size=3,
                        padding=1),
        Conv2d(in_channels=channels[3],
               out_channels=boxes_expand[3] * 10,
               kernel_size=3,
               padding=1)])

    return SSD(num_classes, base_net_model, source_layer_indexes,
               extras, classification_headers, regression_headers, landms_headers, is_test=is_test,
               prior_boxes=prior_boxes,
               device=device)
