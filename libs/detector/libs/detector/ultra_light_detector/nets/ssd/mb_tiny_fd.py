# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-02-14 15:34:50
"""

from torch.nn import Conv2d, Sequential, ModuleList, ReLU
from nets.nn.mb_tiny import Mb_Tiny
from nets.ssd.ssd import SSD


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )

def create_mb_tiny_slim_fd(prior_boxes, num_classes, is_test=False, width_mult=1.0, device="cuda:0"):
    base_net = Mb_Tiny(num_classes)
    base_net_model = base_net.model  # disable dropout layer

    source_layer_indexes = [8, 11, 13]
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=base_net.base_channel * 16, out_channels=base_net.base_channel * 4, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=base_net.base_channel * 4, out_channels=base_net.base_channel * 16,
                            kernel_size=3, stride=2, padding=1),
            ReLU()
        )
    ])
    boxes_expand = [len(boxes) * (len(prior_boxes.aspect_ratios)) for boxes in prior_boxes.min_boxes]

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=base_net.base_channel * 4,
                        out_channels=boxes_expand[0] * 4,
                        kernel_size=3,
                        padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 8,
                        out_channels=boxes_expand[1] * 4,
                        kernel_size=3,
                        padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 16,
                        out_channels=boxes_expand[2] * 4,
                        kernel_size=3,
                        padding=1),
        Conv2d(in_channels=base_net.base_channel * 16,
               out_channels=boxes_expand[3] * 4,
               kernel_size=3,
               padding=1)
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=base_net.base_channel * 4,
                        out_channels=boxes_expand[0] * num_classes,
                        kernel_size=3,
                        padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 8,
                        out_channels=boxes_expand[1] * num_classes,
                        kernel_size=3,
                        padding=1),
        SeperableConv2d(in_channels=base_net.base_channel * 16,
                        out_channels=boxes_expand[2] * num_classes,
                        kernel_size=3,
                        padding=1),
        Conv2d(in_channels=base_net.base_channel * 16,
               out_channels=boxes_expand[3] * num_classes,
               kernel_size=3,
               padding=1)
    ])

    return SSD(num_classes, base_net_model, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, prior_boxes=prior_boxes,
               device=device)
