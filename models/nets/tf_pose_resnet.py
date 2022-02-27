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
import tensorflow as tf

print(tf.__version__)
# from relu = tf.nn.relu(bn1)
logger = logging.getLogger(__name__)
initializer = tf.keras.initializers.he_normal()
bn_initializer = {"momentum": 0.1, "epsilon": 1e-5}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return tf.keras.layers.Conv2D(filters=out_planes, kernel_size=(3, 3), strides=stride, padding="same",
                                  use_bias=False, kernel_initializer=initializer)


class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = tf.keras.layers.BatchNormalization(**bn_initializer)
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = tf.keras.layers.BatchNormalization(**bn_initializer)
        self.downsample = downsample
        self.stride = stride

    def call(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=planes, kernel_size=1, use_bias=False,
                                            kernel_initializer=initializer)
        self.bn1 = tf.keras.layers.BatchNormalization(**bn_initializer)
        self.conv2 = tf.keras.layers.Conv2D(filters=planes, kernel_size=3, strides=stride, padding="same",
                                            use_bias=False, kernel_initializer=initializer)
        self.bn2 = tf.keras.layers.BatchNormalization(**bn_initializer)
        self.conv3 = tf.keras.layers.Conv2D(filters=planes * self.expansion, kernel_size=1, use_bias=False,
                                            kernel_initializer=initializer)
        self.bn3 = tf.keras.layers.BatchNormalization(**bn_initializer)
        self.relu = tf.keras.layers.ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(tf.keras.Model):

    def __init__(self, block, layers, cfg, **kwargs):
        super(PoseResNet, self).__init__()
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same", use_bias=False,
                                            kernel_initializer=initializer)
        self.bn1 = tf.keras.layers.BatchNormalization(**bn_initializer)
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same")

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = tf.keras.layers.Conv2D(
            filters=cfg.MODEL.NUM_JOINTS,
            kernel_size=(extra.FINAL_CONV_KERNEL, extra.FINAL_CONV_KERNEL),
            strides=1,
            padding="same",
            kernel_initializer=initializer
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=planes * block.expansion,
                                       kernel_size=(1, 1),
                                       strides=stride,
                                       use_bias=False,
                                       kernel_initializer=initializer),
                tf.keras.layers.BatchNormalization(**bn_initializer)],
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return tf.keras.Sequential(layers)

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
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                tf.keras.layers.Conv2DTranspose(
                    filters=planes,
                    kernel_size=(kernel, kernel),
                    strides=2,
                    padding="same",
                    use_bias=self.deconv_with_bias,
                    kernel_initializer=initializer))
            layers.append(tf.keras.layers.BatchNormalization(**bn_initializer))
            layers.append(tf.keras.layers.ReLU())
            self.inplanes = planes

        return tf.keras.Sequential(layers)

    def call(self, x, **kwargs):
        x = self.conv1(x)  # input x:torch.Size([32, 3, 256, 192])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)  # torch.Size([32, 2048, 8, 6])->torch.Size([32, 256, 64, 48])
        x = self.final_layer(x)  # torch.Size([32, 256, 64, 48])->torch.Size([32, 17, 64, 48])

        return x


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


# resnet_spec = {18: (Bottleneck, [2, 2, 2, 2]),
#                34: (Bottleneck, [3, 4, 6, 3]),
#                50: (Bottleneck, [3, 4, 6, 3]),
#                101: (Bottleneck, [3, 4, 23, 3]),
#                152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]
    model = PoseResNet(block_class, layers, cfg, **kwargs)
    return model


def demo_for_v1():
    """
    (None, 32, 32, 64), x:(None, 16, 16, 256)
    (None, 64, 64, 32), x_8:(None, 32, 32, 64), x:(None, 16, 16, 256)
    """
    # from models.keras_utils import keras_callback
    from utils import tf_tools
    from models.core.config import config
    tf_tools.set_device_memory()

    model = get_pose_net(config, is_train=False)
    input_size = [256, 192]
    input_shape = [1, input_size[1], input_size[0], 3]
    # inputs = tf.keras.layers.Input(shape=(input_size[1], input_size[0], 3), dtype=tf.float32)
    # output = model.build(input_shape=input_shape)
    # model.summary()
    data = tf.random.uniform((1, 256, 192, 3))
    # print_model_summary(model, input_shape)
    tf_tools.plot_model(model, input_shape)
    out = model(data)
    print(out.shape)


if __name__ == "__main__":
    demo_for_v1()
