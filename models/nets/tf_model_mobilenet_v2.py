import tensorflow as tf

# from relu = tf.nn.relu(bn1)
BN_MOMENTUM = 0.1
initializer = tf.keras.initializers.he_normal()
print(tf.__version__)

import math


# Modified from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py.
# In this version, Relu6 is replaced with Relu to make it ONNX compatible.
# BatchNorm Layer is optional to make it easy do batch norm confusion.

def conv_bn(inp, oup, stride, use_batch_norm=True, onnx_compatible=False):
    ReLU = tf.keras.layers.ReLU() if onnx_compatible else tf.keras.layers.ReLU(max_value=6)
    if use_batch_norm:
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(oup, 3, stride, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=1e-5),
            ReLU]
        )
    else:
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(oup, 3, stride, padding="same", use_bias=False),
            ReLU]
        )


def conv_1x1_bn(inp, oup, use_batch_norm=True, onnx_compatible=False):
    ReLU = tf.keras.layers.ReLU() if onnx_compatible else tf.keras.layers.ReLU(max_value=6)
    if use_batch_norm:
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(oup, 1, 1, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=1e-5),
            ReLU]
        )
    else:
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(oup, 1, 1, padding="same", use_bias=False),
            ReLU]
        )


class InvertedResidual(tf.keras.layers.Layer):
    def __init__(self, inp, oup, stride, expand_ratio, use_batch_norm=True, onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        ReLU = tf.keras.layers.ReLU() if onnx_compatible else tf.keras.layers.ReLU(max_value=6)

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = tf.keras.Sequential([
                    # dw
                    tf.keras.layers.DepthwiseConv2D(3, stride, padding="same", use_bias=False),
                    tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=1e-5),
                    ReLU,
                    # pw-linear
                    tf.keras.layers.Conv2D(oup, 1, 1, padding="same", use_bias=False),
                    tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=1e-5), ]
                )
            else:
                self.conv = tf.keras.Sequential([
                    # dw
                    tf.keras.layers.DepthwiseConv2D(3, stride, padding="same", use_bias=False),
                    ReLU,
                    # pw-linear
                    tf.keras.layers.Conv2D(oup, 1, 1, padding="same", use_bias=False), ]
                )
        else:
            if use_batch_norm:
                self.conv = tf.keras.Sequential([
                    # pw
                    tf.keras.layers.Conv2D(hidden_dim, 1, 1, padding="same", use_bias=False),
                    tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=1e-5),
                    ReLU,
                    # dw
                    tf.keras.layers.DepthwiseConv2D(3, stride, padding="same", use_bias=False),
                    tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=1e-5),
                    ReLU,
                    # pw-linear
                    tf.keras.layers.Conv2D(oup, 1, 1, padding="same", use_bias=False),
                    tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=1e-5), ]
                )
            else:
                self.conv = tf.keras.Sequential([
                    # pw
                    tf.keras.layers.Conv2D(hidden_dim, 1, 1, padding="same", use_bias=False),
                    ReLU,
                    # dw
                    tf.keras.layers.DepthwiseConv2D(3, stride, padding="same", use_bias=False),
                    ReLU,
                    # pw-linear
                    tf.keras.layers.Conv2D(oup, 1, 1, padding="same", use_bias=False), ]
                )

    def call(self, x, **kwargs):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(tf.keras.Model):
    def __init__(self, cfg, width_mult=1., dropout_ratio=0.2, use_batch_norm=True, onnx_compatible=False):
        super(MobileNetV2, self).__init__()
        self.layers_features = {"backbone": 0, "deconv_layers": 0, "final_layer": 0}

        #########################################
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        ########################################

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
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
        self.features.append(conv_1x1_bn(input_channel, self.last_channel,
                                         use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
        # make it nn.Sequential
        self.features = tf.keras.Sequential(self.features)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = tf.keras.layers.Conv2D(
            filters=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            strides=1,
            padding="same",
        )

        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(dropout_ratio),
        #     nn.Linear(self.last_channel, out_features),
        # )

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
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                tf.keras.layers.Conv2DTranspose(
                    filters=planes,
                    kernel_size=kernel,
                    strides=2,
                    padding="same",
                    use_bias=self.deconv_with_bias,
                    kernel_initializer=initializer))
            layers.append(tf.keras.layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=1e-5))
            layers.append(tf.keras.layers.ReLU())
            self.inplanes = planes

        return tf.keras.Sequential(layers)

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

    def call_bk(self, x, **kwargs):
        x = self.features(x)
        # x = x.mean(3).mean(2)
        # print("x.shape:{}".format(x.shape))
        # x = self.classifier(x)
        # print("x.shape:{}".format(x.shape))
        # x = self.bn(x)
        x = self.deconv_layers(x)  # torch.Size([32, 2048, 8, 6])->torch.Size([32, 256, 64, 48])
        x = self.final_layer(x)  # torch.Size([32, 256, 64, 48])->torch.Size([32, 17, 64, 48])
        return x

    def call(self, x, **kwargs):
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


def get_pose_net(cfg, is_train, **kwargs):
    model = MobileNetV2(cfg, **kwargs)
    # if is_train and os.path.isfile(cfg.MODEL.PRETRAINED):
    #     print('=> loading pretrained model {}'.format(cfg.MODEL.PRETRAINED))
    #     state_dict = torch.load(cfg.MODEL.PRETRAINED)
    #     model.load_state_dict(state_dict, strict=False)
    # else:
    #     print("no pretrained")
    # print("no pretrained")
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
    model.build(input_shape=tuple(input_shape))
    # inputs = tf.keras.layers.Input(shape=(input_size[1], input_size[0], 3), dtype=tf.float32)
    # output = model.build(input_shape=input_shape)
    data = tf.random.uniform((1, 256, 192, 3))
    tf_tools.plot_model(model, input_shape, plot=True)
    out = model(data)
    print(out.shape)
    model_file = "tmp_model.h5"
    model.save(filepath=model_file, include_optimizer=False)


def demo_for_v2():
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
    model.build(input_shape=tuple(input_shape))
    # inputs = tf.keras.layers.Input(shape=(input_size[1], input_size[0], 3), dtype=tf.float32)
    # output = model.build(input_shape=input_shape)
    # data = tf.random.uniform((1, 256, 192, 3))
    # tf_tools.print_model_summary(model, input_shape, plot=True)
    # out = model(data)
    # print(out.shape)
    config = model.get_config()
    # fresh_model = Linear.from_config(config)
    # json_str = model.to_json()
    fresh_model = tf.keras.models.model_from_config(config)
    model_file = "tmp_model.h5"
    fresh_model.save(filepath=model_file, include_optimizer=False)


if __name__ == "__main__":
    from utils import tf_tools

    # demo_for_v1()
    demo_for_v2()
