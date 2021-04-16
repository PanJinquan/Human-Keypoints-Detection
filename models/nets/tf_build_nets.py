# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-Human-Pose-Estimation-Pipeline
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-10-09 09:51:33
# --------------------------------------------------------
"""
from models.nets import tf_pose_resnet, tf_model_mobilenet_v2
import tensorflow as tf
import tensorflow.keras.applications


def build_nets(net_type, config, is_train=True):
    """
    :param net_type:
    :param config:
    :param is_train:
    :return:
    """
    if net_type == "tf_pose_resnet":
        model = tf_pose_resnet.get_pose_net(config, is_train=is_train)
    elif net_type == "tf_model_mobilenet_v2":
        width_mult = config.MODEL.EXTRA.WIDTH_MULT
        model = tf_model_mobilenet_v2.get_pose_net(config, is_train=is_train, width_mult=width_mult)
    else:
        raise Exception("Error:{}".format(net_type))
    return model


if __name__ == "__main__":
    """
    
    
    """
    import numpy as np
    import torch
    from utils import tf_tools
    from configs import val_config
    from easydict import EasyDict as edict
    # tf.enable_eager_execution()
    # tf.compat.v1.enable_eager_execution()
    tf_tools.set_device_memory(eager_execution=False)
    # config = val_config.custom_coco_finger_res18_192_256
    config = val_config.custom_coco_finger4_model_mbv2_192_256
    config = edict(config)
    config.MODEL.IMAGE_SIZE = [192, 192]
    config.MODEL.EXTRA.NUM_DECONV_FILTERS = [64, 64, 64]
    config.MODEL.EXTRA.WIDTH_MULT = 0.5
    # config.MODEL.NAME = "pose_resnet"
    config.MODEL.NAME = "model_mobilenet_v2"
    config.MODEL.EXTRA.NUM_LAYERS = 18
    config.MODEL.PRETRAINED = True
    input_size = tuple(config.MODEL.IMAGE_SIZE)  # h,w
    input_shape = [1, input_size[1], input_size[0], 3]
    device = "cpu"
    net_type = "tf_model_mobilenet_v2"
    model = build_nets(net_type=net_type, config=config, is_train=False)
    # tf_tools.summary_model(model, input_size=input_size, plot=True)
    tf_tools.plot_model(model, input_shape)
