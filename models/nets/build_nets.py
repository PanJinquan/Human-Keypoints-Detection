# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-Human-Pose-Estimation-Pipeline
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-10-09 09:51:33
# --------------------------------------------------------
"""
import models.nets as nets


def build_nets(net_type, config, is_train=True):
    """
    :param net_type:
    :param config:
    :param is_train:
    :return:
    """
    # cmd_model = 'pretrained.' + config.MODEL.NAME + '.get_pose_net'
    # model = eval('pretrained.'+config.MODEL.NAME+'.get_pose_net')(config, is_train=True )
    if net_type == "pose_resnet":
        model = nets.pose_resnet.get_pose_net(config, is_train=is_train)
    elif net_type == "pose_hrnet":
        model = nets.pose_hrnet.get_pose_net(config, is_train=is_train)
    elif net_type == "pose_resnetst":
        model = nets.pose_resnetst.get_pose_net(config, is_train=is_train)
    elif net_type == "mobilenet_v2":
        width_mult = config.MODEL.EXTRA.WIDTH_MULT
        model = nets.mobilenet_v2.get_pose_net(config, is_train=is_train, width_mult=width_mult)
    elif net_type == "ir_mobilenet_v2":
        width_mult = config.MODEL.EXTRA.WIDTH_MULT
        model = nets.ir_mobilenet_v2.get_pose_net(config, is_train=is_train, width_mult=width_mult)
    elif net_type == "model_ir_mobilenet_v2":
        width_mult = config.MODEL.EXTRA.WIDTH_MULT
        model = nets.model_ir_mobilenet_v2.get_pose_net(config, is_train=is_train, width_mult=width_mult)
    elif net_type == "model_mobilenet_v2":
        width_mult = config.MODEL.EXTRA.WIDTH_MULT
        model = nets.model_mobilenet_v2.get_pose_net(config, is_train=is_train, width_mult=width_mult)
    elif net_type == "model_mobilenet_v3_large":
        width_mult = config.MODEL.EXTRA.WIDTH_MULT
        model = nets.model_mobilenet_v3_large.get_pose_net(config, is_train=is_train, width_mult=width_mult)
    elif net_type == "model_mobilenet_v3_small":
        width_mult = config.MODEL.EXTRA.WIDTH_MULT
        model = nets.model_mobilenet_v3_small.get_pose_net(config, is_train=is_train, width_mult=width_mult)
    else:
        raise Exception("Error:{}".format(config.MODEL.NAME))
    return model


if __name__ == "__main__":
    """
    IMAGE_SIZE = [256, 256]
    NUM_DECONV_FILTERS = [16, 16, 16]
    Total params: 526,404
    Total memory: 49.06MB
    Total MAdd: 236.66MMAdd
    Total Flops: 114.67MFlops
    Total MemR+W: 100.19MB
    ---------------------------------------------------------------------------
    IMAGE_SIZE = [288, 288]
    NUM_DECONV_FILTERS = [16, 16, 16]
    Total memory: 62.09MB
    Total MAdd: 299.52MMAdd
    Total Flops: 145.13MFlops
    Total MemR+W: 126.32MB
    """
    import numpy as np
    import torch
    from utils import torch_tools
    from configs import val_config
    from easydict import EasyDict as edict

    # config = val_config.custom_coco_finger_res18_192_256
    config = val_config.custom_coco_finger4_model_mbv2_192_256
    # config = val_config.custom_coco_finger4_model_pose_resnetst_256_256
    config = edict(config)
    config.MODEL.EXTRA.WIDTH_MULT = 0.5
    config.MODEL.IMAGE_SIZE = [288, 192]
    config.MODEL.EXTRA.NUM_DECONV_FILTERS = [16, 16, 16]

    # config.MODEL.NAME = "pose_resnet"
    config.MODEL.NAME = "model_mobilenet_v2"
    config.MODEL.EXTRA.NUM_LAYERS = 18
    config.MODEL.PRETRAINED = True
    device = "cpu"
    input_size = tuple(config.MODEL.IMAGE_SIZE)  # h,w
    net_type = config.MODEL.NAME
    model = build_nets(net_type=net_type, config=config, is_train=False)
    model_path = [net_type,
                  str(config.MODEL.EXTRA.WIDTH_MULT),
                  str(config.MODEL.IMAGE_SIZE[0]),
                  str(config.MODEL.EXTRA.NUM_DECONV_FILTERS[0])]
    model_path = "_".join(model_path) + ".pth"
    torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch_tools.load_state_dict(model_path, module=False))
    model = model.to(device)
    model.eval()
    torch_tools.summary_model(model, batch_size=1, input_size=input_size, device=device)
