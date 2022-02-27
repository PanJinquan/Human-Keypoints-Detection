# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-anti-spoofing-pipeline
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-06-02 16:00:47
# --------------------------------------------------------
"""
import torch
import random
import os
import numpy as np
from collections import OrderedDict
from collections.abc import Iterable


def set_env_random_seed(seed=2020):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def print_model(model):
    """
    :param model:
    :return:
    """
    for k, v in model.named_parameters():
        # print(k,v)
        print(k)


def freeze_net_layers(net):
    """
    https://www.zhihu.com/question/311095447/answer/589307812
    example:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
    :param net:
    :return:
    """
    # for param in net.parameters():
    #     param.requires_grad = False
    for name, child in net.named_children():
        # print(name, child)
        for param in child.parameters():
            param.requires_grad = False


def load_state_dict(model_path, module=True):
    """
    Usage:
        model=Model()
        state_dict = torch_tools.load_state_dict(model_path, module=False)
        model.load_state_dict(state_dict)
    :param model_path:
    :param module:
    :return:
    """
    state_dict = None
    if model_path:
        print('=> loading model from {}'.format(model_path))
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        if module:
            state_dict = get_module_state_dict(state_dict)
    else:
        print("Error:no model file:{}".format(model_path))
        exit(0)
    return state_dict


def get_module_state_dict(state_dict):
    """
    :param state_dict:
    :return:
    """
    # 初始化一个空 dict
    new_state_dict = OrderedDict()
    # 修改 key，没有module字段则需要不上，如果有，则需要修改为 module.features
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        new_state_dict[k] = v
    return new_state_dict


def summary_model(model, batch_size=1, input_size=[112, 112], device="cpu"):
    """
    :param model:
    :param batch_size:
    :param input_size:
    :param device:
    :return:
    """
    from torchsummary import summary
    from torchstat import stat
    inputs = torch.randn(size=(batch_size, 3, input_size[0], input_size[1]))
    inputs = inputs.to(device)
    model = model.to(device)
    model.eval()
    output = model(inputs)
    summary(model, input_size=(3, input_size[0], input_size[1]), batch_size=batch_size, device=device)
    stat(model, (3, input_size[0], input_size[1]))
    print("===" * 10)
    print("inputs.shape:{}".format(inputs.shape))
    print("output.shape:{}".format(output.shape))
