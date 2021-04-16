import torch
import torch.nn as nn
import math
import torchvision.models.mobilenet
from models.nets.mobilenetv3 import mobilenetv3


def get_pose_net(cfg, is_train, **kwargs):
    model = mobilenetv3.mobilenetv3_small(pretrained=cfg.MODEL.PRETRAINED, cfg=cfg, **kwargs)
    return model


if __name__ == "__main__":
    """
    Total params: 9,569,552
    -------------------------------------------------------------------------------------------------------------------------------------------------------------
    Total memory: 84.73MB
    Total MAdd: 3.16GMAdd
    Total Flops: 328.18MFlops
    Total MemR+W: 173.22MB

    """
    from models.core.config import config
    from utils import torch_tools

    device = "cuda:0"
    input_size = [256, 256]
    model = get_pose_net(config, is_train=False).to(device)
    input = torch.randn(size=(32, 3, input_size[1], input_size[0]))
    input = input.to(device)
    out = model(input)
    print("out:{}".format(out.shape))
    # torch_tools.summary_model(model, batch_size=1, )
    model.get_layers_features(info=True)
    # torch_tools.summary_model(model, batch_size=1, input_size=input_size)
