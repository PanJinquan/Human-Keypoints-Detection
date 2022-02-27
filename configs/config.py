# -*-coding: utf-8 -*-
"""
    @Project: human-pose-estimation.pytorch
    @File   : config_tmp.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-29 11:13:42
"""
import numpy as np

coco_256_192 = {
    'MODEL': {
        'NAME': 'pose_resnet',
        # 'NAME': 'mobilenet_v2',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 17,
        'IMAGE_SIZE': np.array([192, 256]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  'HEATMAP_SIZE': np.array([48, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [256, 256, 256],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 50
                  # 'WIDTH_MULT': 1.0
                  },
        'STYLE': 'pytorch'},
    'TEST': {
        'BATCH_SIZE': 32,
        'FLIP_TEST': False,
        'POST_PROCESS': True,
        'SHIFT_HEATMAP': True,
        'USE_GT_BBOX': True,
        'OKS_THRE': 0.9,
        'IN_VIS_THRE': 0.2,
        'BBOX_THRE': 1.0,
        'MODEL_FILE': '/data/pretrained/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar',
        # 'MODEL_FILE': '/data/pretrained/pytorch/pose_coco/mobilenet_v2_1.0/model_best.pth.tar',
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                     [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
    }
}

mpii_256_256 = {
    'MODEL': {
        'NAME': 'pose_resnet',
        # 'NAME': 'mobilenet_v2',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 16,
        'IMAGE_SIZE': np.array([256, 256]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  'HEATMAP_SIZE': np.array([64, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [256, 256, 256],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 50
                  # 'WIDTH_MULT': 1.0
                  },
        'STYLE': 'pytorch'},
    'TEST': {
        'BATCH_SIZE': 32,
        'FLIP_TEST': False,
        'POST_PROCESS': True,
        'SHIFT_HEATMAP': True,
        'USE_GT_BBOX': True,
        'OKS_THRE': 0.9,
        'IN_VIS_THRE': 0.2,
        'BBOX_THRE': 1.0,
        # 'MODEL_FILE': 'data/pretrained/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar',
        'MODEL_FILE': 'data/pretrained/pytorch/pose_resnet_50/model_best_256_256.pth.tar',
        # 'MODEL_FILE': 'data/pretrained/pytorch/pose_mpii/mobilenet_v2_1.0/model_best.pth.tar',
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [[0, 1], [1, 2], [3, 4], [4, 5], [2, 6], [6, 3], [12, 11], [7, 12],
                     [11, 10], [13, 14], [14, 15], [8, 9], [8, 7], [6, 7], [7, 13]],
        # "skeleton": [[2, 6], [6, 3], [12, 11], [7, 12],
        #              [11, 10], [13, 14], [14, 15], [8, 9], [8, 7], [6, 7], [7, 13]]

    }

}

mpii_384_384 = {
    'MODEL': {
        'NAME': 'pose_resnet',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 16,
        'IMAGE_SIZE': np.array([384, 384]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  'HEATMAP_SIZE': np.array([96, 96]),
                  'SIGMA': 3,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [256, 256, 256],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 50
                  # 'WIDTH_MULT': 1.0
                  },
        'STYLE': 'pytorch'},
    'TEST': {
        'BATCH_SIZE': 32,
        'FLIP_TEST': False,
        'POST_PROCESS': True,
        'SHIFT_HEATMAP': True,
        'USE_GT_BBOX': True,
        'OKS_THRE': 0.9,
        'IN_VIS_THRE': 0.2,
        'BBOX_THRE': 1.0,
        'MODEL_FILE': './pretrained/pytorch/pose_mpii/pose_resnet_50_384x384.pth.tar',
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [[0, 1], [1, 2], [3, 4], [4, 5], [2, 6], [6, 3], [12, 11], [7, 12],
                     [11, 10], [13, 14], [14, 15], [8, 9], [8, 7], [6, 7], [7, 13]]}
}

custom_mpii_256_256 = {
    'MODEL': {
        # 'NAME': 'pose_resnet',
        'NAME': 'mobilenet_v2',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 6,
        'IMAGE_SIZE': np.array([256, 256]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  'HEATMAP_SIZE': np.array([64, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [256, 256, 256],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 50
                  # 'WIDTH_MULT': 1.0
                  },
        'STYLE': 'pytorch'},
    'TEST': {
        'BATCH_SIZE': 32,
        'FLIP_TEST': False,
        'POST_PROCESS': True,
        'SHIFT_HEATMAP': True,
        'USE_GT_BBOX': True,
        'OKS_THRE': 0.9,
        'IN_VIS_THRE': 0.2,
        'BBOX_THRE': 1.0,
        'MODEL_FILE': './data/pretrained/pytorch/custom_mpii/mobilenet_v2_1.0_256x256_2019-11-06-14-57/models/model_best.pth.tar',
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [[0, 1], [1, 2], [2, 3], [4, 1], [1, 5]]}
}

student_mpii_256_256 = {
    'MODEL': {
        'NAME': 'pose_resnet',
        # 'NAME': 'mobilenet_v2',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 10,
        'IMAGE_SIZE': np.array([256, 256]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  'HEATMAP_SIZE': np.array([64, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [256, 256, 256],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 50
                  # 'WIDTH_MULT': 1.0
                  },
        'STYLE': 'pytorch'},
    'TEST': {
        'BATCH_SIZE': 32,
        'FLIP_TEST': False,
        'POST_PROCESS': True,
        'SHIFT_HEATMAP': True,
        'USE_GT_BBOX': True,
        'OKS_THRE': 0.9,
        'IN_VIS_THRE': 0.2,
        'BBOX_THRE': 1.0,
        # 'MODEL_FILE': 'data/pretrained/pytorch/student_mpii/pose_resnet_50_256x256_2019-12-17-14-14/models/model_best.pth.tar',
        'MODEL_FILE': 'data/pretrained/pytorch/student_mpii/pose_resnet_50_256x256_center_up_alpha-0.1_beta1.0/models/model_best.pth.tar',
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 1], [7, 1], [7, 8], [8, 9]]
    }
}

student_mpii_256_256_v2 = {
    'MODEL': {
        # 'NAME': 'pose_resnet',
        'NAME': 'mobilenet_v2',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 9,
        'IMAGE_SIZE': np.array([256, 256]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  'HEATMAP_SIZE': np.array([64, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [256, 256, 256],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 50
                  # 'WIDTH_MULT': 1.0
                  },
        'STYLE': 'pytorch'},
    'TEST': {
        'BATCH_SIZE': 32,
        'FLIP_TEST': False,
        'POST_PROCESS': True,
        'SHIFT_HEATMAP': True,
        'USE_GT_BBOX': True,
        'OKS_THRE': 0.9,
        'IN_VIS_THRE': 0.2,
        'BBOX_THRE': 1.0,
        'MODEL_FILE': './pretrained/pytorch/student_mpii/mobilenet_v2_1.0_256x256_2019-11-18-14-53/pretrained/model_best.pth.tar',
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [[0, 1], [1, 2], [0, 5], [5, 4], [4, 3], [0, 6], [6, 7], [7, 8]]
        # "skeleton": []
    }
}
