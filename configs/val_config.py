# -*-coding: utf-8 -*-
"""
    @Project: human-pose-estimation.pytorch
    @File   : config_tmp.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-29 11:13:42
"""
import numpy as np

coco_res50_192_256 = {
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
        'MODEL_FILE': 'data/pretrained/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar',
        # 'MODEL_FILE': 'data/pretrained/pytorch/pose_coco/mobilenet_v2_1.0/model_best.pth.tar',
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                     [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
    }
}

coco_res18_192_256 = {
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
                  'NUM_LAYERS': 18
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
        'MODEL_FILE': './data/pretrained/pytorch/pose_coco/pose_resnet_18_256x192_2020-02-20-14-52/pretrained/model_best.pth.tar',
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                     [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
    }
}

custom_coco_person_res18_192_256 = {
    'MODEL': {
        'NAME': 'pose_resnet',
        # 'NAME': 'mobilenet_v2',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 11,
        'IMAGE_SIZE': np.array([192, 256]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  'HEATMAP_SIZE': np.array([48, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [256, 256, 256],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 18
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
        'MODEL_FILE': 'output/custom_coco/pose_resnet_18_256x192_2020-09-25-11-37/pretrained/model_best.pth.tar',
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [(0, 1), (0, 2), (3, 4), (4, 6), (6, 8), (3, 5), (5, 7), (4, 10), (3, 9)]

    }
}

custom_coco_finger_res18_192_256 = {
    'MODEL': {
        'NAME': 'pose_resnet',
        # 'NAME': 'mobilenet_v2',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 10,
        'IMAGE_SIZE': np.array([192, 256]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  'HEATMAP_SIZE': np.array([48, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [256, 256, 256],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 18
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
        'MODEL_FILE': 'work_space/custom_coco/pose_resnet_18_256x192_0.001_finger_pretrained_2020-10-14-15-22/model/best_model_149_0.8970.pth',
        # 'MODEL_FILE': 'work_space/custom_coco/pose_resnet_18_256x192_finger_2020-10-09-09-47/model/best_model_008_0.7637.pth',
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    }
}

custom_coco_finger_model_mbv2_192_256 = {
    'MODEL': {
        # 'NAME': 'pose_resnet',
        'NAME': 'model_mobilenet_v2',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 10,
        'IMAGE_SIZE': np.array([192, 256]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  'HEATMAP_SIZE': np.array([48, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [256, 256, 256],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 18,
                  'WIDTH_MULT': 1.0
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
        'MODEL_FILE': 'work_space/custom_coco/model_mobilenet_v2_1.0_256x192_finger_kaiming_normal_2020-10-10-16-35/model/best_model_144_0.8165.pth',
        # 'MODEL_FILE': 'work_space/custom_coco/pose_resnet_18_256x192_finger_2020-10-09-09-47/model/best_model_008_0.7637.pth',
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    }
}

custom_coco_finger4_model_mbv2_192_256 = {
    'MODEL': {
        # 'NAME': 'pose_resnet',
        'NAME': 'model_mobilenet_v2',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 4,
        'PRETRAINED': "",
        'IMAGE_SIZE': np.array([192, 256]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  'HEATMAP_SIZE': np.array([48, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [256, 256, 256],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 18,
                  'WIDTH_MULT': 0.5
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
        'MODEL_FILE': 'work_space/finger4/custom_coco/model_mobilenet_v2_1.0_256x192_0.001_finger_2020-10-16-17-32/model/best_model_140_0.9306.pth',
        # 'MODEL_FILE': 'work_space/finger4/custom_coco/model_mobilenet_v2_1.0_256x192_0.001_finger_flip_pairs_2020-10-20-11-57/model/best_model_128_0.9251.pth',
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [(0, 1), (2, 3)]

    }
}

custom_coco_finger4_model_mbv2_256_256 = {
    'MODEL': {
        # 'NAME': 'pose_resnet',
        'NAME': 'model_mobilenet_v2',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 4,
        'IMAGE_SIZE': np.array([288, 244]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  # 'EXTRA': {'TARGET_TYPE': 'offset',
                  'HEATMAP_SIZE': np.array([64, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [16, 16, 16],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 18,
                  'WIDTH_MULT': 0.5
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
        # 'MODEL_FILE': "/home/dm/data3/models/finger-keypoints/finger_pen_v3/model_mobilenet_v2_0.5_4_256x256_16_gaussian_finger_tiny_scale1.0_2021-01-22-10-51/model/model_model_mobilenet_v2_191_0.9386.pth",
        # "MODEL_FILE":"/home/dm/data3/models/finger-keypoints/finger_pen_v3/model_mobilenet_v2_0.5_4_256x256_16_gaussian_finger_tiny_scale1.0_2021-01-25-22-32/model/best_model_164_0.9478.pth",
        # "MODEL_FILE":"/home/dm/data3/models/finger-keypoints/finger_pen_v3/model_mobilenet_v2_0.5_4_256x256_16_offset_finger_tiny_scale1.0_2021-02-01-15-00/model/best_model_189_0.9455.pth",
        # "MODEL_FILE": "/home/dm/data3/models/finger-keypoints/pen_v5/model_mobilenet_v2_0.5_4_256x256_16_gaussian_finger_tiny_scale1.0_2021-02-06-14-29/model/best_model_150_0.9390.pth",
        # "MODEL_FILE":"/home/dm/data3/models/finger-keypoints/pen_v5/model_mobilenet_v2_0.5_4_256x256_32_offset_finger_tiny_scale1.0_2021-02-06-15-43/model/best_model_141_0.9357.pth",
        # "MODEL_FILE": "/home/dm/data3/models/finger-keypoints/pen_v5/model_mobilenet_v2_0.5_4_256x256_16_gaussian_finger_tiny_scale1.0_2021-02-06-14-29/model/best_model_186_0.9398.pth",
        "MODEL_FILE": "/home/dm/data3/models/finger-keypoints/pen_v8/model_mobilenet_v2_0.5_4_288x224_16_gaussian_rot50_scale1.0_2021-03-12-14-05/model/best_model_162_0.9635.pth",
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [(0, 1), (2, 3)]
    }
}

custom_coco_finger4_model_mbv2_192_192 = {
    'MODEL': {
        # 'NAME': 'pose_resnet',
        'NAME': 'model_mobilenet_v2',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 4,
        'IMAGE_SIZE': np.array([192, 192]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  'HEATMAP_SIZE': np.array([48, 48]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [64, 64, 64],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 18,
                  'WIDTH_MULT': 0.5
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
        'MODEL_FILE': 'work_space/finger4_tiny/custom_coco/model_mobilenet_v2_0.5_192x192_0.001_adam_gaussian_finger4_tiny_2020-12-14-16-47/model/best_model_137_0.8743.pth',
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [(0, 1), (2, 3)]

    }
}

custom_coco_finger4_model_pose_resnetst_256_256 = {
    'MODEL': {
        # 'NAME': 'pose_resnet',
        'NAME': 'pose_resnetst',
        'PRETRAINED': "",
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 4,
        'IMAGE_SIZE': np.array([256, 256]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  'HEATMAP_SIZE': np.array([64, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [64, 64, 64],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 50,
                  'WIDTH_MULT': 1.0
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
        # 'MODEL_FILE': '/home/dm/panjinquan3/models/finger-keypoints/finger4-v1/pose_resnetst_50_256x256_0.001_adam_finger_pretrained_2020-10-22-11-38/model/best_model_174_0.9853.pth',
        # 'MODEL_FILE': '/data3/panjinquan/models/finger-keypoints/finger4-v1/pose_resnetst_50_256x256_0.001_adam_finger_pretrained_2020-10-22-11-38/model/best_model_174_0.9853.pth',
        'MODEL_FILE': "/home/dm/panjinquan3/models/finger-keypoints/finger_pen/pose_resnetst_50_4_256x256_64_gaussian_finger_tiny_scale1.0_2020-12-21-09-27/model/best_model_187_0.9885.pth",
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [(0, 1), (2, 3)]

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

body_mpii_192_256 = {
    'MODEL': {
        'NAME': 'mobilenet_v2',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 6,
        'IMAGE_SIZE': np.array([192, 256]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  'HEATMAP_SIZE': np.array([48, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [16, 16, 16],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 50,
                  'WIDTH_MULT': 1.0
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
        'MODEL_FILE': 'data/pretrained/body_mpii/best_model_193_93.1101.pth',
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [[0, 1], [1, 2], [2, 3], [4, 1], [1, 5]]
    }
}

body_coco_192_256 = {
    'MODEL': {
        # 'NAME': 'pose_resnet',
        'NAME': 'model_mobilenet_v2',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 11,
        'IMAGE_SIZE': np.array([192, 256]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  # 'EXTRA': {'TARGET_TYPE': 'offset',
                  'HEATMAP_SIZE': np.array([48, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [128, 128, 128],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 18,
                  'WIDTH_MULT': 1.0
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
        "MODEL_FILE": "data/pretrained/body_coco/best_model_182_0.6597.pth",
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [(0, 1), (0, 2), (3, 4), (4, 6), (6, 8), (3, 5), (5, 7), (4, 10), (3, 9)]
    }
}

pig_coco_model_mobilenet_v2 = {
    'MODEL': {
        # 'NAME': 'pose_resnet',
        'NAME': 'model_mobilenet_v2',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 4,
        'IMAGE_SIZE': np.array([256, 256]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  # 'EXTRA': {'TARGET_TYPE': 'offset',
                  'HEATMAP_SIZE': np.array([64, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [16, 16, 16],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 18,
                  'WIDTH_MULT': 1.0
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
        "MODEL_FILE": "/home/dm/data3/Pose/torch-tf-Keypoint-Estimation-Pipeline/work_space/pig/custom_coco/model_mobilenet_v2_1.0_4_256x256_16_gaussian_pig_scale1.2_2021-03-18-20-51/model/best_model_147_0.3532.pth",
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [(0, 1), (1, 2), (2, 3)]
    }
}

person_coco_192_256 = {
    'MODEL': {
        # 'NAME': 'pose_resnet',
        'NAME': 'model_mobilenet_v2',
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 17,
        'IMAGE_SIZE': np.array([192, 256]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  # 'EXTRA': {'TARGET_TYPE': 'offset',
                  'HEATMAP_SIZE': np.array([48, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [128, 128, 128],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 18,
                  'WIDTH_MULT': 1.0
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
        "MODEL_FILE": "data/pretrained/person_coco/best_model_178_0.6272.pth",
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                     [6, 8], [7, 9], [8, 10], [0, 1], [0, 2], [1, 3], [2, 4]]
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

custom_coco_finger4_model_pose_hrnet_256_256 = {
    'MODEL': {
        'NAME': 'pose_hrnet',
        'PRETRAINED': "",
        'INIT_WEIGHTS': True,
        'NUM_JOINTS': 4,
        'IMAGE_SIZE': np.array([256, 256]),
        'EXTRA': {'TARGET_TYPE': 'gaussian',
                  'HEATMAP_SIZE': np.array([64, 64]),
                  'SIGMA': 2,
                  'FINAL_CONV_KERNEL': 1,
                  'DECONV_WITH_BIAS': False,
                  'NUM_DECONV_LAYERS': 3,
                  'NUM_DECONV_FILTERS': [256, 256, 256],
                  'NUM_DECONV_KERNELS': [4, 4, 4],
                  'NUM_LAYERS': 50,
                  'WIDTH_MULT': 1.0,
                  "NUM_LAYERS": 48,
                  "PRETRAINED_LAYERS": ['conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1',
                                        'stage2', 'transition2', 'stage3', 'transition3', 'stage4'],
                  "STAGE2": {
                      "NUM_MODULES": 1,
                      "NUM_BRANCHES": 2,
                      "BLOCK": "BASIC",
                      "NUM_BLOCKS": [4, 4],
                      "NUM_CHANNELS": [48, 96],
                      "FUSE_METHOD": "SUM"},
                  "STAGE3": {
                      "NUM_MODULES": 4,
                      "NUM_BRANCHES": 3,
                      "BLOCK": "BASIC",
                      "NUM_BLOCKS": [4, 4, 4],
                      "NUM_CHANNELS": [48, 96, 192],
                      "FUSE_METHOD": "SUM"},
                  "STAGE4": {
                      "NUM_MODULES": 3,
                      "NUM_BRANCHES": 4,
                      "BLOCK": "BASIC",
                      "NUM_BLOCKS": [4, 4, 4, 4],
                      "NUM_CHANNELS": [48, 96, 192, 384],
                      "FUSE_METHOD": "SUM"}
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
        # "MODEL_FILE": "/home/dm/panjinquan3/models/finger-keypoints/finger4-transform/pose_hrnet_48_256x256_0.001_adam_finger_transforms_v1_2020-10-29-17-20/model/best_model_098_0.9885.pth",
        "MODEL_FILE": "/home/dm/panjinquan3/models/finger-keypoints/finger4-transform/pose_hrnet_48_256x256_0.001_adam_gaussian_finger_transforms_v2_2020-11-03-10-28/model/model_pose_hrnet_195_0.9881.pth",
        'IMAGE_THRE': 0.0,
        'NMS_THRE': 1.0,
        "skeleton": [(0, 1), (2, 3)]
    }
}
