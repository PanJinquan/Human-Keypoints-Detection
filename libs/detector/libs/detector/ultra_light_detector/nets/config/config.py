# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: Ultra-Light-Fast-Generic-Face-Detector-1MB
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-07-02 10:20:19
# --------------------------------------------------------
"""
import numpy as np

face_config = {
    "image_mean_test": np.array([127, 127, 127]),
    "image_mean": np.array([127, 127, 127]),
    "image_std": 128.0,
    "iou_threshold": 0.3,
    "center_variance": 0.1,
    "size_variance": 0.2,
    "min_boxes": [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],  # for Face
    "aspect_ratios": [[1.0, 1.0]],
    "shrinkage": [8, 16, 32, 64],
    "class_names": ['BACKGROUND', 'face'],
    'loc_weight': 2.0,
    'cla_weight': 1.0,
    'landm_weight': 1.0,
}

fingernail_config = {
    "image_mean_test": np.array([127, 127, 127]),
    "image_mean": np.array([127, 127, 127]),
    "image_std": 128.0,
    "iou_threshold": 0.3,
    "center_variance": 0.1,
    "size_variance": 0.2,
    "min_boxes": [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],  # for Face
    "aspect_ratios": [[1.0, 1.0]],
    "shrinkage": [8, 16, 32, 64],
    "class_names": ['BACKGROUND', 'fingernail'],
}

face_person_config = {
    "image_mean_test": np.array([127, 127, 127]),
    "image_mean": np.array([127, 127, 127]),
    "image_std": 128.0,
    "iou_threshold": 0.3,
    "center_variance": 0.1,
    "size_variance": 0.2,
    "min_boxes": [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],  # for Face
    "aspect_ratios": [[1.0, 1.0], [1.2, 1.5], [1.0, 2.0]],
    "shrinkage": [8, 16, 32, 64],
    "class_names": ['BACKGROUND', 'face', 'person'],
    'loc_weight': 2.0,
    'cla_weight': 1.0,
    'landm_weight': 1.0,
}

face_body_config = {
    "image_mean_test": np.array([127, 127, 127]),
    "image_mean": np.array([127, 127, 127]),
    "image_std": 128.0,
    "iou_threshold": 0.3,
    "center_variance": 0.1,
    "size_variance": 0.2,
    "min_boxes": [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],  # for Face
    "aspect_ratios": [[1.0, 1.0], [1.2, 1.5], [1.0, 2.0]],
    "shrinkage": [8, 16, 32, 64],
    "class_names": ["BACKGROUND", 'face', "body"],  # (1920 1080),(960,540)
    'loc_weight': 2.0,
    'cla_weight': 1.0,
    'landm_weight': 1.0,
}

person_config = {
    "image_mean": np.array([127, 127, 127]),
    "image_std": 128.0,
    "iou_threshold": 0.3,
    "center_variance": 0.1,
    "size_variance": 0.2,
    "min_boxes": [[32], [48], [64, 96], [128, 192, 256]],  # for Person
    "aspect_ratios": [[1.0, 1.0], [1.2, 1.5], [1.0, 2.0]],
    "shrinkage": [8, 16, 32, 64],
    "class_names": ['BACKGROUND', 'person'],
    'loc_weight': 2.0,
    'cla_weight': 1.0,
    'landm_weight': 1.0,
}

person_pig_config = {
    "image_mean": np.array([127, 127, 127]),
    "image_std": 128.0,
    "iou_threshold": 0.3,
    "center_variance": 0.1,
    "size_variance": 0.2,
    "min_boxes": [[32], [48], [64, 96], [128, 192, 256]],  # for Person
    "aspect_ratios": [[1.0, 1.0], [1.2, 1.5], [1.0, 2.0], [1.5, 1.2], [2.0, 1.0]],
    "shrinkage": [8, 16, 32, 64],
    "class_names": ['BACKGROUND', 'person', "pig"],
    'loc_weight': 2.0,
    'cla_weight': 1.0,
    'landm_weight': 1.0,
}

custom_config = {
    "image_mean": np.array([127, 127, 127]),
    "image_std": 128.0,
    "iou_threshold": 0.3,
    "center_variance": 0.1,
    "size_variance": 0.2,
    "min_boxes": [[32], [48], [64, 96], [128, 192, 256]],  # for Person
    "aspect_ratios": [[1.0, 1.0], [1.2, 1.5], [1.0, 2.0]],
    "shrinkage": [8, 16, 32, 64],
    "class_names": {'BACKGROUND': 0, 'person': 1, 'person_up': 1, 'person_down': 1},
    'loc_weight': 2.0,
    'cla_weight': 1.0,
    'landm_weight': 1.0,
}
