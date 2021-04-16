# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-01-26 18:45:45
"""
import numpy as np
import math


def get_warpmatrix(theta, size_input, size_dst, size_target):
    '''

    :param theta: angle
    :param size_input:[w,h]
    :param size_dst: [w,h]
    :param size_target: [w,h]/200.0
    :return:
    '''
    size_target = size_target * 200.0
    theta = theta / 180.0 * math.pi
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_target[0] / size_dst[0]
    scale_y = size_target[1] / size_dst[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = math.sin(theta) * scale_y
    matrix[0, 2] = -0.5 * size_target[0] * math.cos(theta) - 0.5 * size_target[1] * math.sin(theta) + 0.5 * size_input[
        0]
    matrix[1, 0] = -math.sin(theta) * scale_x
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = 0.5 * size_target[0] * math.sin(theta) - 0.5 * size_target[1] * math.cos(theta) + 0.5 * size_input[1]
    return matrix


def rotate_points(src_points, angle, c, dst_img_shape, size_target, do_clip=True):
    # src_points: (num_points, 2)
    # img_shape: [h, w, c]
    size_target = size_target * 200.0
    src_img_center = c
    scale_x = (dst_img_shape[0] - 1.0) / size_target[0]
    scale_y = (dst_img_shape[1] - 1.0) / size_target[1]
    radian = angle / 180.0 * math.pi
    radian_sin = -math.sin(radian)
    radian_cos = math.cos(radian)
    dst_points = np.zeros(src_points.shape, dtype=src_points.dtype)
    src_x = src_points[:, 0] - src_img_center[0]
    src_y = src_points[:, 1] - src_img_center[1]
    dst_points[:, 0] = radian_cos * src_x + radian_sin * src_y
    dst_points[:, 1] = -radian_sin * src_x + radian_cos * src_y
    dst_points[:, 0] += size_target[0] * 0.5
    dst_points[:, 1] += size_target[1] * 0.5
    dst_points[:, 0] *= scale_x
    dst_points[:, 1] *= scale_y
    if do_clip:
        dst_points[:, 0] = np.clip(dst_points[:, 0], 0, dst_img_shape[1] - 1)
        dst_points[:, 1] = np.clip(dst_points[:, 1], 0, dst_img_shape[0] - 1)
    return dst_points
