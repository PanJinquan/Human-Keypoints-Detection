# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-03-08 17:20:58
"""
from utils import geometry_tools


class Sitting(object):
    def __init__(self):
        origin = [0, 0]
        self.X_coordinates = geometry_tools.create_vector(origin, point2=[1, 0])
        self.Y_coordinates = geometry_tools.create_vector(origin, point2=[0, 1])
        self.spine_base = None  # 脊柱底
        self.thorax = None  # 胸部
        self.neck = None  # 脖子
        self.head_top = None  # 头部
        self.shoulder_left = None
        self.shoulder_right = None

    def update_key_points(self, key_points):
        self.spine_base = key_points[0]
        self.thorax = key_points[1]
        self.neck = key_points[2]
        self.head_top = key_points[3]
        self.shoulder_left = key_points[4]
        self.shoulder_right = key_points[5]

    def __head_vector(self):
        '''head->thorax'''
        v = geometry_tools.create_vector(self.head_top, self.thorax)
        return v

    def head_status(self):
        '''
        pitch:是围绕X轴旋转，也叫做俯仰角，点头 上负下正
        yaw:  是围绕Y轴旋转，也叫偏航角，摇头 左正右负
        roll: 是围绕Z轴旋转，也叫翻滚角，摆头（歪头）左负 右正
        :return:
        '''
        head_vector = self.__head_vector()
        head_roll = geometry_tools.compute_vector_angle(head_vector,
                                                        self.Y_coordinates,
                                                        minangle=False)
        # bias = False if head_vector[0] > 0 else True
        # if bias:
        #     head_roll = -head_roll
        return head_roll

    def get_status(self, key_points, kp_scores):
        self.update_key_points(key_points)
        status = {}
        head_roll = self.head_status()
        label = self.get_label(head_roll)
        status["label"] = label
        status["head_roll"] = head_roll
        return status

    def get_label(self, head_roll, angle_th=15):
        label = head_roll > angle_th
        return label
