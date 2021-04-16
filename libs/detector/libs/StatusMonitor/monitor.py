# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: StatusMonitor
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-08-13 19:32:01
# --------------------------------------------------------
"""

import os
import numpy as np
import cv2


class Queue():
    """队列"""

    def __init__(self, win_size):
        """
        :param win_size: 队列大小
        """
        self.win_size = win_size
        self.data = []

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return "".join(self.data)

    def put(self, item):
        if len(self.data) == self.win_size:
            self.data.pop(0)
        self.data.append(item)

    def get(self, index=-1):
        return self.data[index]


class StatusMonitor():
    """
    检测视频流状态,计算连续相邻两帧的相似性,
    判断当前状态{"运动":1,"静止":0},
    判断是否需要上传图片
    """

    def __init__(self, win_size=10):
        """
        :param win_size: 记录历史label值的窗口大小
        """
        self.label_queue = Queue(win_size=win_size)  # 定义label队列,用于记录历史窗口的label值

    def get_label_list(self):
        """
        获得label队列数据列表
        :return:
        """
        return self.label_queue.data

    def get_diff_map(self, frame1, frame2):
        """
        计算两帧图片的差异图(两张图的绝对差)
        :param frame1: 图片1
        :param frame2: 图片2
        :return: 差异图
        """
        diff_map = np.abs(frame1 - frame2)
        return diff_map

    @staticmethod
    def cal_diff_map_similarity_l2(diff_map, tok=400):
        """
        计算差异图相似度(距离),
        原理: 将差异图作为特征,按特征值大小进行降序排序,提取tok的差异较大的特征,计算其均方差作为相似距离
        :param diff_map: 差异图
        :return: dist : 相似度(距离)
        """
        diff = np.asarray(diff_map).reshape(-1)
        diff = -np.sort(-diff)  # 降序排序
        diff = diff[:tok]  # 提取tok的差异较大的特征
        dist = np.sum(np.square(diff)) / tok  # 计算均方差作为相似距离
        return dist

    @staticmethod
    def cal_diff_map_similarity_l1(diff_map, tok=400):
        """
        计算差异图相似度(距离),
        原理: 将差异图作为特征,按特征值大小进行降序排序,提取tok的差异较大的特征,计算其平均绝对误差作为相似度(距离)
        :param diff_map: 差异图
        :return: dist : 相似度(距离)
        """
        diff = np.asarray(diff_map).reshape(-1)
        diff = -np.sort(-diff)  # 降序排序
        diff = diff[:tok]  # 提取tok的差异较大的特征
        dist = np.sum(diff) / tok  # 计算平均绝对误差作为相似度(距离)
        return dist

    # @debug.run_time_decorator("get_frame_similarity")
    def get_frame_similarity(self, frame1, frame2, measurement="l1", isshow=False):
        """
        比较两帧图像的相似度(距离)
        :param frame1: 输入frame1图像
        :param frame2: 输入frame2图像
        :param measurement: 相似度(距离)度量方法,l1: 平均绝对误差作为相似度(距离),l2: 均方差作为相似距离
        :param isshow: <bool> 默认为False,是否显示差异图
        :return: <float>相似度(距离)
        """
        diff_map = self.get_diff_map(frame1, frame2)
        if measurement == "l1":
            dist = self.cal_diff_map_similarity_l1(diff_map)
        elif measurement == "l2":
            dist = self.cal_diff_map_similarity_l2(diff_map)
        else:
            return Exception("Error:{}".format(measurement))
        if isshow:
            cv2.imshow("diff_map", diff_map)
        return dist

    def check_upload_status(self, frame, upload_frame, label, up_threshold=0.001, up_flag="1000"):
        """
        判断是否需要上传图片
        思路: 当且仅当前帧图片处于静止状态(label="0"),且历史label中出现上传图片的信号符(flag="1000"),
              并且当前帧图片与上一次上传的图片不相似时(>up_threshold),才需要上传图片
        :param frame: 当前帧图
        :param upload_frame: 上一次上传的图像
        :param label: 当前label: "0"表示静止,"1"表示运动,
        :param up_threshold: 上传图片的相似阈值,当frame与上一次的upload_frame的相似性距离大于该阈值时,则上传图片
                             否则,认为当前帧与上一次上传的图片非常相似,不需要上传
        :param up_flag: 上传图片的信号符: 默认"1000": 表示先出现1次label=1,然后连续3帧的label都是0,则认为需要上传图片,
                     flag="110000": 表示先出现2次label=1,然后连续4帧的label都是0,则认为需要上传图片
        :return: <bool>: True : 表示需要上传图片
                         False: 表示不需要上传图片
        """
        isupload = False
        self.label_queue.put(label)
        if label == "0":  # 如果是静止状态
            label_list = self.label_queue.data
            label_list = "".join(label_list)
            if label_list.count(up_flag) == 1:  # 仅出现一次信号符
                isupload = True
            else:
                isupload = False
            if isupload:
                # 比较当frame与上一次的upload_frame的相似性距离
                dist = self.get_frame_similarity(frame, upload_frame, isshow=False)
                if dist < up_threshold:
                    # 相似图片,无需上传
                    isupload = False
                else:
                    isupload = True
                    # 不相似图片,需上传
                print("up_threshold:{},dist:{:3.8f},isupload:{}".format(up_threshold, dist, isupload))
        return isupload
