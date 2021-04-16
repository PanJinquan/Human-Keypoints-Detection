# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-01-19 19:29:47
"""
import numpy as np
from models.smooth import custom_queue


class MeanFilter():
    def __init__(self, win_size, decay=0.9):
        """
        :param win_size:记录历史信息的窗口大小
        :param decay: 衰减系数，值越大，历史影响衰减的越快，平滑力度越小
        """
        self.queue = custom_queue.Queue(win_size)
        if decay:
            self.weight_decay = self.get_weight(win_size, decay=decay)
        else:
            self.weight_decay = None
        print("prob:{}".format(self.weight_decay))

    def update(self, point):
        if point[0] > 0 and point[1] > 0:
            self.queue.put(point)


    def predict(self):
        if len(self.queue) > 0:
            point = self.filter()
        else:
            point = np.array([[0], [0]])
        return point

    def filter(self, ):
        data = np.asarray(self.queue.data)
        if self.weight_decay is not None:
            p = np.reshape(self.weight_decay[-len(data):], (1, len(data)))
            m = np.matmul(p,data).reshape(-1)
        else:
            m = np.mean(data, axis=0)
        return m

    @staticmethod
    def get_weight(n, decay=0.5):
        """
        当n=5,decay=0.5时，对应的衰减权重为，越远的权重越小
        w=[0.0625 0.0625 0.125  0.25   0.5   ]
        :param n:
        :param decay: 衰减系数，值越大，历史影响衰减的越快，平滑力度越小
        :return:
        """
        r = decay / (1 - decay)
        # 计算衰减权重
        w = [1]
        for i in range(1, n):
            w.append(sum(w) * r)
        # 进行归一化
        w = w / np.sum(w)
        return w
