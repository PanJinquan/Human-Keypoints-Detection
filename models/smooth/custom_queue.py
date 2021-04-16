# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-01-19 19:34:03
"""


class Queue(object):
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

    def get_len(self):
        return self.__len__()

    def get_item(self, index=-1):
        """
        获得队列中某一帧的信息
        :param index:
        :return:
        """
        return self.data[index]

    def get_seq(self, index=0):
        """
        获得队列某一ID的时序信息,比如某个关键点的时序轨迹信息
        :param index:
        :return:
        """
        seq = []
        for i in range(len(self)):
            item = self.get_item(i)
            seq.append(item[index])
        return seq
