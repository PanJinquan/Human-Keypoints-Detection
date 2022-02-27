# -*-coding: utf-8 -*-
"""
    @Project: StatusMonitor
    @File   : demo.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-02-14 15:24:57
"""
import os
import cv2
import numpy as np
import monitor
from utils import image_processing

config_l1 = {"detect_freq": 7,
             "threshold": 0.05,
             "up_threshold": 0.2,
             "win_size": 4,
             "up_flag": "1000",
             "measurement": "l1",
             }

config_l2 = {"detect_freq": 7,
             "threshold": 0.0025,
             "up_threshold": 0.2,
             "win_size": 4,
             "up_flag": "1000",
             "measurement": "l2",
             }


class FeatureDemo(object):
    def __init__(self, out_dir, config):
        """
        :param out_dir: 保存上传图片的路径
        :param config: 配置参数
             --detect_freq  : 检测频率,1S内检测的帧数,默认1秒检测7帧
             --threshold    : 相似度(距离)阈值,用于判断当前状态是: "运动"还是"静止"
             --up_threshold : 上传图片的相似阈值,当frame与上一次的upload_frame的相似性距离大于该阈值时,则上传图片
                              否则,认为当前帧与上一次上传的图片非常相似,不需要上传
             --up_flag      : 上传图片的信号符: 默认"1000": 表示先出现1次label=1,然后连续3帧的label都是0,则认为需要上传图片
             --win_size     : 状态检测器的窗口大小,用于记录历史label信息,推荐等于上传图片的信号符up_flag长度
             --measurement  : 相似度(距离)度量方法,l1: 平均绝对误差作为相似度(距离); l2: 均方差作为相似度(距离),l1比l2稍微快一点
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.out_dir = out_dir

        self.measurement = config["measurement"]
        self.threshold = config["threshold"]
        self.up_threshold = config["up_threshold"]
        self.up_flag = config["up_flag"]
        self.detect_freq = config["detect_freq"]
        self.win_size = config["win_size"]

        self.last_frame = None  # 上一帧的图片
        self.next_frame = None  # 下一帧(当前帧)图片
        self.upload_frame = None  # 记录上一次上传的图片

        # 初始化状态检测器
        self.monitor = monitor.StatusMonitor(win_size=self.win_size)

    def start_capture(self, video_path, save_video=None):
        """
        :param video_path:
        :param save_video:
        :return:
        """
        video_cap = image_processing.get_video_capture(video_path)
        width, height, numFrames, fps = image_processing.get_video_info(video_cap)
        if save_video:
            self.video_writer = image_processing.get_video_writer(save_video, width, height, fps)
        freq = int(fps / self.detect_freq)
        self.frame_id = 0
        while True:
            isSuccess, frame = video_cap.read()
            self.frame_id += 1
            if not isSuccess:
                break
            if self.frame_id % freq == 0 or self.frame_id == 1:
                label = self.task(frame)
            if save_video:
                self.video_writer.write(frame)
        video_cap.release()

    def processInput(self, orig_frame, resize):
        image = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        # 为了避免resize耗时,建议采用最近邻插值方式
        image = cv2.resize(image, tuple(resize), interpolation=cv2.INTER_NEAREST)
        # 高斯滤波,去除噪声干扰
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = np.asarray(image, dtype=np.float32) / 255.0
        return image

    def task(self, orig_frame, resize=[96, 96]):
        """
        42,dist=0.02571568,label=0,label_list=['1', '0', '0', '0']
        :param orig_frame: 原始帧图片
        :param resize: image resize
        :return:
        """
        # TODO
        self.next_frame = self.processInput(orig_frame, resize)
        if self.last_frame is None:
            # 初始帧
            self.last_frame = self.next_frame
            self.upload_frame = self.next_frame
            dist = 0
        else:
            # 计算上一帧图片与下一帧(当前帧)的相似度(距离)
            dist = self.monitor.get_frame_similarity(self.last_frame,
                                                     self.next_frame,
                                                     measurement=self.measurement,
                                                     isshow=True)
        # TODO
        # 判断状态: 0: 静止, 1: 运动
        label = "0" if dist < self.threshold else "1"
        # 与上一次上传的图片比较相似性,判断是否需要上传图片: True(上传),False(不上传),
        isupload = self.monitor.check_upload_status(self.next_frame,
                                                    self.upload_frame,
                                                    label,
                                                    up_threshold=self.up_threshold,
                                                    up_flag=self.up_flag)
        if isupload:
            # 如确定需要上传图片,则记录已上传的图片
            self.upload_frame = self.next_frame
            cv2.imwrite(os.path.join(self.out_dir, "{:0=4d}.jpg".format(self.frame_id)), orig_frame)

        self.last_frame = self.next_frame
        print("{},dist={:3.8f},label={},label_list={}".format(self.frame_id, dist, label,
                                                              str(self.monitor.get_label_list())))
        self.show_result(orig_frame, label, isupload, dist)
        return label

    def show_result(self, image, label, isupload, dist):
        """
        显示状态结果,需要上传图片时,会暂停,以便观察
        :param image:
        :param label:
        :param isupload:
        :param dist:
        :return:
        """
        if label == 0 or label == "0":
            label = "static"  # 静止
        else:
            label = "moving"  # 运动
        label = "dist:{:3.6f}  ".format(dist) + label
        flag = "count:{}".format(self.frame_id)
        if isupload:
            flag += "  upload"
        image = image_processing.resize_image(image, resize_width=800)
        image = image_processing.draw_text(image, point=(1, 15), text=flag, drawType="simple")
        image = image_processing.draw_text(image, point=(1, 30), text=label, drawType="simple")
        cv2.imshow("image", image)
        if isupload:
            cv2.waitKey(0)
        cv2.waitKey(30)
        return image


if __name__ == "__main__":
    # video_path = "data/VID_20200814_110845.mp4"
    video_path = "data/VID_20200814_100237.mp4"
    out_dir = video_path[:-len(".mp4")]
    fd = FeatureDemo(out_dir, config=config_l1)
    # fd = FeatureDemo(out_dir, config=config_l2)
    fd.start_capture(video_path=video_path)
