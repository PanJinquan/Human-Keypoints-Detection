# -*-coding: utf-8 -*-
"""
    @Project: torch-Human-Pose-Estimation-Pipeline
    @File   : demo.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-11-08 15:02:19
"""
import sys
import os

sys.path.append("libs/detector/libs/detector")

import cv2
import numpy as np
import argparse

sys.path.append(os.path.dirname(__file__))
from configs import val_config
from libs.detector.libs.detector.detector import Detector
from utils import image_processing, debug, file_processing, torch_tools
from models import inference

project_root = os.path.dirname(__file__)


class PoseEstimation(inference.PoseEstimation):
    """
     mpii_keypoints_v2 = {0: "r_ankle", 1: "r_knee", 2: "r_hip", 3: "l_hip", 4: "l_knee", 5: "l_ankle", 6: "pelvis",
                         7: "thorax", 8: "upper_neck", 9: "head_top", 10: " r_wrist", 11: "r_elbow", 12: "r_shoulder",
                         13: "l_shoulder", 14: "l_elbow", 15: "l_wrist"}

     mpii_keypoints = {"r_ankle": 0, "r_knee": 1, "r_hip": 2, "l_hip": 3, "l_knee": 4, "l_ankle": 5, "pelvis": 6,
                      "thorax": 7, "upper_neck": 8, "head_top": 9, " r_wrist": 10, "r_elbow": 11, "r_shoulder": 12,
                      "l_shoulder": 13, "l_elbow": 14, "l_wrist": 15}
    """

    def __init__(self, config, model_path=None, threshhold=0.3, device="cuda:0"):
        """
        :param config:
        :param threshhold:
        :param device:
        """
        super(PoseEstimation, self).__init__(config, model_path, threshhold, device)
        self.threshhold = threshhold
        self.detector = Detector(detect_type="ultra_person", device=device)

    def start_capture(self, video_path, save_video=None, detect_freq=1):
        """
        start capture video
        :param video_path: *.avi,*.mp4,...
        :param save_video: *.avi
        :param detect_freq:
        :return:
        """
        video_cap = image_processing.get_video_capture(video_path)
        width, height, numFrames, fps = image_processing.get_video_info(video_cap)
        if save_video:
            self.video_writer = image_processing.get_video_writer(save_video, width, height, fps)
        # freq = int(fps / detect_freq)
        count = 0
        while True:
            isSuccess, frame = video_cap.read()
            if not isSuccess:
                break
            if count % detect_freq == 0:
                kp_points, kp_scores, boxes = self.detect_image(frame,
                                                                threshhold=self.threshhold,
                                                                detect_person=True)
                self.show_result(frame, boxes, kp_points, kp_scores, self.skeleton, waitKey=5)
            if save_video:
                self.video_writer.write(frame)
            count += 1
        video_cap.release()

    def detect_person(self, image):
        bbox_score, labels = self.detector.detect(image, isshow=False)
        boxes, scores = [], []
        if len(bbox_score) > 0:
            boxes = bbox_score[:, 0:4]
            scores = bbox_score[:, 4:5]
        return boxes, scores

    def detect_pose(self, image, boxes, threshhold):
        kp_points, kp_scores = self.detect(image, boxes, threshhold=threshhold)
        return kp_points, kp_scores

    def detect_image(self, frame, threshhold=0.8, detect_person=False):
        '''
        :param frame: bgr image
        :param threshhold:
        :return:
        '''
        if detect_person:
            boxes, scores = self.detect_person(frame)
        else:
            h, w, d = frame.shape
            boxes = [[0, 0, w, h]]
        key_points, kp_scores = self.detect_pose(frame, boxes, threshhold)
        return key_points, kp_scores, boxes

    def detect_image_dir(self, image_dir, detect_person=True, waitKey=0):
        image_list = file_processing.get_files_lists(image_dir)
        for i, image_path in enumerate(image_list):
            bgr_image = cv2.imread(image_path)
            # bgr_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)
            # bgr_image = image_processing.resize_image(bgr_image, resize_height=800)
            kp_points, kp_scores, boxes = self.detect_image(bgr_image,
                                                            threshhold=self.threshhold,
                                                            detect_person=detect_person)
            self.show_result(bgr_image, boxes, kp_points, kp_scores, self.skeleton, waitKey)

    def show_result(self, image, boxes, kp_points, kp_scores, skeleton=None, waitKey=0):
        if not skeleton:
            skeleton = self.skeleton
        image = self.draw_keypoints(image, boxes, kp_points, kp_scores, skeleton)
        cv2.imwrite('test.png', image)
        cv2.imshow('test', image)
        cv2.waitKey(waitKey)

    def draw_keypoints(self,
                       image,
                       boxes,
                       kp_points,
                       kp_scores,
                       skeleton, box_color=(255, 0, 0),
                       circle_color=(0, 255, 0), line_color=(0, 0, 255)):
        """
        :param image:
        :param keypoints:
        :param kp_scores:
        :param bboxes:
        :param scores
        :return:
        """
        vis_image = image.copy()
        vis_image = image_processing.draw_key_point_in_image(vis_image, kp_points,
                                                             circle_color=circle_color,
                                                             line_color=line_color,
                                                             pointline=skeleton,
                                                             thickness=10)
        vis_image = image_processing.draw_image_boxes(vis_image, boxes, color=box_color)
        return vis_image


def get_parser():
    parser = argparse.ArgumentParser(description="Training Pipeline")
    # parser.add_argument("-c", "--config_file", help="configs file", default="configs/config.yaml", type=str)
    parser.add_argument("-d", "--device", help="device: device or cuda", default="cpu", type=str)
    parser.add_argument("-i", "--image_file", help="image file or directory", default="", type=str)
    parser.add_argument("-v", "--video_file", help="video file", default=0, type=str or int)
    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    # COCO共17个关键点
    hp = PoseEstimation(config=val_config.person_coco_192_256, device=parser.device)
    if parser.image_file:
        # 测试图片
        hp.detect_image_dir(image_dir=parser.image_file, detect_person=True, waitKey=0)
    elif isinstance(parser.video_file, str):
        # 测试视频文件
        hp.start_capture(video_path=parser.video_file, save_video=None)
    else:
        # 测试摄像头
        hp.start_capture(video_path=int(parser.video_file))
