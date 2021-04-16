# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: face-person-ssd-pytorch
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-09-14 10:05:12
# --------------------------------------------------------
"""

import sys
import os

sys.path.append(os.getcwd())
import cv2
import argparse
import demo
import numpy as np
from models.prior_boxes import PriorBoxes
from models.onnx_model import ONNXModel
from utils import image_processing, file_processing

class_names = ["BACKGROUND", "face", "person"]


class Detector(demo.Detector):
    """
    Ultra Light Face/Person Detector
    """

    """
    Ultra Light Face/Person Detector
    """

    def __init__(self,
                 model_path,
                 net_type,
                 input_size,
                 class_names,
                 priors_type,
                 candidate_size=200,
                 prob_threshold=0.35,
                 iou_threshold=0.3,
                 device="cuda:0"):
        """
        :param model_path:  path to model(*.pth) file
        :param net_type:  "RFB" (higher precision) or "slim" (faster)'
        :param input_size: model input size
        :param class_names: class_names
        :param priors_type: face or person
        :param candidate_size:nms candidate size
        :param prob_threshold: 置信度分数
        :param iou_threshold:  NMS IOU阈值
        :param device: GPU Device
        """
        self.net_type = net_type
        self.input_size = input_size
        self.priors_type = priors_type
        self.class_names = class_names
        self.candidate_size = candidate_size
        self.iou_threshold = iou_threshold
        self.prob_threshold = prob_threshold
        self.device = device
        self.prior_boxes = PriorBoxes(self.input_size, priors_type=self.priors_type)
        self.image_size = self.prior_boxes.input_size
        self.image_mean = self.prior_boxes.image_mean
        self.image_std = self.prior_boxes.image_std
        self.net = self.build_net(model_path, self.net_type)

    def start_capture(self, video_path, save_video=None, detect_freq=1):
        """
        start capture video
        :param video_path: *.avi,*.mp4,...
        :param save_video: *.avi
        :param detect_freq:
        :return:
        """
        # cv2.moveWindow("test", 1000, 100)
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
                self.task(frame)
            if save_video:
                self.video_writer.write(frame)
            count += 1
        video_cap.release()

    def task(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = self.detect_image(rgb_image, isshow=False)
        self.show_image(rgb_image, boxes, labels, probs, waitKey=10)
        return frame

    def show_image(self, rgb_image, boxes, labels, probs, waitKey=0):
        boxes_name = ["{}:{:3.2f}".format(l, s) for l, s in zip(labels, probs)]
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if len(boxes) > 0:
            bgr_image = image_processing.draw_image_detection_bboxes(rgb_image, boxes, probs, labels)
            waitKey = 0
        cv2.imshow("Det", bgr_image)
        # cv2.imwrite("result.jpg", bgr_image)
        cv2.waitKey(waitKey)


if __name__ == "__main__":
    args = demo.get_parser()
    print(args)
    net_type = args.net_type
    input_size = args.input_size
    priors_type = args.priors_type
    device = args.device
    # model_path = "RFB-person.pth"
    image_dir = args.image_dir
    model_path = args.model_path
    model_path = "/home/dm/panjinquan3/FaceDetector/Ultra-Light-Fast-Generic-Face-Detector-1MB/work_space/test_model/mbv2_face_person/mbv21.0_face_person_640_360_BACKGROUND_MPII_20200920123754/model/best_model_mbv2_027_loss2.2036.pth"
    candidate_size = args.candidate_size
    prob_threshold = args.prob_threshold
    iou_threshold = args.iou_threshold
    det = Detector(model_path,
                   net_type=net_type,
                   input_size=input_size,
                   class_names=class_names,
                   priors_type=priors_type,
                   candidate_size=candidate_size,
                   iou_threshold=iou_threshold,
                   prob_threshold=prob_threshold,
                   device=device)

    video_path = "/home/dm/panjinquan3/dataset/person/video/1/人体误检测试专用.mp4"
    det.start_capture(video_path)
