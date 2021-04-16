# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: finger-pen-tip-keypoint-detection
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-12-22 11:43:13
# --------------------------------------------------------
"""

from demo import PoseEstimation

# -*-coding: utf-8 -*-
"""
    @Project: finger-keypoint-detection
    @File   : demo.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2020-11-08 15:02:19
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))

import cv2
import numpy as np
import random
import argparse
import tensorflow as tf
from utils import image_processing, debug, file_processing
from demo import PoseEstimation
from configs import val_config, teacher_config

project_root = os.path.dirname(__file__)
print("TF:{}".format(tf.__version__))


class PoseAction(PoseEstimation):
    def __init__(self, config, threshhold=0.0, device="cuda:0"):
        """
        :param model_path: 模型文件，支持ONNX，TFlite,PB和h5等格式的模型
        :param input_size: 模型输入input size
        :param threshhold: 关键点置信度
        """
        super(PoseAction, self).__init__(config, threshhold=threshhold)

    def start_capture(self, video_path, save_root, detect_freq=1):
        """
        start capture video
        :param video_path: *.avi,*.mp4,...
        :param save_video: *.avi
        :param detect_freq:
        :return:
        """
        file_processing.create_dir(save_root)
        result_json = []
        # cv2.moveWindow("test", 1000, 100)
        postfix = os.path.basename(video_path).split(".")[-1]
        basename = os.path.basename(video_path)[:-len(postfix) - 1]
        time = file_processing.get_time()
        video_cap = image_processing.get_video_capture(video_path)
        width, height, numFrames, fps = image_processing.get_video_info(video_cap)
        # freq = int(fps / detect_freq)
        count = 0
        while True:
            isSuccess, frame = video_cap.read()
            if not isSuccess:
                break
            # print("frame:{}".format(count))
            if count % detect_freq == 0:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = image_processing.resize_image(frame, resize_height=500)
                result = self.task(frame)
                result["id"] = count
                result_json.append(result)
            count += 1

        video_cap.release()
        file_processing.write_json_path(os.path.join(save_root, "{}_{}.json".format(basename, time)), result_json)
        print("have:{}".format(len(result_json)))

    def start_capture_for_neg(self, video_path, save_root, detect_freq=1):
        """
        start capture video
        :param video_path: *.avi,*.mp4,...
        :param save_video: *.avi
        :param detect_freq:
        :return:
        """
        file_processing.create_dir(save_root)
        time = file_processing.get_time()
        # cv2.moveWindow("test", 1000, 100)
        postfix = os.path.basename(video_path).split(".")[-1]
        basename = os.path.basename(video_path)[:-len(postfix) - 1]
        video_cap = image_processing.get_video_capture(video_path)
        width, height, numFrames, fps = image_processing.get_video_info(video_cap)
        # freq = int(fps / detect_freq)
        count = 0
        result_json = []
        winsize = int(random.uniform(fps * 1.0, fps * 2.5))
        flag = 0
        while True:
            isSuccess, frame = video_cap.read()
            if not isSuccess:
                break
            # print("frame:{}".format(count))
            if count % detect_freq == 0:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = image_processing.resize_image(frame, resize_height=500)
                result = self.task(frame)
                result["id"] = count
                result_json.append(result)
            count += 1
            if count % winsize == 0:
                flag += 1
                file_processing.write_json_path(
                    os.path.join(save_root, "{}_{:0=3d}_{}.json".format(basename, flag, time)),
                    result_json)
                print("have:{}".format(len(result_json)))
                count = 0
                result_json = []
                winsize = int(random.uniform(fps * 1.0, fps * 2.5))

        video_cap.release()

    def task(self, bgr_image):
        height, width, _ = bgr_image.shape
        key_points, kp_scores, body_rects = self.detect_image(bgr_image, threshhold=self.threshhold)
        points, scores = key_points[0], kp_scores[0]
        points = points / (width, height)
        result = {"points": points.tolist(), "scores": scores.tolist()}
        self.show_result(bgr_image, key_points, body_rects, kp_scores, waitKey=15)
        return result

    def detect_video_dir(self, video_root, save_root, waitKey=0):
        """
        :param image_dir:
        :param waitKey:
        :return:
        """
        if os.path.isdir(video_root):
            video_list = file_processing.get_files_list(video_root, postfix=["*.mp4", "*.MP4", "*.avi", "*.AVI",
                                                                             "*.MOV", "*.mov", "*.rmvb", "*.RMVB",
                                                                             "*.3gp"])
        else:
            video_list = [video_root]

        for i, video_path in enumerate(video_list):
            print(video_path)
            sub = os.path.basename(os.path.dirname(video_path))
            save_dir = os.path.join(save_root, sub)
            if "none" == sub:
                # self.start_capture_for_neg(video_path, save_dir)
                pass
            else:
                self.start_capture(video_path, save_dir)


def get_parser():
    # model_path = "data/pretrained/tf_pose_resnet_50"
    # model_path = "data/pretrained/model_mobilenet_v2.h5"
    # model_path = "data/pretrained/model_mobilenet_v2_optimize_float16.tflite"
    # model_path = "data/pretrained/model_mobilenet_v2"
    # model_path = "data/pretrained/model_mobilenet_v2.pb"
    model_path = "data/pretrained/onnx/model_mobilenet_v2_0.5_256_256_sim.onnx"
    # model_path = "data/pretrained/onnx/model_mobilenet_v2_0.5_4_256x256_8.onnx"
    image_dir = "data/test_image/test4.jpg"
    parser = argparse.ArgumentParser(description='detect_imgs')
    parser.add_argument('--model_path', default=model_path, type=str, help='model_path')
    parser.add_argument('--image_dir', default=image_dir, type=str, help='directory or image path')
    parser.add_argument('--threshhold', default=0.0, type=float, help='score threshold')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # hp = PoseAction(config=val_config.custom_coco_finger4_model_pose_resnetst_256_256, device="cuda:0")
    # hp = PoseAction(config=val_config.coco_res18_192_256, device="cuda:0")
    # hp = PoseAction(config=val_config.mpii_256_256, device="cuda:0")
    # hp = PoseAction(config=val_config.custom_coco_finger_res18_192_256, device="cuda:0")
    # hp = PoseAction(config=val_config.custom_coco_finger4_model_mbv2_192_256, device="cuda:0")
    # hp = PoseAction(config=val_config.custom_coco_finger4_model_mbv2_256_256, device="cuda:0")
    hp = PoseAction(config=val_config.custom_coco_finger4_model_mbv2_256_256, device="cuda:0")
    # hp = PoseAction(config=val_config.custom_coco_finger4_model_mbv2_192_192, device="cuda:0")
    # hp = PoseAction(config=val_config.custom_coco_finger4_model_pose_resnetst_256_256, device="cuda:0")
    # hp = PoseAction(config=teacher_config.custom_coco_finger4_model_pose_resnetst_256_256, device="cuda:0")
    # hp = PoseAction(config=teacher_config.custom_coco_finger4_model_pose_hrnet_256_256, device="cuda:0")
    # hp = PoseAction(config=val_config.custom_coco_finger_model_mbv2_192_256, device="cuda:0")
    # hp = PoseAction(config=val_config.custom_coco_person_res18_192_256, device="cuda:0")
    # hp = PoseAction(config=val_config.custom_mpii_256_256, device="cuda:0")
    # hp = PoseAction(config=val_config.student_mpii_256_256, device="cuda:0")
    # hp = PoseAction(config=val_config.student_mpii_256_256_v2, device="cuda:0")
    # video_path = "./data/video/video.avi"
    video_path = "/home/dm/panjinquan3/dataset/finger_keypoint/pen_v1/action/video_crop"
    # video_path = "/home/dm/panjinquan3/dataset/finger_keypoint/pen_v1/video_crop"
    save_root = "/home/dm/panjinquan3/dataset/finger_keypoint/pen_v1/action/video_json2"
    # video_path = "data/video/VID_20200814_110845.mp4"
    # save_video = "data/video/VID_20200814_110845_result.mp4"
    # hp.start_capture(video_path, save_root)
    hp.detect_video_dir(video_path, save_root)
