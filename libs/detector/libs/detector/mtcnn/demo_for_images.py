# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-07-06 11:19:37
# --------------------------------------------------------
"""
import sys
import os

sys.path.append(os.getcwd())

import copy
import cv2
import mtcnn
from utils import image_processing, file_processing


class MTCNNDemo(mtcnn.MTCNN):
    def __init__(self):
        min_face_size = 20.0
        thresholds = [0.6, 0.7, 0.95]
        nms_thresholds = [0.7, 0.7, 0.7]
        super(MTCNNDemo, self).__init__(min_face_size, thresholds, nms_thresholds)

    def detect_image(self, rgb_image, resize_height=None, resize_width=None):
        height, width, _ = rgb_image.shape
        input_image = image_processing.resize_image(rgb_image, resize_height, resize_width)
        bbox_score, landmarks = self.detect_image(input_image)
        resize_height, resize_width, _ = input_image.shape
        bbox_score_scale = [width / resize_width, height / resize_height] * 2 + [1.0]
        landmarks_scale = [width / resize_width, height / resize_height] * 5
        bbox_score = bbox_score * bbox_score_scale
        landmarks = landmarks * landmarks_scale
        return bbox_score, landmarks

    def detect_image_dir(self, image_dir, resize_height=None, resize_width=None):
        if os.path.isfile(image_dir):
            image_list = [image_dir]
            out_dir = os.path.join(os.path.dirname(image_dir), "result2")

        else:
            image_list = file_processing.get_files(image_dir, postfix=["*.jpg"])
            out_dir = os.path.join(image_dir, "result2")
        for image_path in image_list:
            rgb_image = image_processing.read_image(image_path, colorSpace="RGB")
            bbox_score, landmarks = self.detect_image(rgb_image, resize_height, resize_width)
            bboxes, scores, landmarks = mt.adapter_bbox_score_landmarks(bbox_score, landmarks)
            # bboxes = image_processing.get_square_bboxes(bboxes, fixed="L")
            rgb_image = self.show_landmark_boxes("image", rgb_image, landmarks, bboxes)
            # rgb_image = image_processing.show_landmark_boxes("image", rgb_image, landmarks, bboxes
            image_name = os.path.basename(image_path)
            self.save_data(rgb_image, image_name, out_dir, bboxes, scores, landmarks)

    def save_data(self, rgb_image, image_name, out_dir, bboxes, scores, landmarks):
        save_image_path = os.path.join(out_dir, image_name)
        save_json_path = os.path.join(out_dir, image_name[:-len(".jpg")] + ".json")
        file_processing.create_file_path(save_image_path)
        image_processing.save_image(save_image_path, rgb_image)
        data = []
        for box, score, landm in zip(bboxes, scores, landmarks):
            d = {"bboxes": box.tolist(),
                 "scores": score.tolist(),
                 "landmarks": landm.tolist()}
            data.append(d)
        file_processing.write_json_path(save_json_path, data)

    def show_landmark_boxes(self, win_name, image, landmarks_list, boxes):
        image = self.draw_landmark(image, landmarks_list)
        image = self.draw_image_boxes(image, boxes)
        image_processing.cv_show_image(win_name, image, waitKey=10)
        return image

    def draw_landmark(self, image, landmarks_list, point_color=(0, 0, 255)):
        image = copy.copy(image)
        point_size = 12
        thickness = -1  # 可以为 0 、4、8
        for landmarks in landmarks_list:
            for i, landmark in enumerate(landmarks):
                # 要画的点的坐标
                point = (int(landmark[0]), int(landmark[1]))
                cv2.circle(image, point, point_size, point_color, thickness)
        return image

    def draw_image_boxes(self, bgr_image, boxes_list, color=(0, 0, 255)):
        thickness = 6
        for box in boxes_list:
            x1, y1, x2, y2 = box
            point1 = (int(x1), int(y1))
            point2 = (int(x2), int(y2))
            cv2.rectangle(bgr_image, point1, point2, color, thickness=thickness)
        return bgr_image


if __name__ == "__main__":
    # image_dir = "/media/dm/dm1/git/python-learning-notes/dataset/dmai_demo/face_person_landmark/landmark"
    image_dir = "/media/dm/dm1/git/python-learning-notes/dataset/dmai_demo/face_person_landmark/landmark/009.jpg"
    mt = MTCNNDemo()
    mt.detect_image_dir(image_dir, resize_height=480)
