# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: python-learning-notes
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-08-22 16:19:37
# --------------------------------------------------------
"""
import sys
import os

sys.path.insert(0, os.getcwd())

import cv2
import numpy as np
from utils import image_processing, file_processing, debug

project_root = os.path.join(os.path.dirname(__file__))


class Detector():

    def __init__(self, detect_type="mtcnn",device="cuda:0"):
        self.device = device
        self.class_names = None
        self.detect_type = detect_type
        if self.detect_type == "mtcnn":
            sys.path.append(os.path.join(os.path.dirname(__file__), "mtcnn"))
            from mtcnn import mtcnn
            self.class_names = ["BACKGROUND", "face"]
            self.detector = mtcnn.MTCNN()
        elif self.detect_type == "ultra_face":
            sys.path.append(os.path.join(os.path.dirname(__file__), "ultra_light_detector"))
            from ultra_light_detector import demo as ultra_detector
            self.class_names = ["BACKGROUND", "face"]
            args = ultra_detector.get_parser()
            print(args)
            net_type = "rfb"
            input_size = [640, 640]
            priors_type = "face"
            model_path = "ultra_light_detector/data/pretrained/pth/rfb_face_640_640.pth"
            model_path = os.path.join(project_root, model_path)
            candidate_size = args.candidate_size
            prob_threshold = args.prob_threshold
            iou_threshold = args.iou_threshold
            self.detector = ultra_detector.Detector(model_path,
                                                    net_type=net_type,
                                                    input_size=input_size,
                                                    class_names=self.class_names,
                                                    priors_type=priors_type,
                                                    candidate_size=candidate_size,
                                                    iou_threshold=iou_threshold,
                                                    prob_threshold=prob_threshold,
                                                    device=self.device)
        elif self.detect_type == "ultra_person":
            sys.path.append(os.path.join(os.path.dirname(__file__), "ultra_light_detector"))
            from ultra_light_detector import demo as ultra_detector
            self.class_names = ["BACKGROUND", "person"]
            args = ultra_detector.get_parser()
            print(args)
            net_type = "rfb"
            input_size = [640, 360]
            priors_type = "person"
            model_path = "ultra_light_detector/data/pretrained/pth/rfb_person_640_360.pth"
            model_path = os.path.join(project_root, model_path)
            candidate_size = args.candidate_size
            prob_threshold = args.prob_threshold
            iou_threshold = args.iou_threshold
            self.detector = ultra_detector.Detector(model_path,
                                                    net_type=net_type,
                                                    input_size=input_size,
                                                    class_names=self.class_names,
                                                    priors_type=priors_type,
                                                    candidate_size=candidate_size,
                                                    iou_threshold=iou_threshold,
                                                    prob_threshold=prob_threshold,
                                                    device=self.device)

        elif self.detect_type == "ultra_face_person":
            sys.path.append(os.path.join(os.path.dirname(__file__), "ultra_light_detector"))
            from ultra_light_detector import demo as ultra_detector
            self.class_names = ["BACKGROUND", "face", "person"]
            args = ultra_detector.get_parser()
            print(args)
            net_type = "mbv2"
            input_size = [640, 360]
            priors_type = "face_person"
            model_path = "ultra_light_detector/data/pretrained/pth/mbv2_face_person_640_360.pth"
            model_path = os.path.join(project_root, model_path)
            candidate_size = args.candidate_size
            prob_threshold = args.prob_threshold
            iou_threshold = args.iou_threshold
            self.detector = ultra_detector.Detector(model_path,
                                                    net_type=net_type,
                                                    input_size=input_size,
                                                    class_names=self.class_names,
                                                    priors_type=priors_type,
                                                    candidate_size=candidate_size,
                                                    iou_threshold=iou_threshold,
                                                    prob_threshold=prob_threshold,
                                                    device=self.device)
        elif self.detect_type == "ultra_person_pig":
            sys.path.append(os.path.join(os.path.dirname(__file__), "ultra_light_detector"))
            from ultra_light_detector import demo as ultra_detector
            self.class_names = ["BACKGROUND", "person", "pig"]
            args = ultra_detector.get_parser()
            print(args)
            net_type = "mbv2"
            input_size = [416, 416]
            priors_type = "person_pig"
            model_path = "ultra_light_detector/data/pretrained/pth/mbv2_person_pig_416_416.pth"
            model_path = os.path.join(project_root, model_path)
            candidate_size = args.candidate_size
            prob_threshold = args.prob_threshold
            iou_threshold = args.iou_threshold
            self.detector = ultra_detector.Detector(model_path,
                                                    net_type=net_type,
                                                    input_size=input_size,
                                                    class_names=self.class_names,
                                                    priors_type=priors_type,
                                                    candidate_size=candidate_size,
                                                    iou_threshold=iou_threshold,
                                                    prob_threshold=prob_threshold,
                                                    device=self.device)

        elif "darknet" in self.detect_type:
            sys.path.append(os.path.join(os.path.dirname(__file__), "darknet"))
            from darknet import yolo_det
            self.class_names = yolo_det.COCO_NAME
            self.detector = yolo_det.YOLODetection()
        elif self.detect_type == "dfsd":
            self.class_names = ["BACKGROUND", "face"]
            sys.path.append(os.path.join(os.path.dirname(__file__), "dfsd_landmark"))
            from dfsd_landmark.demo import FaceLandmarkDetection
            prob_threshold = 0.95
            self.detector = FaceLandmarkDetection(prob_threshold=prob_threshold)
        else:
            raise Exception("Error:{}".format(self.detect_type))
        print("detect_type:{},class_names:{}".format(self.detect_type, self.class_names))

    def start_capture(self, video_path, save_video=None, detect_freq=1, isshow=True):
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
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.task(frame, isshow)
            if save_video:
                self.video_writer.write(frame)
            count += 1
        video_cap.release()

    def task(self, frame, isshow=True):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox_score = self.detect(rgb_image, isshow=False)
        if isshow:
            image, boxes, probs = self.show_image(rgb_image, bbox_score)

    def detect(self, rgb_image, isshow=False):
        """
        :param rgb_image:
        :param isshow:
        :return:
        """
        bbox_score = np.asarray([])
        if self.detect_type == "mtcnn":
            bbox_score, landmarks = self.detector.detect_image(rgb_image)
            labels = np.ones(shape=(len(bbox_score)), dtype=np.int32)
        elif "ultra" in self.detect_type:
            boxes, labels, probs = self.detector.detect_image(rgb_image, isshow=False)
            if len(boxes) > 0:
                bbox_score = np.hstack((boxes, probs.reshape(-1, 1)))
        elif self.detect_type == "darknet_person":
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            dets = self.detector.detect_image(bgr_image)
            # get person bbox
            boxes, probs, labels = self.detector.get_boxes_scores_labels(dets, filter_label=0)
            if len(boxes) > 0:
                bbox_score = np.hstack((boxes, probs.reshape(-1, 1)))
        elif self.detect_type == "darknet":
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            dets = self.detector.detect_image(bgr_image)
            # get coco detection
            boxes, probs, labels = self.detector.get_boxes_scores_labels(dets, filter_label=None)
            if len(boxes) > 0:
                bbox_score = np.hstack((boxes, probs.reshape(-1, 1)))
        elif self.detect_type == "dfsd":
            boxes, probs, landmarks = self.detector.detect(rgb_image, isshow=False)
            labels = np.ones(shape=(len(boxes)), dtype=np.int32)
            if len(boxes) > 0:
                bbox_score = np.hstack((boxes, probs.reshape(-1, 1)))
        else:
            raise Exception("Error:{}".format(self.detect_type))
        if isshow:
            self.show_image(rgb_image, bbox_score)
        return bbox_score, labels

    def detect_image_dir(self, image_dir, isshow=True):
        """
        :param image_dir: directory or image file path
        :param isshow:<bool>
        :return:
        """
        if os.path.isdir(image_dir):
            image_list = file_processing.get_files_lists(image_dir, postfix=["*.jpg", "*.png"])
        elif os.path.isfile(image_dir):
            image_list = [image_dir]
        else:
            raise Exception("Error:{}".format(image_dir))
        for img_path in image_list:
            orig_image = cv2.imread(img_path)
            rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            bbox_score, labels = self.detect(rgb_image, isshow=False)

            if isshow:
                image, boxes, probs = self.show_image(rgb_image, bbox_score, labels)
                self.save_result(img_path, image, boxes, probs)

    def show_image(self, image, bbox_score, labels, waitKey=0):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.class_names:
            labels = [self.class_names[l] for l in labels]
        if len(bbox_score) > 0:
            boxes = bbox_score[:, :4]
            probs = bbox_score[:, 4]
            image = image_processing.draw_image_detection_bboxes(image, boxes, probs, labels)
        else:
            boxes, probs = np.asarray([]), np.asarray([])
        cv2.imshow("Det", image)
        cv2.waitKey(waitKey)
        return image, boxes, probs

    def save_result(self, img_path, image, boxes, probs):
        out_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), "det-result")
        basename = os.path.basename(img_path).split(".")[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if isinstance(boxes, np.ndarray):
            boxes = boxes.tolist()
        if isinstance(probs, np.ndarray):
            probs = probs.tolist()
        out_json_path = os.path.join(out_dir, basename + ".json")
        json_data = {"boxes": boxes}
        file_processing.write_json_path(out_json_path, json_data)


if __name__ == "__main__":
    image_dir = "/home/dm/project/python-learning-notes/libs/detector/test_image"
    # det = Detector(detect_type="ultra_face")
    det = Detector(detect_type="ultra_person_pig")
    # det = Detector(detect_type="dfsd")
    # det = Detector(detect_type="mtcnn")
    # det = Detector(detect_type="darknet")
    det.detect_image_dir(image_dir, isshow=True)
    # det.start_capture(video_path, isshow=True)
