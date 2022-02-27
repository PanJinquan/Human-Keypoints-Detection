# -*- coding: utf-8 -*-
"""
This code is used to batch detect images in a folder.
"""
import sys
import os

sys.path.append(os.getcwd())
import cv2
import argparse
import demo_for_landms
import numpy as np
from models.ssd.prior_boxes import PriorBoxes
from models.onnx_model import ONNXModel
from utils import image_processing, file_processing
from models import box_utils

# class_names = ["BACKGROUND", "face", "person"]
# class_names = ["BACKGROUND", "person"]
class_names = ["BACKGROUND", "face"]


def get_parser():
    input_size = [300, 300]  # [W,H]
    priors_type = "face"
    # model_path = "data/pretrained/onnx/mode-face-person-640-360.onnx"
    # model_path = "data/pretrained/onnx/rfb1.0_face_person_640_360.onnx"
    model_path = "data/pretrained/onnx/rfb_landms1.0_face_416_416.opt.onnx"
    # priors_type = "face"
    image_dir = "data/test_images/0.jpg"
    parser = argparse.ArgumentParser(description='detect_imgs')
    parser.add_argument('--model_path', default=model_path, type=str, help='model_path')
    parser.add_argument('--net_type', default="mbv2", type=str,
                        help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
    parser.add_argument('--input_size', nargs='+', help="--input size 112 112", type=int, default=input_size)
    parser.add_argument('--prob_threshold', default=0.5, type=float, help='score threshold')
    parser.add_argument('--iou_threshold', default=0.3, type=float, help='iou_threshold')
    parser.add_argument('--candidate_size', default=200, type=int, help='nms candidate size')
    parser.add_argument('--image_dir', default=image_dir, type=str, help='directory or image path')
    parser.add_argument('--device', default="cuda:0", type=str, help='cuda:0 or cpu')
    parser.add_argument('--priors_type', default=priors_type, type=str, help='priors type:face or person')
    args = parser.parse_args()
    return args


class Detector(demo_for_landms.Detector):
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
                 prob_threshold=0.65,
                 iou_threshold=0.3,
                 freeze_header=False,
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
        self.prior_boxes = PriorBoxes(self.input_size, priors_type=self.priors_type, freeze_header=freeze_header)
        self.image_size = self.prior_boxes.input_size
        self.image_mean = self.prior_boxes.image_mean
        self.image_std = self.prior_boxes.image_std
        self.net = self.build_net(model_path, self.net_type)
        if not self.prior_boxes.freeze_header:
            self.center_variance = self.prior_boxes.center_variance
            self.size_variance = self.prior_boxes.size_variance
            self.priors = self.prior_boxes.priors

    def build_net(self, model_path, net_type=None):
        """
        :param model_path:  path to model(*.pth) file
        :param net_type:  "RFB" (higher precision) or "slim" (faster)'
        :return:
        """
        net = ONNXModel(model_path)
        return net

    def forward(self, image_tensor):
        """
        :param image_tensor:
        :return:
        """
        scores, boxes, ldmks = self.net.forward(image_tensor)
        if not self.prior_boxes.freeze_header:
            import torch
            # scores = F.softmax(scores, dim=2)
            # boxes = locations  # this line should be added.
            boxes = torch.from_numpy(boxes)
            ldmks = torch.from_numpy(ldmks)
            boxes = box_utils.convert_locations_to_boxes(boxes,
                                                         self.priors,
                                                         self.center_variance,
                                                         self.size_variance)
            ldmks = box_utils.decode_landms(ldmks, self.priors,
                                            variances=[self.center_variance,
                                                       self.size_variance])
            boxes = box_utils.center_form_to_corner_form(boxes)
            boxes = np.asarray(boxes)
            ldmks = np.asarray(ldmks)
        return scores, boxes, ldmks

    def predict(self, rgb_image, top_k=-1, prob_threshold=None, iou_threshold=None):
        """
        :param rgb_image: RGB Image
        :param top_k: keep top_k results. If k <= 0, keep all the results.
        :param prob_threshold:
        :param iou_threshold:
        :return:
        """
        height, width, _ = rgb_image.shape
        image_tensor = self.pre_process(rgb_image)
        scores, boxes,landms = self.forward(image_tensor)
        boxes = boxes[0]
        scores = scores[0]
        landms = landms[0]
        if not prob_threshold:
            prob_threshold = self.prob_threshold
        if not iou_threshold:
            iou_threshold = self.iou_threshold
        boxes, labels, probs, landms = self.post_process(boxes, scores, landms, width, height, top_k, prob_threshold,
                                                         iou_threshold)
        # boxes, scores, landms = self.adapter_bbox_score_landmarks(dets, landms)
        if len(boxes) == 0:
            return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
        face_index = labels == self.class_names.index("face")
        landms[~face_index, :] = 0
        return boxes, labels, probs, landms

if __name__ == "__main__":
    args = get_parser()
    print(args)
    net_type = args.net_type
    input_size = args.input_size
    priors_type = args.priors_type
    device = args.device
    image_dir = args.image_dir
    model_path = args.model_path
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
    det.detect_image_dir(image_dir, isshow=True)
