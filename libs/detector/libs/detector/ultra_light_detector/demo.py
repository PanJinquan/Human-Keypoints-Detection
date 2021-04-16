# -*- coding: utf-8 -*-

"""
This code is used to batch detect images in a folder.
"""
import sys
import os
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)
import cv2
import argparse
import torch
import numpy as np
from nets import nets
from nets.ssd.prior_boxes import PriorBoxes
from nets.nms import py_bbox_nms
from utils import image_processing, file_processing, debug
from nets import box_utils

class_names = ["BACKGROUND", "face", "person"]


def get_parser():
    input_size = [640, 360]  # [W,H]
    priors_type = "face_person"
    model_path = "data/pretrained/pth/mbv2_face_person_640_360.pth"
    model_path = os.path.join(project_root, model_path)
    image_dir = "data/test_images"
    parser = argparse.ArgumentParser(description='detect_imgs')
    parser.add_argument('--model_path', default=model_path, type=str, help='model_path')
    parser.add_argument('--net_type', default="mbv2", type=str,
                        help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
    parser.add_argument('--input_size', nargs='+', help="--input size 112 112", type=int, default=input_size)
    parser.add_argument('--prob_threshold', default=0.65, type=float, help='score threshold')
    parser.add_argument('--iou_threshold', default=0.3, type=float, help='iou_threshold')
    parser.add_argument('--candidate_size', default=200, type=int, help='nms candidate size')
    parser.add_argument('--image_dir', default=image_dir, type=str, help='directory or image path')
    parser.add_argument('--device', default="cuda:0", type=str, help='cuda:0 or cpu')
    parser.add_argument('--priors_type', default=priors_type, type=str, help='priors type:face or person')
    args = parser.parse_args()
    return args


class Detector(object):
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
                 freeze_header=True,
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
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.priors_type = priors_type
        self.candidate_size = candidate_size
        self.iou_threshold = iou_threshold
        self.prob_threshold = prob_threshold
        self.device = device
        self.prior_boxes = PriorBoxes(self.input_size, priors_type=self.priors_type, freeze_header=freeze_header)
        self.image_size = self.prior_boxes.input_size
        self.image_mean = self.prior_boxes.image_mean
        self.image_std = self.prior_boxes.image_std
        self.net = self.build_net(model_path, self.net_type)
        if not self.prior_boxes.freeze_header:
            self.center_variance = self.prior_boxes.center_variance
            self.size_variance = self.prior_boxes.size_variance
            self.priors = self.prior_boxes.priors.to(self.device)

    def build_net(self, model_path, net_type):
        """
        :param model_path:  path to model(*.pth) file
        :param net_type:  "RFB" (higher precision) or "slim" (faster)'
        :return:
        """
        net = nets.build_net(net_type,
                             prior_boxes=self.prior_boxes,
                             num_classes=self.num_classes,
                             width_mult=1.0,
                             is_test=True,
                             device=self.device)
        # torch_tools.summary_model(net, batch_size=1, input_size=input_size, device=device)
        net.load(model_path, strict=True)
        net = net.to(self.device)
        net.eval()
        return net

    @debug.run_time_decorator("forward")
    def forward(self, image_tensor):
        """
        :param image_tensor:
        :return: scores: shape=([1, num_bboxes, num_class])
                 boxes:  shape=([1, num_bboxes, 4]),boxes=[[xmin,ymin,xmax,ymax]]
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            scores, boxes = self.net.forward(image_tensor)
            # feature = self.get_feature()
            if not self.prior_boxes.freeze_header:
                # scores = F.softmax(scores, dim=2)
                # boxes = locations  # this line should be added.
                boxes = box_utils.convert_locations_to_boxes(boxes,
                                                             self.priors,
                                                             self.center_variance,
                                                             self.size_variance)
                boxes = box_utils.center_form_to_corner_form(boxes)
        return scores, boxes

    def get_feature(self):
        feature = self.net.get_feature()
        return feature

    @debug.run_time_decorator("pre_process")
    def pre_process(self, image):
        """
        self.mean = [127,127,127]
        self.std = [128]
        self.transform = PredictionTransform(size, mean, std)
        :param image:
        :return:
        """
        # image = self.transform(image)
        image = cv2.resize(image, (self.input_size[0], self.input_size[1]))
        image = image.astype(np.float32)
        image -= self.image_mean
        image /= self.image_std
        image = image.transpose(2, 0, 1)  # HWC->CHW
        image_tensor = image[np.newaxis, :]
        return image_tensor

    @debug.run_time_decorator("post_process")
    def post_process(self, boxes, scores, width, height, top_k, prob_threshold, iou_threshold):
        """
        :param boxes:
        :param scores:
        :param width: orig image width
        :param height:orig image height
        :param top_k: keep top_k results. If k <= 0, keep all the results.
        :param prob_threshold:
        :param iou_threshold:
        :return: boxes, labels, probs

        """
        # this version of nms is slower on GPU, so recommend move data to CPU.
        if not isinstance(boxes, np.ndarray):
            boxes = boxes.data.cpu().numpy()
        if not isinstance(scores, np.ndarray):
            scores = scores.data.cpu().numpy()
        picked_boxes_probs = []
        picked_labels = []
        for class_index in range(1, scores.shape[1]):
            probs = scores[:, class_index]
            index = probs > prob_threshold
            subset_probs = probs[index]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[index, :]
            sub_boxes_probs = py_bbox_nms.per_class_nms(subset_boxes,
                                                        subset_probs,
                                                        prob_threshold=prob_threshold,
                                                        iou_threshold=iou_threshold,
                                                        top_k=top_k,
                                                        keep_top_k=self.candidate_size)
            picked_boxes_probs.append(sub_boxes_probs)
            picked_labels += [class_index] * sub_boxes_probs.shape[0]

        if len(picked_boxes_probs) == 0:
            return np.asarray([]), np.asarray([]), np.asarray([])
        picked_boxes_probs = np.concatenate(picked_boxes_probs)
        boxes = picked_boxes_probs[:, :4]
        probs = picked_boxes_probs[:, 4]
        # conver normalized coordinates to image coordinates
        # bboxes_scale = [width, height, width, height]
        image_size = [width, height]
        bboxes_scale = np.asarray(image_size * 2, dtype=np.float32)
        boxes = boxes * bboxes_scale
        labels = np.asarray(picked_labels)
        return boxes, labels, probs

    @debug.run_time_decorator("predict")
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
        image_tensor = torch.from_numpy(image_tensor)
        scores, boxes = self.forward(image_tensor)
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.prob_threshold
        if not iou_threshold:
            iou_threshold = self.iou_threshold
        boxes, labels, probs = self.post_process(boxes, scores, width, height, top_k, prob_threshold, iou_threshold)
        return boxes, labels, probs

    # @debug.run_time_decorator("detect_image")
    def detect_image(self, rgb_image, isshow=True):
        """
        :param rgb_image:  input RGB Image
        :param isshow:
        :return:
        """
        boxes, labels, probs = self.predict(rgb_image,
                                            iou_threshold=self.iou_threshold,
                                            prob_threshold=self.prob_threshold)
        if not isinstance(boxes, np.ndarray):
            boxes = boxes.detach().cpu().numpy()
        if not isinstance(labels, np.ndarray):
            labels = labels.detach().cpu().numpy()
        if not isinstance(probs, np.ndarray):
            probs = probs.detach().cpu().numpy()
        if isshow:
            print("boxes:{}\nlabels:{}\nprobs:{}".format(boxes, labels, probs))
            self.show_image(rgb_image, boxes, labels, probs)
        return boxes, labels, probs

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
            boxes, labels, probs = self.detect_image(rgb_image, isshow=isshow)

    def show_image(self, rgb_image, boxes, labels, probs, waitKey=0):
        boxes_name = ["{}:{:3.2f}".format(l, s) for l, s in zip(labels, probs)]
        rgb_image = image_processing.draw_image_detection_bboxes(rgb_image, boxes, probs, labels)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Det", bgr_image)
        cv2.imwrite("result.jpg", bgr_image)
        cv2.waitKey(waitKey)


if __name__ == "__main__":
    args = get_parser()
    print(args)
    net_type = args.net_type
    input_size = args.input_size
    priors_type = args.priors_type
    device = args.device
    # model_path = "RFB-person.pth"
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
    # det.detect_image_dir(image_dir, isshow=False)
    # det.detect_image_dir(image_dir, isshow=False)
    # det.detect_image_dir(image_dir, isshow=False)
    # det.detect_image_dir(image_dir, isshow=False)
    # det.detect_image_dir(image_dir, isshow=False)
    # det.detect_image_dir(image_dir, isshow=False)
    # det.detect_image_dir(image_dir, isshow=False)
    # det.detect_image_dir(image_dir, isshow=False)
    det.detect_image_dir(image_dir, isshow=True)
