# -*-coding: utf-8 -*-
"""
    @Project: torch-Human-Pose-Estimation-Pipeline
    @File   : demo.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-11-08 15:02:19
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))
import numpy as np
import cv2
import copy
import torch
import torchvision.transforms as transforms
from easydict import EasyDict as edict
from configs import val_config
from models.tools.transforms import get_affine_transform
from models.core.inference import get_final_preds, get_final_preds_offset
from models.nets.build_nets import build_nets
from utils import image_processing, debug, file_processing, torch_tools

project_root = os.path.dirname(__file__)


class PoseEstimation():
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
        self.config = edict(config)
        print(self.config)
        self.threshhold = threshhold
        self.device = device
        # coco_skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
        #                  [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
        # self.skeleton = coco_skeleton
        self.skeleton = self.config.TEST.skeleton
        # self.skeleton = custom_mpii_skeleton
        self.input_size = tuple(self.config.MODEL.IMAGE_SIZE)  # w,h
        self.net_type = self.config.MODEL.NAME
        self.model_path = self.config.TEST.MODEL_FILE
        if not model_path:
            self.model_path = self.config.TEST.MODEL_FILE
        self.transform = self.get_transforms()
        # self.model_path = os.path.join(project_root, self.model_path)
        self.model = self.build_model(self.net_type, self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_transforms(self):
        """
        input_tensor = image_processing.image_normalization(image,
                                                            mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
        input_tensor = input_tensor.transpose(2, 0, 1)  # [H0,W1,C2]-[C,H,W]
        input_tensor = torch.from_numpy(input_tensor)
        :return:
        """
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # b,g,r
        #                          std=[0.229, 0.224, 0.225]),
        # ])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return transform

    def build_model(self, net_type, model_path):
        """
        build model
        :param net_type:
        :param model_path:
        :return:
        """
        model = build_nets(net_type=net_type, config=self.config, is_train=False)
        state_dict = torch_tools.load_state_dict(model_path, module=False)
        model.load_state_dict(state_dict)
        return model

    @debug.run_time_decorator("detect")
    def detect(self, image, bboxes, threshhold=0.3):
        """
        image:image
        box:  [xmin, ymin, xmax, ymax]
        """
        kp_points, kp_scores = [], []
        for box in bboxes:
            points, scores = self.inference(image, box, threshhold)
            kp_points.append(points)
            kp_scores.append(scores)
        return kp_points, kp_scores

    def get_transform(self, image):
        input_tensor = self.transform(image)
        return input_tensor

    def inference(self, bgr_image, box, threshhold=0.1):
        """
        input_tensor = image_processing.image_normalization(image,
                                                             mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
        input_tensor = input_tensor.transpose(2, 0, 1)  # [H0,W1,C2]-[C,H,W]
        input_tensor = torch.from_numpy(input_tensor)
        :param bgr_image:
        :param body_rect:
        :param threshhold:
        :return:
        """
        # image, center, scale = self.pre_process3(bgr_image, body_rect)
        # image, center, scale = self.pre_process2(bgr_image, box)
        image, center, scale = self.get_input_center_scale(bgr_image, box)
        # image_processing.show_image_rects("body_rect", bgr_image, [body_rect])
        # image_processing.cv_show_image("image", image, waitKey=0)
        input_tensor = self.transform(image)
        # input_tensor = self.get_transform(image)
        # input_tensor = np.asarray(image / 255.0, dtype=np.float32)
        # input_tensor = image_processing.image_normalization(image,
        #                                                     mean=[0.485, 0.456, 0.406],
        #                                                     std=[0.229, 0.224, 0.225])
        # input_tensor = input_tensor.transpose(2, 0, 1)  # [H0,W1,C2]-[C,H,W]
        # input_tensor = torch.from_numpy(input_tensor)
        input_tensor = input_tensor.unsqueeze(0)
        output = self.forward(input_tensor)
        output = output.clone().cpu().numpy()
        key_point, kp_score = self.post_process(bgr_image, output, center, scale, threshhold)
        return key_point, kp_score

    @debug.run_time_decorator("forward")
    def forward(self, input_tensor):
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output_tensor = self.model(input_tensor)
        return output_tensor

    def __get_crop_images(self, image, box):
        crop_image = image_processing.get_bboxes_image(image, bboxes_list=[box])[0]
        # image_processing.cv_show_image("person", crop_image, waitKey=0)
        crop_image = image_processing.resize_image(crop_image, resize_height=self.input_size[0])
        crop_image = image_processing.center_crop_padding(crop_image, crop_size=self.input_size)
        return crop_image

    def get_input_center_scale(self, image, box):
        '''
        :param image: 图像
        :param box: 检测框
        :return: 截取的当前检测框图像，中心坐标及尺寸
        '''
        aspect_ratio = 0.75
        pixel_std = 200

        def _box2cs(box):
            x = box[0]
            y = box[1]
            w = box[2] - box[0]
            h = box[3] - box[1]
            return _xywh2cs(x, y, w, h)

        def _xywh2cs(x, y, w, h):
            center = np.zeros((2), dtype=np.float32)
            center[0] = x + w * 0.5
            center[1] = y + h * 0.5

            if w > aspect_ratio * h:
                h = w * 1.0 / aspect_ratio
            elif w < aspect_ratio * h:
                w = h * aspect_ratio
            scale = np.array(
                [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
                dtype=np.float32)
            if center[0] != -1:
                scale = scale * 1.25
            return center, scale

        box = copy.deepcopy(box)
        center, scale = _box2cs(box)
        trans = get_affine_transform(center, scale, 0, self.input_size)
        input = cv2.warpAffine(image, trans, (self.input_size[0], self.input_size[1]), flags=cv2.INTER_LINEAR)
        return input, center, scale

    @debug.run_time_decorator("post_process")
    def post_process(self, bgr_image, heatmap, center, scale, threshhold):
        # compute coordinate
        key_point, kp_score = self.get_final_output(heatmap, center, scale, threshhold)
        # key_point, kp_score = get_final_preds(self.config, heatmap.clone().cpu().numpy(), np.asarray([center]),
        #                                    np.asarray([scale]))
        key_point = image_processing.points_protection(key_point,
                                                       height=bgr_image.shape[0],
                                                       width=bgr_image.shape[1])
        return key_point, kp_score

    def get_final_output(self, pred, center, scale, threshhold=0.0):
        # compute coordinate
        if self.config.MODEL.EXTRA.TARGET_TYPE == 'gaussian':
            key_point, kp_score = get_final_preds(self.config, pred, np.asarray([center]), np.asarray([scale]))
        else:
            self.config.LOSS = edict()
            self.config.LOSS.KPD = 4.0
            key_point, kp_score, _ = get_final_preds_offset(self.config, pred, np.asarray([center]),
                                                            np.asarray([scale]))
        key_point, kp_score = key_point[0, :], kp_score[0, :]
        # for custom_mpii_256_256 cal head coordinate
        # key_point[3, :] = (key_point[3, :] + key_point[2, :]) / 2
        # score[3] = (score[3] + score[2]) / 2
        index = kp_score < threshhold
        index = index.reshape(-1)
        key_point[index, :] = (0, 0)
        key_point = np.abs(key_point)
        return key_point, kp_score

    @staticmethod
    def center_scale2rect(center, scale, pixel_std=200):
        w = pixel_std * scale[0]
        h = pixel_std * scale[1]
        x = center[0] - 0.5 * w
        y = center[1] - 0.5 * h
        rect = [x, y, w, h]
        return rect

    @staticmethod
    def adjust_center_scale(center, scale, alpha=15.0, beta=1.25, type="center_default"):
        '''
         Adjust center/scale slightly to avoid cropping limbs
        if  c[0] != -1:
            c[1] = c[1] + 15 * s[1]
            s = s * 1.25
        :param center:
        :param scale:官方的说法是：person scale w.r.t. 200 px height,理解是这个scale=图片中人体框的高度/200
        :param alpha:
        :param beta:
        :return:
        '''
        if center[0] != -1:
            if type == "center_up":
                rect = PoseEstimation.center_scale2rect(center, scale)
                x, y, w, h = rect
                center[0] = x + w * 0.5
                center[1] = y + h * 0.5 + alpha * h
                scale = scale * beta
            elif type == "center_default":
                center[1] = center[1] + alpha * scale[1]
                scale = scale * beta
            elif type == "center":
                center = center
                scale = scale
        return center, scale

    def show_result(self, image, boxes, kp_points, kp_scores, skeleton=None, waitKey=0):
        if not skeleton:
            skeleton = self.skeleton
        image = self.draw_keypoints(image, boxes, kp_points, kp_scores, skeleton)
        cv2.imshow('frame', image)
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


if __name__ == '__main__':
    MODEL_FILE = "work_space/body/person_coco/model_mobilenet_v2_1.0_11_192x256_128_gaussian_person_coco_2021-04-07-18-14/model/best_model_182_0.6597.pth",
    pose = PoseEstimation(val_config.body_coco_192_256, MODEL_FILE, device="cuda:0")
    image_path = "../data/test_images/test1.jpg"
    image = cv2.imread(image_path)
    boxes = [[0, 0, 800, 800]]
    kp_points, kp_scores = pose.detect(image, boxes, threshhold=0.1)
    print(kp_points)
    print(kp_scores)
    pose.show_result(image, boxes, kp_points, kp_scores)
