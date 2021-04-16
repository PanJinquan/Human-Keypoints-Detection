import cv2
import numpy as np
import torch
from ..utils import box_utils


class Predictor(object):
    def __init__(self,
                 net,
                 input_size,
                 image_mean=0.0,
                 image_std=1.0,
                 nms_method=None,
                 iou_threshold=0.3,
                 prob_threshold=0.01,
                 candidate_size=200,
                 sigma=0.5,
                 device=None):
        """
        :param net:
        :param input_size:
        :param image_mean:
        :param image_std:
        :param nms_method:
        :param iou_threshold:
        :param prob_threshold:
        :param candidate_size:
        :param sigma:
        :param device:
        """
        self.net = net
        self.iou_threshold = iou_threshold
        self.prob_threshold = prob_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method
        self.sigma = sigma

        self.input_size = input_size
        self.mean = image_mean
        self.std = image_std
        # self.transform = PredictionTransform(size, mean, std)
        self.device = device
        self.net.to(self.device)
        self.net.eval()

    def forward(self, image_tensor):
        """
        :param image_tensor:
        :return:
        """
        with torch.no_grad():
            scores, boxes = self.net.forward(image_tensor)
        return scores, boxes

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
        image -= self.mean
        image /= self.std
        image = image.transpose(2, 0, 1)  # HWC->CHW
        image_tensor = torch.from_numpy(image)
        # image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    def post_process(self, boxes, scores, width, height, top_k, prob_threshold, iou_threshold):
        """
        :param boxes:
        :param scores:
        :param width: orig image width
        :param height:orig image height
        :param top_k: keep top_k results. If k <= 0, keep all the results.
        :param prob_threshold:
        :param iou_threshold:
        :return:
        """
        # this version of nms is slower on GPU, so recommend move data to CPU.
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = box_utils.nms(box_probs,
                                      self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if len(picked_box_probs) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]

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
        image_tensor = image_tensor.to(self.device)
        scores, boxes = self.forward(image_tensor)
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.prob_threshold
        if not iou_threshold:
            iou_threshold = self.iou_threshold
        boxes, labels, probs = self.post_process(boxes, scores, width, height, top_k, prob_threshold, iou_threshold)
        return boxes, labels, probs
