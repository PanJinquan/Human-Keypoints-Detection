# -*-coding: utf-8 -*-
"""
    @Project: torch-Face-Recognize-Pipeline
    @File   : onet.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-09-25 16:42:16
"""
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from libs.detector.mtcnn.net import box_utils
from collections import OrderedDict


class ONetLandmark():
    def __init__(self, onet_path, device):
        '''
        :param onet_path: model path
        :param device: cuda or cpu
        '''
        self.onet = ONet(onet_path).to(device)
        self.onet.eval()
        self.device = device

    def get_faces_landmarks(self, faces):
        input_tensor = []
        bounding_boxes = []
        for face in faces:
            h, w, d = np.shape(face)
            xmin, ymin = 0, 0
            # resize image for net inputs
            input_face = cv2.resize(face, (48, 48))
            input_face = box_utils._preprocess(input_face)
            input_face = torch.FloatTensor(input_face)
            input_tensor.append(input_face)
            bounding_boxes.append([xmin, ymin, w, h])

        input_tensor = torch.cat(input_tensor).to(self.device)
        output = self.onet(input_tensor)
        landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
        # offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
        # probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]
        # compute landmark points in src face
        # landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        # landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]
        bounding_boxes = np.asarray(bounding_boxes)
        widths = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0  # xmax-xmin
        heights = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0  # ymax-ymin
        xmins, ymins = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmins, 1) + np.expand_dims(widths, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymins, 1) + np.expand_dims(heights, 1) * landmarks[:, 5:10]
        # adapter landmarks
        landmarks_list = []
        for landmark in landmarks:
            face_landmarks = [[landmark[j], landmark[j + 5]] for j in range(5)]
            landmarks_list.append(face_landmarks)
        return landmarks_list


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)


class ONet(nn.Module):

    def __init__(self, onet_path):
        super(ONet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(64, 64, 3, 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', nn.Conv2d(64, 128, 2, 1)),
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(1152, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

        weights = np.load(os.path.dirname(__file__) + "/onet.npy", allow_pickle=True)[()]
        # weights = np.load(onet_path, allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        a = F.softmax(a, dim=1)
        return c, b, a
