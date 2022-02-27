# -*-coding: utf-8 -*-
"""
    @Project: face.evoLVe.PyTorch
    @File   : mtcnn.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-23 13:57:41
"""
import sys
import os

sys.path.append(os.getcwd())
import numpy as np
import PIL.Image as Image
import torch
from torch.autograd import Variable
from net.get_nets import PNet, RNet, ONet
from net.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from net.first_stage import run_first_stage
from net import box_utils
from utils import image_processing


class MTCNN():
    def __init__(self, min_face_size=20.0, thresholds=[0.6, 0.7, 0.8], nms_thresholds=[0.7, 0.7, 0.7], device="cuda:0"):
        # LOAD MODELS
        self.pnet = PNet().to(device)
        self.rnet = RNet().to(device)
        self.onet = ONet().to(device)
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.nms_thresholds = nms_thresholds
        self.device = device

    def detect_cpu(self, rgb_image):
        if isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image)
        # BUILD AN IMAGE PYRAMID
        width, height = rgb_image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / self.min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []

        # run P-Net on different scales
        for s in scales:
            boxes = run_first_stage(rgb_image, self.pnet, scale=s, threshold=self.thresholds[0], device=self.device)
            bounding_boxes.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        if not bounding_boxes:
            return [], []
        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], self.nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        # use offsets predicted by pnet to transform bounding boxes
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2

        img_boxes = get_image_boxes(bounding_boxes, rgb_image, size=24)
        img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        output = self.rnet(img_boxes)
        offsets = output[0].data.numpy()  # shape [n_boxes, 4]
        probs = output[1].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, self.nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3

        img_boxes = get_image_boxes(bounding_boxes, rgb_image, size=48)
        if len(img_boxes) == 0:
            return [], []
        img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        output = self.onet(img_boxes)
        landmarks = output[0].data.numpy()  # shape [n_boxes, 10]
        offsets = output[1].data.numpy()  # shape [n_boxes, 4]
        probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, self.nms_thresholds[2], mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]

        return bounding_boxes, landmarks

    def detect_image(self, rgb_image):
        """
        Arguments:
            rgb_image: an instance of PIL.Image.
            min_face_size: a float number.
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.

        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """
        if isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image)
        # BUILD AN IMAGE PYRAMID
        width, height = rgb_image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / self.min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []

        with torch.no_grad():
            # run P-Net on different scales
            for s in scales:
                boxes = run_first_stage(rgb_image, self.pnet, scale=s, threshold=self.thresholds[0], device=self.device)
                bounding_boxes.append(boxes)

            # collect boxes (and offsets, and scores) from different scales
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            if len(bounding_boxes) == 0:
                return [], []
            bounding_boxes = np.vstack(bounding_boxes)

            keep = nms(bounding_boxes[:, 0:5], self.nms_thresholds[0])
            bounding_boxes = bounding_boxes[keep]

            # use offsets predicted by pnet to transform bounding boxes
            bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            # shape [n_boxes, 5]

            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 2

            img_boxes = get_image_boxes(bounding_boxes, rgb_image, size=24)
            img_boxes = torch.FloatTensor(img_boxes).to(self.device)

            output = self.rnet(img_boxes)
            offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > self.thresholds[1])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]

            keep = nms(bounding_boxes, self.nms_thresholds[1])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 3

            img_boxes = get_image_boxes(bounding_boxes, rgb_image, size=48)
            if len(img_boxes) == 0:
                return [], []
            img_boxes = torch.FloatTensor(img_boxes).to(self.device)
            output = self.onet(img_boxes)
            landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > self.thresholds[2])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]

            # compute landmark points
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, self.nms_thresholds[2], mode='min')
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]

        return bounding_boxes, landmarks

    def landmarks_forward(self, bounding_boxes, rgb_image):
        if isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image)
        # img_boxes = get_image_boxes(bounding_boxes, rgb_image, size=48)
        img_boxes = self.get_image_crop(bounding_boxes, rgb_image, size=48)
        if len(img_boxes) == 0:
            return [], []
        img_boxes = torch.FloatTensor(img_boxes).to(self.device)
        output = self.onet(img_boxes)
        landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]
        return landmarks

    def face_landmarks_forward(self, faces):
        input_tensor=[]
        for face in faces:
            height, width, depths = np.shape(face)
            xmin, ymin = 0, 0
            # resize image for net inputs
            input_face = image_processing.resize_image(face, 48, 48)
            input_face = box_utils._preprocess(input_face)
            input_face = torch.FloatTensor(input_face)
            input_tensor.append(input_face)

        input_tensor = torch.cat(input_tensor)
        input_tensor = torch.FloatTensor(input_tensor).to(self.device)
        output = self.onet(input_tensor)
        landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
        # compute landmark points in src face
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]
        # adapter landmarks
        landmarks_list = []
        for landmark in landmarks:
            face_landmarks = [[landmark[j], landmark[j + 5]] for j in range(5)]
            landmarks_list.append(face_landmarks)
        return landmarks_list

    def get_image_crop(self, bounding_boxes, image, size=48):
        if not isinstance(image, np.ndarray):
            rgb_image = np.asarray(image)
        else:
            rgb_image = image
        # resize
        bboxes = bounding_boxes[:, :4]
        scores = bounding_boxes[:, 4:]
        num_boxes = len(bboxes)
        img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')
        for i, box in enumerate(bboxes):
            img_box = image_processing.get_bboxes_image(rgb_image, [box], resize_height=size, resize_width=size)
            img_box = img_box[0]
            img_boxes[i, :, :, :] = box_utils._preprocess(img_box)
        return img_boxes

    @staticmethod
    def adapter_bbox_score_landmarks(bbox_score, landmarks):
        landmarks_list = []
        for landmark in landmarks:
            face_landmarks = [[landmark[j], landmark[j + 5]] for j in range(5)]
            landmarks_list.append(face_landmarks)
        landmarks = np.asarray(landmarks_list)
        bbox_score = np.asarray(bbox_score)
        bboxes = bbox_score[:, :4]
        scores = bbox_score[:, 4:]
        return bboxes, scores, landmarks

    @staticmethod
    def adapter_bbox_score(bbox_score):
        bbox_score = np.asarray(bbox_score)
        bboxes = bbox_score[:, :4]
        scores = bbox_score[:, 4:]
        return bboxes, scores

    @staticmethod
    def adapter_landmarks(landmarks):
        landmarks_list = []
        for landmark in landmarks:
            face_landmarks = [[landmark[j], landmark[j + 5]] for j in range(5)]
            landmarks_list.append(face_landmarks)
        landmarks = np.asarray(landmarks_list)
        return landmarks


if __name__ == "__main__":
    # img = Image.open('some_img.jpg')  # modify the image path to yours
    # bounding_boxes, landmarks = detect_faces(img)  # detect bboxes and landmarks for all faces in the image
    # show_results(img, bounding_boxes, landmarks)  # visualize the results
    # image_path = "/media/dm/dm/project/dataset/face_recognition/NVR/JPEGImages/2000.jpg"
    image_path = "/media/dm/dm1/git/python-learning-notes/dataset/dataset/A/test1.jpg"

    image = image_processing.read_image(image_path, colorSpace="RGB")
    mt = MTCNN()
    bbox_score, landmarks = mt.detect_image(image)
    bboxes, scores, landmarks = mt.adapter_bbox_score_landmarks(bbox_score, landmarks)
    # image_processing.show_image_boxes("image",image,bboxes)
    # image_processing.show_landmark_boxes("image", image, landmarks, bboxes)
    # image_processing.show_landmark_boxes("image2", image, landmarks, bboxes)
    faces = image_processing.get_bboxes_image(image, bboxes)
    # landmarks2 = mt.landmarks_forward(bbox_score, image)
    # bboxes, scores, landmarks2 = mt.adapter_bbox_score_landmarks(bbox_score, landmarks2)
    # image_processing.show_landmark_boxes("image2", image, landmarks2, bboxes)

    for face in faces:
        image_processing.cv_show_image("face", face)
        image_processing.show_landmark_boxes("image", image, landmarks, bboxes)
        landmarks = mt.face_landmarks_forward([face])
        image_processing.show_landmark("landmark", face, landmarks)
