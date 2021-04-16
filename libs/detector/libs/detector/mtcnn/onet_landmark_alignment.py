# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-Face-Recognize-Pipeline
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2019-12-31 09:11:25
# --------------------------------------------------------
"""
import numpy as np
from utils import image_processing
from libs.detector.mtcnn.net import box_utils, onet_landmark
from libs.alignment import align_trans


class ONetLandmarkDet(onet_landmark.ONetLandmark):
    def __init__(self, device):
        super(ONetLandmarkDet, self).__init__(onet_path=None, device=device)

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

    def face_alignment(self, faces, face_resize):
        landmarks = self.get_faces_landmarks(faces)
        faces = align_trans.get_alignment_faces_list_resize(faces, landmarks, face_size=face_resize)
        # image_processing.cv_show_image("image", faces[0])
        return faces


if __name__ == "__main__":
    # img = Image.open('some_img.jpg')  # modify the image path to yours
    # bounding_boxes, landmarks = detect_faces(img)  # detect bboxes and landmarks for all faces in the image
    # show_results(img, bounding_boxes, landmarks)  # visualize the results
    # image_path = "/media/dm/dm/project/dataset/face_recognition/NVR/JPEGImages/2000.jpg"
    image_path1 = "/media/dm/dm1/FaceRecognition/torch-Face-Recognize-Pipeline/data/test_images/face1.jpg"
    image_path2 = "/media/dm/dm1/FaceRecognition/torch-Face-Recognize-Pipeline/data/test_images/face2.jpg"

    face1 = image_processing.read_image(image_path1, colorSpace="RGB", resize_height=200, resize_width=100)
    face2 = image_processing.read_image(image_path2, colorSpace="RGB", resize_height=100, resize_width=200)
    lmdet = ONetLandmarkDet(device="cuda:0")
    # image_processing.cv_show_image("face", face)
    # image_processing.show_landmark_boxes("image", image, landmarks, bboxes)
    faces = []
    faces.append(face1)
    faces.append(face2)
    landmarks = lmdet.get_faces_landmarks(faces)
    alig_faces = lmdet.face_alignment(faces, face_resize=[112, 112])
    for i in range(len(faces)):
        image_processing.show_landmark("landmark", faces[i], [landmarks[i]])
        image_processing.cv_show_image("alig_faces", alig_faces[i])
