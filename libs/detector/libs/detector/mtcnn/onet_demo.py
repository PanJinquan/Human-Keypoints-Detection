# -*-coding: utf-8 -*-
import os, sys

sys.path.append(os.path.dirname(__file__))
import copy
import numpy as np
import cv2
import PIL.Image as Image
from libs.detector.mtcnn.net import onet_landmark


def show_landmark(win_name, img, landmarks_list):
    '''
    显示landmark
    :param win_name:
    :param image:
    :param landmarks_list: [[x1, y1], [x2, y2]]
    :param boxes:     [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
    :return:
    '''
    image = copy.deepcopy(img)
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 4  # 可以为 0 、4、8
    for landmarks in landmarks_list:
        for landmark in landmarks:
            # 要画的点的坐标
            point = (int(landmark[0]), int(landmark[1]))
            cv2.circle(image, point, point_size, point_color, thickness)
    image = Image.fromarray(image)
    image.show(win_name)


def show_image_boxes(win_name, image, boxes_list):
    '''
    :param win_name:
    :param image:
    :param boxes_list:[[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
    :return:
    '''
    for box in boxes_list:
        x1, y1, x2, y2 = box
        point1 = (int(x1), int(y1))
        point2 = (int(x2), int(y2))
        cv2.rectangle(image, point1, point2, (0, 0, 255), thickness=2)
    image = Image.fromarray(image)
    image.show(win_name)


def main():
    image_path = "./test.jpg"
    onet_path = "./XMC2-landmark-detection.pth.tar"
    image = Image.open(image_path)
    image = np.array(image)
    # bbox=[xmin,ymin,xmax,ymax]
    face_bbox = [69, 58, 173, 201]
    xmin, ymin, xmax, ymax = face_bbox
    # cut face ROI
    face = image[ymin:ymax, xmin:xmax]
    # show face
    Image.fromarray(face).show("face")
    # show face bbox
    show_image_boxes("image", image, [face_bbox])
    # init landmark Detection
    lmdet = onet_landmark.ONetLandmark(onet_path, device="cuda:0")
    landmarks = lmdet.get_faces_landmarks([face])
    # show face landmarks
    show_landmark("face-landmark", face, landmarks)
    print("landmarks:{}".format(landmarks))


if __name__ == "__main__":
    main()
