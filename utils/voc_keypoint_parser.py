# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-05 11:01:49
# --------------------------------------------------------
"""
import os
import xmltodict
import numpy as np
import cv2
import glob
import random
from tqdm import tqdm
from utils import file_processing
from utils.voc_parser import VOCDataset, ConcatDataset
# from models.dataloader.voc_parser import PolygonParser, ConcatDataset
from utils import geometry_tools


class VOCKeyPointDataset(VOCDataset):

    def __init__(self,
                 filename=None,
                 data_root=None,
                 anno_dir=None,
                 image_dir=None,
                 class_names=None,
                 transform=None,
                 target_transform=None,
                 color_space="RGB",
                 keep_difficult=False,
                 shuffle=False,
                 check=False):
        """
        :param filename:
        :param data_root:
        :param anno_dir:
        :param image_dir:
        :param transform:
        :param color_space:
        :param keep_difficult:
        :param shuffle:
        """
        self.target_transform = target_transform
        super().__init__(filename=filename,
                         data_root=data_root,
                         anno_dir=anno_dir,
                         image_dir=image_dir,
                         class_names=class_names,
                         transform=transform,
                         color_space=color_space,
                         keep_difficult=keep_difficult,
                         shuffle=shuffle,
                         check=check)

    def checking(self, image_ids: list, ignore_empty=True):
        """
        :param image_ids:
        :param ignore_empty : 是否去除一些空数据
        :return:
        """
        dst_ids = []
        # image_id = image_id[:100]
        # image_ids = image_ids[100:]
        for image_id in tqdm(image_ids):
            image_file, annotation_file = self.get_image_anno_file(image_id)
            if not os.path.exists(annotation_file):
                continue
            if not os.path.exists(image_file):
                continue
            bboxes, labels, keypoints, is_difficult = self.get_annotation(annotation_file)
            if not self.keep_difficult:
                bboxes = bboxes[is_difficult == 0]
                # labels = labels[is_difficult == 0]
            if ignore_empty and (len(bboxes) == 0 or len(labels) == 0):
                print("illegal annotation:{}".format(annotation_file))
                continue
            dst_ids.append(image_id)
        print("have nums image:{},legal image:{}".format(len(image_ids), len(dst_ids)))
        return dst_ids

    def fileter_landms(self, boxes, labels, landms):
        for i in range(len(boxes)):
            landm = landms[i, :]
            # if landm=-1,label=-1
            if landm[0] < 0:
                landms[i, :] = 0.0
                # labels[i] = -1.0
                # labels[i] = 0.0
                if labels[i] == 1:
                    labels[i] = -1
        # keypoints[:, 0::2] /= width
        # keypoints[:, 1::2] /= height
        return boxes, labels, landms

    def __getitem__(self, index):
        """
        :param index: int or str
        :return:rgb_image
        """
        # print("image_id:{}".format(image_id))
        # image_id = "DN0000063_00.png"
        image_file, annotation_file = self.get_image_anno_file(index)
        # bboxes, labels, is_difficult = self.get_annotation(annotation_file)
        bboxes, labels, keypoints, is_difficult = self.get_annotation(annotation_file)
        angles = self.get_keypoint_angles(keypoints)
        image = self.read_image(image_file, color_space=self.color_space)
        if self.transform:
            # rgb_image, bboxes, labels = self.transform(rgb_image, bboxes, labels)
            bboxes, labels, keypoints = self.fileter_landms(bboxes, labels, keypoints)
            image, bboxes, labels, keypoints = self.transform(image, bboxes, labels, landms=keypoints)
            keypoints = keypoints["keypoints"]
        if self.target_transform:
            bboxes, labels, keypoints = self.target_transform(bboxes, labels, keypoints)
        angles = image_processing.data_normalization(angles, omin=0, omax=1, imin=-180, imax=180)
        # return image, bboxes, labels, angles
        bboxes_labels = np.hstack((bboxes, labels, angles))
        return image, bboxes_labels

    def get_annotation(self, xml_file):
        """
        keypoint关节点的格式 : [x_1, y_1, v_1,...,x_k, y_k, v_k]
        其中x,y为Keypoint的坐标，v为可见标志
            v = 0 : 未标注点
            v = 1 : 标注了但是图像中不可见（例如遮挡）
            v = 2 : 标注了并图像可见
        :param xml_file:
        :param class_vertical_formula: class_vertical_formula = {class_name: i for i, class_name in enumerate(class_names)}
        :return:
        """
        try:
            content = self.read_xml2json(xml_file)
            annotation = content["annotation"]
            # get image shape
            width = int(annotation["size"]["width"])
            height = int(annotation["size"]["height"])
            depth = int(annotation["size"]["depth"])
            filename = annotation["filename"]
            objects = annotation["object"]
        except Exception as e:
            print("illegal annotation:{}".format(xml_file))
            objects = []
        objects_list = []
        if not isinstance(objects, list):
            objects = [objects]
        for object in objects:
            name = object["name"]
            if self.class_names and name not in self.class_names:
                continue
            difficult = int(object["difficult"])
            xmin = float(object["bndbox"]["xmin"])
            xmax = float(object["bndbox"]["xmax"])
            ymin = float(object["bndbox"]["ymin"])
            ymax = float(object["bndbox"]["ymax"])
            # rect = [xmin, ymin, xmax - xmin, ymax - ymin]
            bbox = [xmin, ymin, xmax, ymax]
            # get person keypoints ,if exist
            if 'keypoints' in object:
                keypoints = [float(i) for i in object["keypoints"].split(",")]
            else:
                keypoints = [0] * 4
            item = {}
            item["bbox"] = bbox
            item["keypoints"] = keypoints
            item["difficult"] = difficult
            if self.class_dict:
                name = self.class_dict[name]
            item["name"] = name
            objects_list.append(item)
        boxes, labels, keypoints, is_difficult = self.get_objects_items(objects_list)
        return boxes, labels, keypoints, is_difficult

    def get_objects_items(self, objects_list):
        """
        :param objects_list:
        :return:
        """
        bboxes = []
        labels = []
        keypoints = []
        is_difficult = []
        for item in objects_list:
            bboxes.append(item["bbox"])
            labels.append(item['name'])
            keypoints.append(item['keypoints'])
            is_difficult.append(item['difficult'])
        bboxes = np.array(bboxes, dtype=np.float32)
        # labels = np.array(labels, dtype=np.int64)
        labels = np.asarray(labels).reshape(-1, 1)
        keypoints = np.array(keypoints, dtype=np.float32)
        is_difficult = np.array(is_difficult, dtype=np.uint8)
        return bboxes, labels, keypoints, is_difficult

    def get_keypoint_angles(self, keypoints):
        angles = []
        for pair in keypoints:
            point0, point1, angle = self.get_angle(pair)
            angles.append(angle)
        angles = np.asarray(angles, dtype=np.float32).reshape(-1, 1)
        return angles

    def get_cos_sin_values(self, angles):
        """
        :param angles: (num_boxes,2)=[[cos0,sin0],[cos1,sin1],...]
        :return:
        """
        c = np.cos(angles * np.pi / 180.0)
        s = np.sin(angles * np.pi / 180.0)
        out = np.hstack((c, s))
        return out

    @staticmethod
    def get_angle(pair_points):
        assert len(pair_points) == 4
        if isinstance(pair_points, np.ndarray):
            pair_points = pair_points.tolist()
        point0 = pair_points[0:2]
        point1 = pair_points[2:4]
        point0_ = geometry_tools.image2plane_coordinates([point0])[0]
        point1_ = geometry_tools.image2plane_coordinates([point1])[0]
        angle = geometry_tools.compute_horizontal_angle(point1_, point0_)
        return point0, point1, angle


def draw_pair_points_bbox_angle(image, bboxes, labels, angels=None, keypoints_pair=None):
    """
    :param image:
    :param bboxes:
    :param labels:
    :param keypoints_pair:[point1,point2]
    :return:
    """

    bboxes = np.asarray(bboxes)
    labels = np.asarray(labels).reshape(-1).tolist()
    if keypoints_pair is not None:
        keypoints_pair = np.asarray(keypoints_pair)
    print("==" * 10)
    print(angels)
    if angels is not None:
        angels = np.asarray(angels)
        angels = image_processing.data_normalization(angels, omin=-180, omax=180, imin=0, imax=1)
    pointline = [(0, 1)]
    angle = None
    for i in range(len(labels)):
        box = bboxes[i, :]
        image = image_processing.draw_image_bboxes_text(image, [box], [labels[i]])
        height, width, d = image.shape
        # image_processing.show_image_boxes(None, image, joints_bbox, color=(255, 0, 0))
        if keypoints_pair is not None:
            point0, point1, angle = VOCKeyPointDataset.get_angle(keypoints_pair[i, :])
            image = image_processing.draw_key_point_arrowed_in_image(image, [[point1, point0]], pointline=pointline)
            print(angle)
        if angels is not None:
            angle = angels[i]
        start = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        end = [box[2], (box[1] + box[3]) / 2]
        end = geometry_tools.rotate_point(end, start, angle, height)
        image = image_processing.draw_key_point_arrowed_in_image(image, [[start, end]],
                                                                 pointline=pointline,
                                                                 color=(255, 0, 0))
        vis_image = image_processing.resize_image(image, 1000)
        image_processing.cv_show_image("Det", vis_image, waitKey=0)


def angles_normalization():
    angle = -180
    norm_angele = image_processing.data_normalization(angle, omin=0, omax=1, imin=-180, imax=180)
    unnorm_angele = image_processing.data_normalization(norm_angele, omin=-180, omax=180, imin=0, imax=1)
    print("angle:{},norm_angele:{},unnorm_angele:{}".format(angle, norm_angele, unnorm_angele))


if __name__ == "__main__":
    from utils import image_processing, file_processing
    # from modules.image_transforms import data_transforms
    import torchvision
    import torch
    import torch.utils.data as torch_utils

    isshow = True
    data_root = "/media/dm/dm1/git/python-learning-notes/dataset/finger/"
    image_dir = data_root + "images"
    # data_root = "/home/dm/panjinquan3/dataset/wider_face_add_lm_10_10/"
    # image_dir = data_root + "JPEGImages"
    anno_dir = data_root + "Annotations"
    filenames = data_root + "file_id.txt"
    # class_names = ["face", "person"]
    # class_names = ["face"]
    # class_names = ["finger"]
    # json_dir = data_root + '/Annotations'
    shuffle = False
    # class_names = ["face"]
    class_names = None
    # anno_list = file_processing.get_files_list(json_dir, postfix=["*.xml"])
    # image_id_list = file_processing.get_files_id(anno_list)
    size = 300
    # transform = data_transforms.TestLandmsTransform(size, mean=0.0, std=1.0)
    # transform = data_transforms.TestTransform(size, mean=0.0, std=1.0)
    transform = None
    voc = VOCKeyPointDataset(filename=None,
                             data_root=None,
                             anno_dir=anno_dir,
                             image_dir=image_dir,
                             class_names=class_names,
                             transform=transform,
                             check=False)
    voc = ConcatDataset([voc, voc])
    # voc = torch_utils.ConcatDataset([voc, voc])
    print("have num:{}".format(len(voc)))
    for i in range(len(voc)):
        # image, bboxes, labels, keypoints = voc.__getitem__(i)
        # image, bboxes, labels, angles = voc.__getitem__(i)
        image, bboxes_labels = voc.__getitem__(i)
        bboxes = bboxes_labels[:, 0:4]
        labels = bboxes_labels[:, 4:5]
        angles = bboxes_labels[:, 5:6]
        height, width, depth = image.shape
        boxes_scale = [width, height] * 2
        if isshow:
            # draw_pair_points_bbox_angle(image, bboxes, labels, angels=None, keypoints_pair=keypoints)
            draw_pair_points_bbox_angle(image, bboxes, labels, angels=angles, keypoints_pair=None)
