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


class Dataset(object):
    """
    from torch.utils.data import DataLoader, ConcatDataset
    """

    def __init__(self, **kwargs):
        self.image_id = []

    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def read_files(filename, *args):
        """
        :param filename:
        :return:
        """
        image_id = []
        with open(filename, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip().split(" ")[0]
                image_id.append(line.rstrip())
        return image_id


class VOCDataset(Dataset):

    def __init__(self,
                 filename=None,
                 data_root=None,
                 anno_dir=None,
                 image_dir=None,
                 class_names=None,
                 transform=None,
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
        super(VOCDataset, self).__init__()
        self.class_names, self.class_dict = self.parser_classes(class_names)
        parser = self.parser_paths(filename, data_root, anno_dir, image_dir)
        self.data_root, self.anno_dir, self.image_dir, self.image_id = parser
        self.postfix = self.get_image_postfix(self.image_dir, self.image_id)
        self.transform = transform
        self.color_space = color_space
        self.keep_difficult = keep_difficult
        if check:
            self.image_id = self.checking(self.image_id)
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_id)
        self.num_images = len(self.image_id)
        self.classes = list(self.class_dict.values())
        print("class_dict:{}".format(self.class_dict))
        print("image id:{}".format(len(self.image_id)))

    def get_image_postfix(self, image_dir, image_id):
        """
        获得图像文件后缀名
        :param image_dir:
        :return:
        """
        if "." in image_id[0]:
            postfix = ""
        else:
            image_list = glob.glob(os.path.join(image_dir, "*"))
            postfix = os.path.basename(image_list[0]).split(".")[1]
        return postfix

    def __get_image_anno_file(self, image_dir, anno_dir, image_id: str, img_postfix):
        """
        :param image_dir:
        :param anno_dir:
        :param image_id:
        :param img_postfix:
        :return:
        """
        if not img_postfix and "." in image_id:
            image_id, img_postfix = image_id.split(".")
        image_file = os.path.join(image_dir, "{}.{}".format(image_id, img_postfix))
        annotation_file = os.path.join(anno_dir, "{}.xml".format(image_id))
        return image_file, annotation_file

    def checking(self, image_ids: list):
        """
        '/home/dm/panjinquan3/dataset/Character/gimage_v1/JPEGImages/image_000123.jpg'
        :param image_ids:
        :return:
        """
        dst_ids = []
        # image_id = image_id[:100]
        image_ids = image_ids[100:]
        for image_id in tqdm(image_ids):
            image_file, annotation_file = self.__get_image_anno_file(self.image_dir, self.anno_dir, image_id,
                                                                     self.postfix)
            if not os.path.exists(annotation_file):
                continue
            if not os.path.exists(image_file):
                continue
            bboxes, labels, is_difficult = self.get_annotation(annotation_file)
            if not self.keep_difficult:
                bboxes = bboxes[is_difficult == 0]
                # labels = labels[is_difficult == 0]
            if len(bboxes) == 0 or len(labels) == 0:
                continue
            dst_ids.append(image_id)
        return dst_ids

    def parser_classes(self, class_names):
        """
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        :param class_names:
                    str : class file
                    list: ["face","person"]
                    dict: 可以自定义label的id{'BACKGROUND': 0, 'person': 1, 'person_up': 1, 'person_down': 1}
        :return:
        """
        if isinstance(class_names, str):
            class_names = super().read_files(class_names)
        if isinstance(class_names, list):
            class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        elif isinstance(class_names, dict):
            class_dict = class_names
        else:
            class_dict = None
        return class_names, class_dict

    def parser_paths(self, filenames=None, data_root=None, anno_dir=None, image_dir=None):
        """
        :param filenames:
        :param data_root:
        :param anno_dir:
        :param image_dir:
        :return:
        """
        if isinstance(data_root, str):
            anno_dir = os.path.join(data_root, "Annotations") if not anno_dir else anno_dir
            image_dir = os.path.join(data_root, "JPEGImages") if not image_dir else image_dir
        if isinstance(filenames, str):
            data_root = os.path.dirname(filenames)
        image_id = self.read_files(filenames, anno_dir)
        if not anno_dir:
            anno_dir = os.path.join(data_root, "Annotations")
        if not image_dir:
            image_dir = os.path.join(data_root, "JPEGImages")
        return data_root, anno_dir, image_dir, image_id

    def crop_image(self, image, bbox):
        """
        :param image:
        :param bbox:
        :return:
        """
        # bboxes = image_processing.extend_bboxes([bbox], scale=[1.5, 1.5])
        # bboxes = image_processing.extend_bboxes([bbox], scale=[1.2, 1.2])
        bboxes = image_processing.extend_bboxes([bbox], scale=[1.3, 1.3])
        images = image_processing.get_bboxes_crop_padding(image, bboxes)
        return images, bboxes

    def __getitem__(self, index):
        """
        :param index: int or str
        :return:rgb_image
        """
        image_id = self.index2id(index)
        # print(image_id)
        # image_id = "DN0000063_00.png"
        image_file, annotation_file = self.__get_image_anno_file(self.image_dir, self.anno_dir, image_id, self.postfix)
        bboxes, labels, is_difficult = self.get_annotation(annotation_file)
        rgb_image = self.read_image(image_file, color_space=self.color_space)
        if self.transform:
            rgb_image, bboxes, labels = self.transform(rgb_image, bboxes, labels)
        return rgb_image, bboxes, labels

    def get_image_anno_file(self, index):
        """
        :param index:
        :return:
        """
        image_id = self.index2id(index)
        image_file, annotation_file = self.__get_image_anno_file(self.image_dir, self.anno_dir, image_id, self.postfix)
        return image_file, annotation_file

    def index2id(self, index):
        """
        :param index: int or str
        :return:
        """
        if isinstance(index, str):
            image_id = index
        else:
            image_id = self.image_id[index]
        return image_id

    def __len__(self):
        return len(self.image_id)

    def get_annotation(self, xml_file):
        """
        :param xml_file:
        :param class_dict: class_dict = {class_name: i for i, class_name in enumerate(class_names)}
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
            item = {}
            item["bbox"] = bbox
            item["difficult"] = difficult
            if self.class_dict:
                name = self.class_dict[name]
            item["name"] = name
            objects_list.append(item)
        bboxes, labels, is_difficult = self.get_objects_items(objects_list)
        return bboxes, labels, is_difficult

    def get_objects_items(self, objects_list):
        """
        :param objects_list:
        :return:
        """
        bboxes = []
        labels = []
        is_difficult = []
        for item in objects_list:
            bboxes.append(item["bbox"])
            labels.append(item['name'])
            is_difficult.append(item['difficult'])
        bboxes = np.array(bboxes, dtype=np.float32)
        # labels = np.array(labels, dtype=np.int64)
        labels = np.asarray(labels).reshape(-1, 1)
        is_difficult = np.array(is_difficult, dtype=np.uint8)
        return bboxes, labels, is_difficult

    @staticmethod
    def read_files(filename, *args):
        """
        :param filename:
        :return:
        """
        if not filename:  # if None
            assert args
            anno_list = []
            for a in args:
                anno_list += file_processing.get_files_list(a, postfix=["*.xml"])
            image_id = VOCDataset.get_files_id(anno_list)
        elif isinstance(filename, list):
            image_id = filename
        elif isinstance(filename, str):
            # image_id = super().read_files(filename)
            image_id = Dataset.read_files(filename)
        else:
            image_id = None
            assert Exception("Error:{}".format(filename))
        return image_id

    @staticmethod
    def get_files_id(file_list):
        """
        :param file_list:
        :return:
        """
        image_idx = []
        for path in file_list:
            basename = os.path.basename(path)
            id = basename.split(".")[0]
            image_idx.append(id)
        return image_idx

    @staticmethod
    def read_xml2json(xml_file):
        """
        import xmltodict
        :param xml_file:
        :return:
        """
        with open(xml_file, encoding='utf-8') as fd:  # 将XML文件装载到dict里面
            content = xmltodict.parse(fd.read())
        return content

    def read_image(self, image_file, color_space="RGB"):
        """
        :param image_file:
        :param color_space:
        :return:
        """
        image = cv2.imread(str(image_file))
        if color_space.lower() == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class ConcatDataset(Dataset):
    """ Concat Dataset """

    def __init__(self, datasets, shuffle=False):
        """
        import torch.utils.data as torch_utils
        voc1 = PolygonParser(filename1)
        voc2 = PolygonParser(filename2)
        voc=torch_utils.ConcatDataset([voc1, voc2])
        ====================================
        :param datasets:
        :param shuffle:
        """
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'dataset should not be an empty iterable'
        # super(ConcatDataset, self).__init__()
        if not isinstance(datasets, list):
            datasets = [datasets]
        self.image_id = []
        self.dataset = datasets
        self.shuffle = shuffle
        for dataset_id, dataset in enumerate(self.dataset):
            image_id = dataset.image_id
            image_id = self.add_dataset_id(image_id, dataset_id)
            self.image_id += image_id
            self.classes = dataset.classes
        if shuffle:
            random.seed(200)
            random.shuffle(self.image_id)

    def add_dataset_id(self, image_id, dataset_id):
        """
        :param image_id:
        :param dataset_id:
        :return:
        """
        out_image_id = []
        for id in image_id:
            out_image_id.append({"dataset_id": dataset_id, "image_id": id})
        return out_image_id

    def __getitem__(self, index):
        """
        :param index: int
        :return:
        """
        dataset_id = self.image_id[index]["dataset_id"]
        image_id = self.image_id[index]["image_id"]
        dataset = self.dataset[dataset_id]
        # print(dataset.data_root, image_id)
        data = dataset.__getitem__(image_id)
        return data

    def __len__(self):
        return len(self.image_id)


def VOCDatasets(filenames=None,
                data_root=None,
                anno_dir=None,
                image_dir=None,
                class_names=None,
                transform=None,
                color_space="RGB",
                keep_difficult=False,
                shuffle=False,
                check=False):
    """
    :param filenames:
    :param data_root:
    :param anno_dir:
    :param image_dir:
    :param class_names:
    :param transform:
    :param color_space:
    :param keep_difficult:
    :param shuffle:
    :param check:
    :return:
    """
    if not isinstance(filenames, list) and os.path.isfile(filenames):
        filenames = [filenames]
    datas = []
    for filename in filenames:
        data = VOCDataset(filename=filename,
                          data_root=data_root,
                          anno_dir=anno_dir,
                          image_dir=image_dir,
                          class_names=class_names,
                          transform=transform,
                          color_space=color_space,
                          keep_difficult=keep_difficult,
                          shuffle=shuffle,
                          check=check)
        datas.append(data)
    voc = ConcatDataset(datas, shuffle=shuffle)
    return voc


if __name__ == "__main__":
    from utils import image_processing, file_processing
    from modules.image_transforms import data_transforms

    isshow = True
    # data_root = "/home/dm/panjinquan3/dataset/MPII/"
    # data_root = "/media/dm/dm2/git/python-learning-notes/dataset/Test_Voc"
    # anno_dir = '/home/dm/panjinquan3/dataset/finger/finger/Annotations'
    # image_dir = '/home/dm/panjinquan3/dataset/finger/finger/JPEGImages'
    # data_root = "/home/dm/panjinquan3/dataset/Character/gimage_v1/"
    # data_root = "/home/dm/panjinquan3/dataset/finger/finger_v5/"
    # data_root = '/home/dm/panjinquan3/dataset/Character/gimage_v1/'
    data_root = "/home/dm/panjinquan3/dataset/hook_circle/SRC/v2/crop/"
    # data_root = "/home/dm/panjinquan3/dataset/Character/gimage_v1"
    # data_root = "/home/dm/panjinquan3/dataset/finger/finger_v3/"
    # data_root = "/home/dm/panjinquan3/dataset/finger/finger_test/"
    # data_root = "/home/dm/panjinquan3/dataset/OpenImages/"
    image_dir = data_root + "JPEGImages"
    anno_dir = data_root + "Annotations"
    filenames = data_root + "trainval.txt"
    # class_names = ["face", "person"]
    class_names = ["face"]
    # anno_dir = data_root + '/Annotations'
    shuffle = False
    # class_names = ["face"]
    # class_names = None
    class_names = {"circle": 0, "hook": 1, "slash": 2, "underline": 3}
    # anno_list = file_processing.get_files_list(anno_dir, postfix=["*.xml"])
    # image_id_list = file_processing.get_files_id(anno_list)
    size = [300,100]
    # transform = data_transforms.TrainAugmentation(size, mean=0.0, std=1.0)
    transform = data_transforms.TestTransform(size, mean=0.0, std=1.0)
    voc = VOCDataset(filename=filenames,
                     data_root=None,
                     anno_dir=anno_dir,
                     image_dir=image_dir,
                     class_names=class_names,
                     transform=transform,
                     check=True)
    voc = ConcatDataset([voc, voc])
    # voc = torch_utils.ConcatDataset([voc, voc])
    print("have num:{}".format(len(voc)))
    for i in range(len(voc)):
        image, bboxes, labels = voc.__getitem__(i)
        height, width, depth = image.shape
        boxes_scale = [width, height] * 2
        boxes = bboxes * boxes_scale
        print(i, boxes)
        if isshow:
            # image = np.asarray(image, dtype=np.uint8)
            image = np.asarray(image * 255.0, dtype=np.uint8)
            boxes_name = labels.reshape(-1)
            image = image_processing.draw_image_bboxes_text(image, boxes, boxes_name=boxes_name)
            # dst_image=image_processing.get_bbox_crop_padding(dst_image,bbox=[-10,100,100,500])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", image)
            cv2.waitKey(0)
