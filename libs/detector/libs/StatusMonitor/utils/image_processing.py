# -*-coding: utf-8 -*-
"""
    @Project: IntelligentManufacture
    @File   : image_processing.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-02-14 15:34:50
"""

import os
import copy
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import matplotlib
import base64
import PIL.Image as Image


def get_color_map(nums=25):
    colors = [
        "#FF0000", "#FF7F50", "#B0171F", "#872657", "#FF00FF",
        "#FFFF00", "#FF8000", "#FF9912", "#DAA569", "#FF6100",
        "#0000FF", "#3D59AB", "#03A89E", "#33A1C9", "#00C78C",
        "#00FF00", "#385E0F", "#00C957", "#6B8E23", "#2E8B57",
        "#A020F0", "#8A2BE2", "#A066D3", "#DA70D6", "#DDA0DD"]
    colors = colors * int(np.ceil(nums / len(colors)))
    return colors


def get_color(id):
    color = convert_color_map(COLOR_MAP[id])
    return color


def set_class_set(class_set=set()):
    global CLASS_SET
    CLASS_SET = class_set


COLOR_MAP = get_color_map(200)
CLASS_SET = set()

cmap = plt.get_cmap('rainbow')


def get_colors(nums):
    colors = [cmap(i) for i in np.linspace(0, 1, nums + 2)]
    colors = [(c[2], c[1], c[0]) for c in colors]
    return colors


def convert_color_map(color, colorType="BGR"):
    '''
    :param color:
    :param colorType:
    :return:
    '''
    assert (len(color) == 7 and color[0] == "#"), "input color error:color={}".format(color)
    R = color[1:3]
    G = color[3:5]
    B = color[5:7]

    R = int(R, 16)
    G = int(G, 16)
    B = int(B, 16)
    if colorType == "BGR":
        return (B, G, R)
    elif colorType == "RGB":
        return (R, G, B)
    else:
        assert "colorType error "


def bound_protection(points, height, width):
    """
    Avoid array overbounds
    :param points:
    :param height:
    :param width:
    :return:
    """
    points[points[:, 0] > width, 0] = width - 1  # x
    points[points[:, 1] > height, 1] = height - 1  # y

    # points[points[:, 0] < 0, 0] = 0  # x
    # points[points[:, 1] < 0, 1] = 0  # y
    return points


def tensor2image(batch_tensor, index=0):
    """
    convert tensor to image
    :param batch_tensor:
    :param index:
    :return:
    """
    image_tensor = batch_tensor[index, :]
    image = np.array(image_tensor, dtype=np.float32)
    image = np.squeeze(image)
    image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
    return image


def get_image_tensor(image_path, image_size, transpose=False):
    image = read_image(image_path)
    # transform = default_transform(image_size)
    # torch_image = transform(image).detach().numpy()
    image = resize_image(image, int(128 * image_size[0] / 112), int(128 * image_size[1] / 112))
    image = center_crop(image, crop_size=image_size)
    image_tensor = image_normalization(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if transpose:
        image_tensor = image_tensor.transpose(2, 0, 1)  # NHWC->NCHW
    image_tensor = image_tensor[np.newaxis, :]
    # std = np.std(torch_image-image_tensor)
    return image_tensor


def image_clip(image):
    """
    :param image:
    :return:
    """
    image = np.clip(image, 0, 1)
    return image


def transpose(data):
    data = data.transpose(2, 0, 1)  # HWC->CHW
    return data


def untranspose(data):
    if len(data.shape) == 3:
        data = data.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
    else:
        data = data.transpose(1, 0)
    return data


def show_batch_image(title, batch_imgs, index=0):
    '''
    批量显示图片
    :param title:
    :param batch_imgs:
    :param index:
    :return:
    '''
    image = batch_imgs[index, :]
    # image = image.numpy()  #
    image = np.array(image, dtype=np.float32)
    image = np.squeeze(image)
    image = untranspose(image)
    if title:
        cv_show_image(title, image)


def show_image(title, rgb_image):
    '''
    调用matplotlib显示RGB图片
    :param title: 图像标题
    :param rgb_image: 图像的数据
    :return:
    '''
    # plt.figure("show_image")
    # print(image.dtype)
    channel = len(rgb_image.shape)
    if channel == 3:
        plt.imshow(rgb_image)
    else:
        plt.imshow(rgb_image, cmap='gray')
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title(title)  # 图像题目
    plt.show()


def cv_show_image(title, image, type='rgb', waitKey=0):
    '''
    调用OpenCV显示RGB图片
    :param title: 图像标题
    :param image: 输入RGB图像
    :param type:'rgb' or 'bgr'
    :return:
    '''
    img = copy.copy(image)
    channels = img.shape[-1]
    if channels == 3 and type == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 将BGR转为RGB
    if title:
        cv2.imshow(title, img)
        cv2.waitKey(waitKey)


def image_fliplr(image):
    '''
    左右翻转
    :param image:
    :return:
    '''
    image = np.fliplr(image)
    return image


def get_prewhiten_image(x):
    '''
    图片白化处理
    :param x:
    :return:
    '''
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def image_normalization(image, mean=None, std=None):
    '''
    正则化，归一化
    image[channel] = (image[channel] - mean[channel]) / std[channel]
    :param image: numpy image
    :param mean: [0.5,0.5,0.5]
    :param std:  [0.5,0.5,0.5]
    :return:
    '''
    # 不能写成:image=image/255
    if isinstance(mean, list):
        mean = np.asarray(mean, dtype=np.float32)
    if isinstance(std, list):
        std = np.asarray(std, dtype=np.float32)
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    if mean is not None:
        image = np.subtract(image, mean)
    if std is not None:
        image = np.multiply(image, 1 / std)
    return image


def data_normalization(data, ymin, ymax):
    '''
    NORMALIZATION 将数据x归一化到任意区间[ymin,ymax]范围的方法
    :param data:  输入参数x：需要被归一化的数据,numpy
    :param ymin: 输入参数ymin：归一化的区间[ymin,ymax]下限
    :param ymax: 输入参数ymax：归一化的区间[ymin,ymax]上限
    :return: 输出参数y：归一化到区间[ymin,ymax]的数据
    '''
    xmax = np.max(data)  # %计算最大值
    xmin = np.min(data)  # %计算最小值
    y = (ymax - ymin) * (data - xmin) / (xmax - xmin) + ymin
    return y


def cv_image_normalization(image, min_val=0.0, max_val=1.0):
    '''

    :param image:
    :param min_val:
    :param max_val:
    :param norm_type:
    :param dtype:
    :param mask:
    :return:
    '''
    dtype = cv2.CV_32F
    norm_type = cv2.NORM_MINMAX
    out = np.zeros(shape=image.shape, dtype=np.float32)
    cv2.normalize(image, out, alpha=min_val, beta=max_val, norm_type=norm_type, dtype=dtype)
    return out


def get_prewhiten_images(images_list, normalization=False):
    '''
    批量白化图片处理
    :param images_list:
    :param normalization:
    :return:
    '''
    out_images = []
    for image in images_list:
        if normalization:
            image = image_normalization(image)
        image = get_prewhiten_image(image)
        out_images.append(image)
    return out_images


def read_image(filename, resize_height=None, resize_width=None, normalization=False, colorSpace='RGB'):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :param colorSpace 输出格式：RGB or BGR
    :return: 返回的图片数据
    '''

    bgr_image = cv2.imread(filename)
    # bgr_image = cv2.imread(filename,cv2.IMREAD_IGNORE_ORIENTATION|cv2.IMREAD_UNCHANGED)
    if bgr_image is None:
        print("Warning: no image:{}".format(filename))
        return None
    if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    if colorSpace == 'RGB':
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    elif colorSpace == "BGR":
        image = bgr_image
    else:
        exit(0)
    # show_image(filename,image)
    # image=Image.open(filename)
    image = resize_image(image, resize_height, resize_width)
    image = np.asanyarray(image)
    if normalization:
        image = image_normalization(image)
    # show_image("src resize image",image)
    return image


def read_image_pil(filename, resize_height=None, resize_width=None, normalization=False):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :param colorSpace 输出格式：RGB or BGR
    :return: 返回的图片数据
    '''

    rgb_image = Image.open(filename)
    rgb_image = np.asarray(rgb_image)
    if rgb_image is None:
        print("Warning: no image:{}".format(filename))
        return None
    if len(rgb_image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)

    # show_image(filename,image)
    # image=Image.open(filename)
    image = resize_image(rgb_image, resize_height, resize_width)
    if normalization:
        image = image_normalization(image)
    # show_image("src resize image",image)
    return image


def read_image_gbk(filename, resize_height=None, resize_width=None, normalization=False, colorSpace='RGB'):
    '''
    解决imread不能读取中文路径的问题,读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :param colorSpace 输出格式：RGB or BGR
    :return: 返回的RGB图片数据
    '''
    try:
        with open(filename, 'rb') as f:
            data = f.read()
            data = np.asarray(bytearray(data), dtype="uint8")
            bgr_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        bgr_image = None
    # 或者：
    # bgr_image=cv2.imdecode(np.fromfile(filename,dtype=np.uint8),cv2.IMREAD_COLOR)
    if bgr_image is None:
        print("Warning: no image:{}".format(filename))
        return None
    if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    if colorSpace == 'RGB':
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    elif colorSpace == "BGR":
        image = bgr_image
    else:
        exit(0)
    # show_image(filename,image)
    # image=Image.open(filename)
    image = resize_image(image, resize_height, resize_width)
    image = np.asanyarray(image)
    if normalization:
        image = image_normalization(image)
    # show_image("src resize image",image)
    return image


def requests_url(url):
    '''
    读取网络数据流
    :param url:
    :return:
    '''
    stream = None
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            stream = res.content
    except Exception as e:
        print(e)
    return stream


def read_images_url(url, resize_height=None, resize_width=None, normalization=False, colorSpace='RGB'):
    '''
    根据url或者图片路径，读取图片
    :param url:
    :param resize_height:
    :param resize_width:
    :param normalization:
    :param colorSpace:
    :return:
    '''
    if re.match(r'^https?:/{2}\w.+$', url):
        stream = requests_url(url)
        if stream is None:
            bgr_image = None
        else:
            content = np.asarray(bytearray(stream), dtype="uint8")
            bgr_image = cv2.imdecode(content, cv2.IMREAD_COLOR)
            # pil_image = PIL.Image.open(BytesIO(stream))
            # rgb_image=np.asarray(pil_image)
            # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    else:
        bgr_image = cv2.imread(url)

    if bgr_image is None:
        print("Warning: no image:{}".format(url))
        return None
    if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", url)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    if colorSpace == 'RGB':
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    elif colorSpace == "BGR":
        image = bgr_image
    else:
        pass
    image = resize_image(image, resize_height, resize_width)
    image = np.asanyarray(image)
    if normalization:
        image = image_normalization(image)
    # show_image("src resize image",image)
    return image


def read_image_batch(image_list):
    '''
    批量读取图片
    :param image_list:
    :return:
    '''
    image_batch = []
    out_image_list = []
    for image_path in image_list:
        image = read_images_url(image_path)
        if image is None:
            print("no image:{}".format(image_path))
            continue
        image_batch.append(image)
        out_image_list.append(image_path)
    return image_batch, out_image_list


def fast_read_image_roi(filename, orig_rect, ImreadModes=cv2.IMREAD_COLOR, normalization=False, colorSpace='RGB'):
    '''
    快速读取图片的方法
    :param filename: 图片路径
    :param orig_rect:原始图片的感兴趣区域rect
    :param ImreadModes: IMREAD_UNCHANGED
                        IMREAD_GRAYSCALE
                        IMREAD_COLOR
                        IMREAD_ANYDEPTH
                        IMREAD_ANYCOLOR
                        IMREAD_LOAD_GDAL
                        IMREAD_REDUCED_GRAYSCALE_2
                        IMREAD_REDUCED_COLOR_2
                        IMREAD_REDUCED_GRAYSCALE_4
                        IMREAD_REDUCED_COLOR_4
                        IMREAD_REDUCED_GRAYSCALE_8
                        IMREAD_REDUCED_COLOR_8
                        IMREAD_IGNORE_ORIENTATION
    :param normalization: 是否归一化
    :param colorSpace 输出格式：RGB or BGR
    :return: 返回感兴趣区域ROI
    '''
    # 当采用IMREAD_REDUCED模式时，对应rect也需要缩放
    scale = 1
    if ImreadModes == cv2.IMREAD_REDUCED_COLOR_2 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_2:
        scale = 1 / 2
    elif ImreadModes == cv2.IMREAD_REDUCED_GRAYSCALE_4 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_4:
        scale = 1 / 4
    elif ImreadModes == cv2.IMREAD_REDUCED_GRAYSCALE_8 or ImreadModes == cv2.IMREAD_REDUCED_COLOR_8:
        scale = 1 / 8
    rect = np.array(orig_rect) * scale
    rect = rect.astype(int).tolist()
    bgr_image = cv2.imread(filename, flags=ImreadModes)

    if bgr_image is None:
        print("Warning: no image:{}".format(filename))
        return None
    if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    if colorSpace == 'RGB':
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    elif colorSpace == "BGR":
        image = bgr_image
    image = np.asanyarray(image)
    if normalization:
        image = image_normalization(image)
    roi_image = get_rect_image(image, rect)
    return roi_image


def resize_image(image, resize_height=None, resize_width=None):
    '''
    tf.image.resize_images(images,size),images=[batch, height, width, channels],size=(new_height, new_width)
    cv2.resize(image, dsize=(resize_width, resize_height)),与image.shape相反
    images[50,10]与image.shape的原理相同，它表示的是image=(y=50,x=10)
    :param image:
    :param resize_height:
    :param resize_width:
    :return:
    '''
    image_shape = np.shape(image)
    height = image_shape[0]
    width = image_shape[1]
    if (resize_height is None) and (resize_width is None):  # 错误写法：resize_height and resize_width is None
        return image
    if resize_height is None:
        resize_height = int(height * resize_width / width)
    elif resize_width is None:
        resize_width = int(width * resize_height / height)
    image = cv2.resize(image, dsize=(resize_width, resize_height))
    return image


def scale_image(image, scale):
    '''
    :param image:
    :param scale: (scale_w,scale_h)
    :return:
    '''
    image = cv2.resize(image, dsize=None, fx=scale[0], fy=scale[1])
    return image


def get_rect_image(image, rect):
    '''
    :param image:
    :param rect: [x,y,w,h]
    :return:
    '''
    shape = image.shape  # h,w
    height = shape[0]
    width = shape[1]
    image_rect = (0, 0, width, height)
    rect = get_rect_intersection(rect, image_rect)
    rect = [int(i) for i in rect]
    x, y, w, h = rect
    cut_img = image[y:(y + h), x:(x + w)]
    return cut_img


def get_rects_image(image, rects_list, resize_height=None, resize_width=None):
    '''
    获得裁剪区域
    :param image:
    :param rects_list:
    :param resize_height:
    :param resize_width:
    :return:
    '''
    rect_images = []
    for rect in rects_list:
        roi = get_rect_image(image, rect)
        roi = resize_image(roi, resize_height, resize_width)
        rect_images.append(roi)
    return rect_images


def get_bboxes_image(image, bboxes_list, resize_height=None, resize_width=None):
    '''
    获得裁剪区域
    :param image:
    :param bboxes_list:
    :param resize_height:
    :param resize_width:
    :return:
    '''
    rects_list = bboxes2rects(bboxes_list)
    rect_images = get_rects_image(image, rects_list, resize_height, resize_width)
    return rect_images


def bboxes2rects(bboxes_list):
    '''
    将bboxes=[x1,y1,x2,y2] 转为rect=[x1,y1,w,h]
    :param bboxes_list:
    :return:
    '''
    rects_list = []
    for bbox in bboxes_list:
        x1, y1, x2, y2 = bbox
        rect = [x1, y1, (x2 - x1), (y2 - y1)]
        rects_list.append(rect)
    return rects_list


def rects2bboxes(rects_list):
    '''
    将rect=[x1,y1,w,h]转为bboxes=[x1,y1,x2,y2]
    :param rects_list:
    :return:
    '''
    bboxes_list = []
    for rect in rects_list:
        x1, y1, w, h = rect
        x2 = x1 + w
        y2 = y1 + h
        b = (x1, y1, x2, y2)
        bboxes_list.append(b)
    return bboxes_list


def bboxes2center(bboxes_list):
    '''
    将bboxes=[x1,y1,x2,y2] 转为center_list=[cx,cy,w,h]
    :param bboxes_list:
    :return:
    '''
    center_list = []
    for bbox in bboxes_list:
        x1, y1, x2, y2 = bbox
        center = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
        center_list.append(center)
    return center_list


def center2bboxes(center_list):
    '''
    将center_list=[cx,cy,w,h] 转为bboxes=[x1,y1,x2,y2]
    :param bboxes_list:
    :return:
    '''
    bboxes_list = []
    for c in center_list:
        cx, cy, w, h = c
        bboxes = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        bboxes_list.append(bboxes)
    return bboxes_list


def center2rects(center_list):
    '''
    将center_list=[cx,cy,w,h] 转为rect=[x,y,w,h]
    :param bboxes_list:
    :return:
    '''
    rect_list = []
    for c in center_list:
        cx, cy, w, h = c
        rect = [cx - w / 2, cy - h / 2, w, h]
        rect_list.append(rect)
    return rect_list


def scale_rect(orig_rect, orig_shape, dest_shape):
    '''
    对图像进行缩放时，对应的rectangle也要进行缩放
    :param orig_rect: 原始图像的rect=[x,y,w,h]
    :param orig_shape: 原始图像的维度shape=[h,w]
    :param dest_shape: 缩放后图像的维度shape=[h,w]
    :return: 经过缩放后的rectangle
    '''
    new_x = int(orig_rect[0] * dest_shape[1] / orig_shape[1])
    new_y = int(orig_rect[1] * dest_shape[0] / orig_shape[0])
    new_w = int(orig_rect[2] * dest_shape[1] / orig_shape[1])
    new_h = int(orig_rect[3] * dest_shape[0] / orig_shape[0])
    dest_rect = [new_x, new_y, new_w, new_h]
    return dest_rect


def get_rect_intersection(rec1, rec2):
    '''
    计算两个rect的交集坐标
    :param rec1:
    :param rec2:
    :return:
    '''
    xmin1, ymin1, xmax1, ymax1 = rects2bboxes([rec1])[0]
    xmin2, ymin2, xmax2, ymax2 = rects2bboxes([rec2])[0]
    x1 = max(xmin1, xmin2)
    y1 = max(ymin1, ymin2)
    x2 = min(xmax1, xmax2)
    y2 = min(ymax1, ymax2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return (x1, y1, w, h)


def get_bbox_intersection(box1, box2):
    '''
    计算两个boxes的交集坐标
    :param rec1:
    :param rec2:
    :return:
    '''
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    x1 = max(xmin1, xmin2)
    y1 = max(ymin1, ymin2)
    x2 = min(xmax1, xmax2)
    y2 = min(ymax1, ymax2)
    return (x1, y1, x2, y2)


def draw_image_rects(bgr_image, rect_list, color=(0, 0, 255)):
    thickness = 2
    for rect in rect_list:
        x, y, w, h = rect
        point1 = (int(x), int(y))
        point2 = (int(x + w), int(y + h))
        cv2.rectangle(bgr_image, point1, point2, color, thickness=thickness)
    return bgr_image


def draw_image_boxes(bgr_image, boxes_list, color=(0, 0, 255)):
    thickness = 2
    for box in boxes_list:
        x1, y1, x2, y2 = box
        point1 = (int(x1), int(y1))
        point2 = (int(x2), int(y2))
        cv2.rectangle(bgr_image, point1, point2, color, thickness=thickness)
    return bgr_image


def show_image_rects(win_name, image, rect_list, type="rgb", color=(0, 0, 255), waitKey=0):
    '''
    :param win_name:
    :param image:
    :param rect_list:[[ x, y, w, h],[ x, y, w, h]]
    :return:
    '''
    image = draw_image_rects(image, rect_list, color)
    cv_show_image(win_name, image, type, waitKey=waitKey)
    return image


def show_image_boxes(win_name, image, boxes_list, color=(0, 0, 255), waitKey=0):
    '''
    :param win_name:
    :param image:
    :param boxes_list:[[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
    :return:
    '''
    image = draw_image_boxes(image, boxes_list, color)
    cv_show_image(win_name, image, waitKey=waitKey)
    return image


def draw_image_bboxes_text(rgb_img, boxes, boxes_name, color=None, drawType="custom", top=True):
    """
    :param boxes_name:
    :param bgr_image: bgr image
    :param color: BGR color:[B,G,R]
    :param boxes: [[x1,y1,x2,y2],[x1,y1,x2,y2]]
    :return:
    """
    rgb_image = rgb_img.copy()
    # color_map=list(matplotlib.colors.cnames.values())
    # color_map=list(reversed(color_map))
    if not color:
        class_set = list(CLASS_SET) if CLASS_SET else list(set(boxes_name))
    set_color = color
    for name, box in zip(boxes_name, boxes):
        if not color:
            cls_id = class_set.index(name)
            set_color = get_color(cls_id)
        box = [int(b) for b in box]
        # cv2.rectangle(bgr_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2, 8, 0)
        # cv2.putText(bgr_image, name, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
        # cv2.rectangle(bgr_image, (box[0], box[1]), (box[2], box[3]), color, 2, 8, 0)
        # cv2.putText(bgr_image, str(name), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness=2)
        custom_bbox_line(rgb_image, box, set_color, name, drawType, top)
    # cv2.imshow(title, bgr_image)
    # cv2.waitKey(0)
    return rgb_image


def show_image_bboxes_text(title, rgb_image, boxes, boxes_name, color=None, drawType="custom", waitKey=0, top=True):
    '''
    :param boxes_name:
    :param bgr_image: bgr image
    :param color: BGR color:[B,G,R]
    :param boxes: [[x1,y1,x2,y2],[x1,y1,x2,y2]]
    :return:
    '''
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    bgr_image = draw_image_bboxes_text(bgr_image, boxes, boxes_name, color, drawType, top)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    cv_show_image(title, rgb_image, waitKey=waitKey)
    return rgb_image


def draw_image_rects_text(rgb_image, rects, rects_name, color=None, drawType="custom", top=True):
    boxes = rects2bboxes(rects)
    rgb_image = draw_image_bboxes_text(rgb_image, boxes, rects_name, color, drawType, top)
    return rgb_image


def show_image_rects_text(title, rgb_image, rects, rects_name, color=None, drawType="custom", waitKey=0, top=True):
    '''
    :param rects_name:
    :param bgr_image: bgr image
    :param rects: [[x1,y1,w,h],[x1,y1,w,h]]
    :return:
    '''
    boxes = rects2bboxes(rects)
    rgb_image = show_image_bboxes_text(title, rgb_image, boxes, rects_name, color, drawType, waitKey, top)
    return rgb_image


def draw_image_detection_rects(rgb_image, rects, probs, lables, color=None):
    bboxes = rects2bboxes(rects)
    rgb_image = draw_image_detection_bboxes(rgb_image, bboxes, probs, lables, color)
    return rgb_image


def show_image_detection_rects(title, rgb_image, rects, probs, lables, color=None, waitKey=0):
    '''
    :param title:
    :param rgb_image:
    :param rects: [[x1,y1,w,h],[x1,y1,w,h]]
    :param probs:
    :param lables:
    :return:
    '''
    bboxes = rects2bboxes(rects)
    rgb_image = show_image_detection_bboxes(title, rgb_image, bboxes, probs, lables, color, waitKey)
    return rgb_image


def draw_image_detection_bboxes(rgb_image, bboxes, probs, labels, color=None):
    '''
    :param title:
    :param rgb_image:
    :param bboxes:  [[x1,y1,x2,y2],[x1,y1,x2,y2]]
    :param probs:
    :param labels:
    :return:
    '''
    class_set = list(CLASS_SET)
    if not class_set:
        class_set = list(set(labels))
    boxes_name = combile_label_prob(labels, probs)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    # color_map=list(matplotlib.colors.cnames.values())
    # color_map=list(reversed(color_map))
    set_color = color
    for l, name, box in zip(labels, boxes_name, bboxes):
        if not color:
            cls_id = class_set.index(l)
            set_color = get_color(cls_id)
        box = [int(b) for b in box]
        # cv2.rectangle(bgr_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2, 8, 0)
        # cv2.putText(bgr_image, name, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
        # cv2.rectangle(bgr_image, (box[0], box[1]), (box[2], box[3]), color, 2, 8, 0)
        # cv2.putText(bgr_image, str(name), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness=2)
        custom_bbox_line(bgr_image, box, set_color, name, drawType="custom")
    # cv2.imshow(title, bgr_image)
    # cv2.waitKey(0)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


def show_image_detection_bboxes(title, rgb_image, bboxes, probs, lables, color=None, waitKey=0):
    rgb_image = draw_image_detection_bboxes(rgb_image, bboxes, probs, lables, color)
    cv_show_image(title, rgb_image, waitKey=waitKey)
    return rgb_image


def custom_bbox_line(img, bbox, color, name, drawType="custom", top=True):
    """
    :param img:
    :param bbox:
    :param color:
    :param name:
    :param drawType:
    :param top:
    :return:
    """
    if drawType == "simple":
        fontScale = 0.6
        thickness = 1
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness, 8, 0)
        cv2.putText(img, str(name), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)
    elif drawType == "custom":
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # draw score roi
        # fontScale = 0.4
        fontScale = 0.6
        thickness = 1
        text_size, baseline = cv2.getTextSize(str(name), cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)
        if top:
            text_loc = (bbox[0], bbox[1] - text_size[1])
        else:
            # text_loc = (bbox[0], bbox[3])
            # text_loc = (bbox[2], bbox[3] - text_size[1])
            text_loc = (bbox[2], bbox[1] + text_size[1])

        cv2.rectangle(img, (text_loc[0] - 2 // 2, text_loc[1] - 2 - baseline),
                      (text_loc[0] + text_size[0], text_loc[1] + text_size[1]), color, -1)
        # draw score value
        cv2.putText(img, str(name), (text_loc[0], text_loc[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                    (255, 255, 255), thickness, 8)
    return img


def show_boxList(win_name, boxList, rgb_image, waitKey=0):
    '''
    [xmin,ymin,xmax,ymax]
    :param win_name:
    :param boxList:
    :param rgb_image:
    :return:
    '''
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    for item in boxList:
        name = item["label"]
        xmin = item["xtl"]
        xmax = item["xbr"]
        ymin = item["ytl"]
        ymax = item["ybr"]
        # box=[xbr,ybr,xtl,ytl]
        box = [xmin, ymin, xmax, ymax]
        box = [int(float(b)) for b in box]
        cv2.rectangle(bgr_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2, 8, 0)
        cv2.putText(bgr_image, name, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
    # cv2.imshow(title, bgr_image)
    # cv2.waitKey(0)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    if win_name:
        cv_show_image(win_name, rgb_image, waitKey=waitKey)
    return rgb_image


def draw_landmark(image, landmarks_list, point_color=(0, 0, 255), vis_id=False):
    image = copy.copy(image)
    point_size = 1
    thickness = 4  # 可以为 0 、4、8
    for landmarks in landmarks_list:
        for i, landmark in enumerate(landmarks):
            # 要画的点的坐标
            point = (int(landmark[0]), int(landmark[1]))
            cv2.circle(image, point, point_size, point_color, thickness)
            if vis_id:
                image = draw_points_text(image, [point], texts=str(i), color=point_color, drawType="simple")
    return image


def show_landmark_boxes(win_name, img, landmarks_list, boxes):
    '''
    显示landmark和boxex
    :param win_name:
    :param image:
    :param landmarks_list: [[x1, y1], [x2, y2]]
    :param boxes:     [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
    :return:
    '''
    image = draw_landmark(img, landmarks_list)
    image = show_image_boxes(win_name, image, boxes)
    return image


def show_landmark(win_name, img, landmarks_list, vis_id=False, waitKey=0):
    '''
    显示landmark和boxex
    :param win_name:
    :param image:
    :param landmarks_list: [[x1, y1], [x2, y2]]
    :return:
    '''
    image = draw_landmark(img, landmarks_list, vis_id=vis_id)
    cv_show_image(win_name, image, waitKey=waitKey)
    return image


def draw_points_text(img, points, texts=None, color=(0, 0, 255), drawType="custom"):
    '''

    :param img:
    :param points:
    :param texts:
    :param color:
    :param drawType: custom or simple
    :return:
    '''
    thickness = 5
    if texts is None:
        texts = [""] * len(points)
    for point, text in zip(points, texts):
        point = (int(point[0]), int(point[1]))
        cv2.circle(img, point, thickness, color, -1)
        draw_text(img, point, text, bg_color=color, drawType=drawType)
    return img


def draw_text(img, point, text, bg_color=(255, 0, 0), drawType="custom"):
    '''
    :param img:
    :param point:
    :param text:
    :param drawType: custom or simple
    :return:
    '''
    fontScale = 0.4
    thickness = 5
    text_thickness = 1
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    if drawType == "custom":
        text_size, baseline = cv2.getTextSize(str(text), fontFace, fontScale, thickness)
        text_loc = (point[0], point[1] + text_size[1])
        cv2.rectangle(img, (text_loc[0] - 2 // 2, text_loc[1] - 2 - baseline),
                      (text_loc[0] + text_size[0], text_loc[1] + text_size[1]), bg_color, -1)
        # draw score value
        cv2.putText(img, str(text), (text_loc[0], text_loc[1] + baseline), fontFace, fontScale,
                    (255, 255, 255), text_thickness, 8)
    elif drawType == "simple":
        cv2.putText(img, str(text), point, fontFace, 0.5, (255, 0, 0))
    return img


def draw_text_line(img, point, text_line: str, bg_color=(255, 0, 0), drawType="custom"):
    '''
    :param img:
    :param point:
    :param text:
    :param drawType: custom or custom
    :return:
    '''
    fontScale = 0.4
    thickness = 5
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    text_line = text_line.split("\n")
    # text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
    text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
    for i, text in enumerate(text_line):
        if text:
            draw_point = [point[0], point[1] + (text_size[1] + 2 + baseline) * i]
            img = draw_text(img, draw_point, text, bg_color, drawType)
    return img


def draw_key_point_in_image(image, key_points, pointline=[]):
    '''
    :param key_points: list(ndarray(19,2)) or ndarray(n_person,19,2)
    :param image:
    :param pointline: `auto`->pointline = circle_line(len(points), iscircle=True)
    :return:
    '''
    img = copy.deepcopy(image)
    person_nums = len(key_points)
    for person_id, points in enumerate(key_points):
        if points is None:
            continue
        color = get_color(person_id)
        img = draw_point_line(img, points, pointline, color, check=True)
    return img


def draw_point_line(img, points, pointline=[], color=(0, 255, 0), texts=None, drawType="simple", check=True):
    '''
    在图像中画点和连接线
    :param img:
    :param points: 点列表
    :param pointline: `auto`->pointline = circle_line(len(points), iscircle=True)
    :param color:
    :param texts:
    :param drawType: simple or custom
    :param check:
    :return:
    '''
    points = np.asarray(points, dtype=np.int32)
    image = copy.copy(img)
    line_thickness = 1
    if texts is None:
        texts = list(range(len(points)))
    image = draw_points_text(image, points, texts=texts, color=color, drawType=drawType)
    if pointline == "auto":
        pointline = circle_line(len(points), iscircle=True)
    for point_index in pointline:
        point1 = tuple(points[point_index[0]])
        point2 = tuple(points[point_index[1]])
        if check:
            if point1 is None or point2 is None:
                continue
            if sum(point1) <= 0 or sum(point2) <= 0:
                continue
        cv2.line(image, point1, point2, color, line_thickness)  # 绿色，3个像素宽度
    return image


def circle_line(num_point, iscircle=True):
    '''
    产生连接线的点,用于绘制连接线
    points_line=circle_line(len(points),iscircle=True)
    >> [(0, 1), (1, 2), (2, 0)]
    :param num_point:
    :param iscircle: 首尾是否相连
    :return:
    '''
    start = 0
    end = num_point - 1
    points_line = []
    for i in range(start, end + 1):
        if i == end and iscircle:
            points_line.append([end, start])
        elif i != end:
            points_line.append([i, i + 1])
    return points_line


def cv_paste_image(im, mask, start_point=(0, 0)):
    """
    :param im:
    :param start_point:
    :param mask:
    :return:
    """
    xim, ymin = start_point
    shape = mask.shape  # h, w, d
    im[ymin:(ymin + shape[0]), xim:(xim + shape[1])] = mask
    return im


def pil_paste_image(im, mask, start_point=(0, 0)):
    """
    :param im:
    :param mask:
    :param start_point:
    :return:
    """
    out = Image.fromarray(im)
    mask = Image.fromarray(mask)
    out.paste(mask, start_point, mask)
    return np.asarray(out)


def cv_rotate(image, angle, center=None, scale=1.0):  # 1
    (h, w) = image.shape[:2]  # 2
    if center is None:  # 3
        center = (w // 2, h // 2)  # 4
    M = cv2.getRotationMatrix2D(center, angle, scale)  # 5
    rotated = cv2.warpAffine(image, M, (w, h))  # 6
    return rotated  # 7


def rgb_to_gray(image):
    '''
    RGB to Gray image
    :param image:
    :return:
    '''
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def convert_color_space(image, colorSpace='RGB'):
    if colorSpace == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif colorSpace == 'BGR':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif colorSpace == 'GRAY':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif colorSpace == 'COLOR':
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        raise Exception("colorSpace error:{}".format(colorSpace))
    return image


def save_image(image_path, rgb_image, toUINT8=False):
    '''
    保存图片
    :param image_path:
    :param rgb_image:
    :param toUINT8:
    :return:
    '''
    save_dir = os.path.dirname(image_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if toUINT8:
        rgb_image = np.asanyarray(rgb_image * 255, dtype=np.uint8)
    if len(rgb_image.shape) == 2:  # 若是灰度图则转为三通道
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, bgr_image)


def save_image_lable_dir(save_root, image_list, image_ids, index):
    for i, (image, id) in enumerate(zip(image_list, image_ids)):
        image_path = os.path.join(save_root, str(id))
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        image_path = os.path.join(image_path, str(index) + "_" + str(i) + ".jpg")
        save_image(image_path, image, toUINT8=False)


def combime_save_image(orig_image, dest_image, out_dir, name, prefix):
    '''
    命名标准：out_dir/name_prefix.jpg
    :param orig_image:
    :param dest_image:
    :param image_path:
    :param out_dir:
    :param prefix:
    :return:
    '''
    dest_path = os.path.join(out_dir, name + "_" + prefix + ".jpg")
    save_image(dest_path, dest_image)

    dest_image = np.hstack((orig_image, dest_image))
    save_image(os.path.join(out_dir, "{}_src_{}.jpg".format(name, prefix)), dest_image)


def combile_label_prob(label_list, prob_list):
    '''
    将label_list和prob_list拼接在一起，以便显示
    :param label_list:
    :param prob_list:
    :return:
    '''
    info = [str(l) + ":" + str(p)[:5] for l, p in zip(label_list, prob_list)]
    return info


def nms_bboxes_cv2(bboxes_list, scores_list, labels_list, width=None, height=None, score_threshold=0.5,
                   nms_threshold=0.45):
    '''
    NMS
    fix a bug: cv2.dnn.NMSBoxe bboxes, scores params must be list and float data,can not be float32 or int
    :param bboxes_list: [list[xmin,ymin,xmax,ymax],[],,,]
    :param scores_list: [float,...]
    :param labels_list: [int,...]
    :param width:
    :param height:
    :param score_threshold:
    :param nms_threshold:
    :return:
    '''
    assert isinstance(scores_list, list), "scores_list must be list"
    assert isinstance(bboxes_list, list), "bboxes_list must be list"
    assert isinstance(labels_list, list), "labels_list must be list"

    dest_bboxes_list = []
    dest_scores_list = []
    dest_labels_list = []
    # bboxes_list,scores_list, labels_list=filtering_scores(bboxes_list, scores_list, labels_list, score_threshold=score_threshold)
    if width is not None and height is not None:
        for i, box in enumerate(bboxes_list):
            x1 = box[0] * width
            y1 = box[1] * height
            x2 = box[2] * width
            y2 = box[3] * height
            bboxes_list[i] = [x1, y1, x2, y2]
    scores_list = np.asarray(scores_list, dtype=np.float).tolist()
    # fix a bug: cv2.dnn.NMSBoxe bboxes, scores params must be list and float data,can not be float32 or int
    indices = cv2.dnn.NMSBoxes(bboxes_list, scores_list, score_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        dest_bboxes_list.append(bboxes_list[i])
        dest_scores_list.append(scores_list[i])
        dest_labels_list.append(labels_list[i])
    return dest_bboxes_list, dest_scores_list, dest_labels_list


def filtering_scores(bboxes_list, scores_list, labels_list, score_threshold=0.0):
    '''
    filtering low score bbox
    :param bboxes_list:
    :param scores_list:
    :param labels_list:
    :param score_threshold:
    :return:
    '''
    dest_scores_list = []
    dest_labels_list = []
    dest_bboxes_list = []
    for i, score in enumerate(scores_list):
        if score < score_threshold:
            continue
        dest_scores_list.append(scores_list[i])
        dest_labels_list.append(labels_list[i])
        dest_bboxes_list.append(bboxes_list[i])
    return dest_bboxes_list, dest_scores_list, dest_labels_list


def image_to_base64(rgb_image):
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    image = cv2.imencode('.jpg', bgr_image)[1]
    # image_base64 = str(base64.b64encode(image))[2:-1]
    image_base64 = base64.b64encode(image)
    image_base64 = str(image_base64, encoding='utf-8')
    return image_base64


def base64_to_image(image_base64):
    # base64解码
    img_data = base64.b64decode(image_base64)
    # 转换为np数组
    rgb_array = np.fromstring(img_data, np.uint8)
    # 转换成opencv可用格式
    img = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
    # img = cv2.imdecode(rgb_array, cv2.COLOR_BGR2RGB)
    return img


def read_image_base64(image_path, resize_height=None, resize_width=None):
    if resize_height is None and resize_width is None:
        with open(image_path, 'rb') as f_in:
            image_base64 = base64.b64encode(f_in.read())
            image_base64 = str(image_base64, encoding='utf-8')
    else:
        rgb_image = read_image(image_path, resize_height, resize_width)
        image_base64 = image_to_base64(rgb_image)
    return image_base64


def bin2image(bin_data, resize_height=None, resize_width=None, normalization=False, colorSpace='RGB'):
    data = np.asarray(bytearray(bin_data), dtype="uint8")
    bgr_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if colorSpace == 'RGB':
        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
    elif colorSpace == "BGR":
        image = bgr_image
    else:
        exit(0)
    # show_image(filename,image)
    # image=Image.open(filename)
    image = resize_image(image, resize_height, resize_width)
    image = np.asanyarray(image)
    if normalization:
        image = image_normalization(image)
    # show_image("src resize image",image)
    return image


def post_process(input, axis=1):
    '''
    l2_norm
    :param input:
    :param axis:
    :return:
    '''
    # norm = torch.norm(input, 2, axis, True)
    # output = torch.div(input, norm)
    output = input / np.linalg.norm(input, axis=1, keepdims=True)
    return output


def softmax(x, axis=1):
    # 计算每行的最大值
    row_max = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max = row_max.reshape(-1, 1)
    x = x - row_max

    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


def convert_anchor(anchors, height, width):
    '''
    height, width, _ = img.shape
    :param win_name:
    :param img:
    :param anchors: <class 'tuple'>: (nums, 4)
    :return: boxes_list:[xmin, ymin, xmax, ymax]
    '''
    boxes_list = []
    for index, anchor in enumerate(anchors):
        xmin = anchor[0] * width
        ymin = anchor[1] * height
        xmax = anchor[2] * width
        ymax = anchor[3] * height
        boxes_list.append([xmin, ymin, xmax, ymax])
    return boxes_list


def get_rect_crop_padding(image, rect):
    """
    :param image:
    :param rect:
    :return:
    """
    rect = [int(v) for v in rect]
    rows, cols, d = image.shape  # h,w,d
    x, y, width, height = rect
    crop_x1 = max(0, x)
    crop_y1 = max(0, y)
    crop_x2 = min(cols, x + width)  # 图像范围
    crop_y2 = min(rows, y + height)
    left_x = -x
    top_y = -y
    right_x = x + width - cols
    down_y = y + height - rows
    roi_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
    # 只要存在边界越界的情况，就需要边界填充
    if top_y > 0 or down_y > 0 or left_x > 0 or right_x > 0:
        left_x = np.where(left_x > 0, left_x, 0)
        right_x = np.where(right_x > 0, right_x, 0)
        top_y = np.where(top_y > 0, top_y, 0)
        down_y = np.where(down_y > 0, down_y, 0)
        roi_image = cv2.copyMakeBorder(roi_image, top_y, down_y, left_x, right_x, cv2.BORDER_CONSTANT, value=0)
    return roi_image


def get_bbox_crop_padding(image, bbox):
    """
    :param image:
    :param bbox:
    :return:
    """
    rect = bboxes2rects([bbox])[0]
    roi_image = get_rect_crop_padding(image, rect)
    return roi_image


def get_bboxes_crop_padding(image, bboxes, resize_height=None, resize_width=None):
    """
    :param image:
    :param bboxes:
    :param resize:
    :return:
    """
    rects = bboxes2rects(bboxes)
    roi_images = []
    for rect in rects:
        roi_image = get_rect_crop_padding(image, rect)
        roi_image = resize_image(roi_image, resize_height, resize_width)
        roi_images.append(roi_image)
    return roi_images


def get_rects_crop_padding(image, rects, resize_height=None, resize_width=None):
    """
    :param image:
    :param rects:
    :param resize:
    :return:
    """
    roi_images = []
    for rect in rects:
        roi_image = get_rect_crop_padding(image, rect)
        roi_image = resize_image(roi_image, resize_height, resize_width)
        roi_images.append(roi_image)
    return roi_images


def center_crop(image, crop_size=[112, 112]):
    '''
    central_crop
    :param image: input numpy type image
    :param crop_size:crop_size must less than x.shape[:2]=[crop_h,crop_w]
    :return:
    '''
    h, w = image.shape[:2]
    y = int(round((h - crop_size[0]) / 2.))
    x = int(round((w - crop_size[1]) / 2.))
    y = np.where(y > 0, y, 0)
    x = np.where(x > 0, x, 0)
    return image[y:y + crop_size[0], x:x + crop_size[1]]


def center_crop_padding(image, crop_size):
    """
    :param image:
    :param crop_size: [crop_h,crop_w]
    :return:
    """
    h, w = image.shape[:2]
    y = int(round((h - crop_size[0]) / 2.))
    x = int(round((w - crop_size[1]) / 2.))
    rect = [x, y, crop_size[1], crop_size[0]]
    roi_image = get_rect_crop_padding(image, rect)
    return roi_image


def extend_face2body_bboxes(faces_boxes, width_factor=1.0, height_factor=1.0):
    '''
    extend boxes ,such as extend faces_boxes to body_boxes
    :param faces_boxes:
    :param width_factor:
    :param height_factor:
    :return:
    '''
    body_boxes = []
    for face_box in faces_boxes:
        [x1, y1, x2, y2] = face_box
        w = (x2 - x1)
        h = (y2 - y1)
        x1 = x1 - width_factor * w
        y1 = y1 - width_factor * h
        x1 = np.where(x1 > 0, x1, 0)
        y1 = np.where(y1 > 0, y1, 0)
        w = 3 * width_factor * w
        h = height_factor * h
        body_boxes.append([x1, y1, x1 + w, y1 + h])
    return body_boxes


def extend_rects(rects, scale=[1.0, 1.0]):
    """
    :param rects:
    :param scale: [sx,sy]==>(W,H)
    :return:
    """
    bboxes = rects2bboxes(rects)
    out_bboxes = extend_bboxes(bboxes, scale)
    out_rects = bboxes2rects(out_bboxes)
    return out_rects


def extend_bboxes(bboxes, scale=[1.0, 1.0]):
    """
    :param bboxes: [[xmin, ymin, xmax, ymax]]
    :param scale: [sx,sy]==>(W,H)
    :return:
    """
    out_bboxes = []
    sx = scale[0]
    sy = scale[1]
    for box in bboxes:
        xmin, ymin, xmax, ymax = box
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        ex_w = (xmax - xmin) * sx
        ex_h = (ymax - ymin) * sy
        ex_xmin = cx - 0.5 * ex_w
        ex_ymin = cy - 0.5 * ex_h
        ex_xmax = ex_xmin + ex_w
        ex_ymax = ex_ymin + ex_h
        ex_box = [ex_xmin, ex_ymin, ex_xmax, ex_ymax]
        out_bboxes.append(ex_box)
    return out_bboxes


def get_square_bboxes(bboxes, fixed="H"):
    '''
    :param bboxes:
    :param fixed: (W)width (H)height,(L) longest edge
    :return:
    '''
    new_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        cx, cy = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
        if fixed in ["H", "h"]:
            dd = h / 2
        elif fixed in ["W", "w"]:
            dd = w / 2
        elif fixed in ["L", "l"]:
            l = max(w, h)
            dd = l / 2
        else:
            l = max(w, h)
            dd = l / 2 * fixed
        fxmin = int(cx - dd)
        fymin = int(cy - dd)
        fxmax = int(cx + dd)
        fymax = int(cy + dd)
        new_bbox = (fxmin, fymin, fxmax, fymax)
        new_bboxes.append(new_bbox)
    return new_bboxes


def get_square_rects(rects, fixed="H"):
    """
    ------------------------
    e.g.:
    image_path = "../dataset/dataset/A/test1.jpg"
    image = read_image(image_path, 400, 400)
    rects = [[100, 50, 180, 50]]
    print(rects)
    rect_image = show_image_rects("rect", image, rects, waitKey=1)
    rects2 = get_square_rects(rects, fixed="h")
    # red is square_rects
    rect_image = show_image_rects("rect", rect_image, rects2, color=(255, 0, 0), waitKey=0)
    ------------------------
    :param rects:
    :param fixed:
    :return:
    """
    bboxes = rects2bboxes(rects)
    out_bboxes = get_square_bboxes(bboxes, fixed)
    out_rects = bboxes2rects(out_bboxes)
    return out_rects


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def addMouseCallback(winname, param, callbackFunc=None):
    '''
     添加点击事件
    :param winname:
    :param param:
    :param callbackFunc:
    :return:
    '''
    cv2.namedWindow(winname)

    def default_callbackFunc(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("(x,y)=({},{}),data={}".format(x, y, param[y][x]))

    if callbackFunc is None:
        callbackFunc = default_callbackFunc
    cv2.setMouseCallback(winname, callbackFunc, param)


class EventCv():
    def __init__(self):
        self.image = None

    def update_image(self, image):
        self.image = image

    def callback_print_image(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("(x,y)=({},{}),data={}".format(x, y, self.image[y][x]))

    def add_mouse_event(self, winname, param=None, callbackFunc=None):
        '''
         添加点击事件
        :param winname:
        :param param:
        :param callbackFunc:
        :return:
        '''
        cv2.namedWindow(winname)
        if callbackFunc is None:
            callbackFunc = self.callback_print_image
        cv2.setMouseCallback(winname, callbackFunc, param=param)


def get_video_capture(video_path, width=None, height=None, fps=None):
    """
     --   7W   Pix--> width=320,height=240
     --   30W  Pix--> width=640,height=480
     720P,100W Pix--> width=1280,height=720
     960P,130W Pix--> width=1280,height=1024
    1080P,200W Pix--> width=1920,height=1080
    :param video_path:
    :param width:
    :param height:
    :return:
    """
    video_cap = cv2.VideoCapture(video_path)
    # 设置分辨率
    if width:
        video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        video_cap.set(cv2.CAP_PROP_FPS, 15)
    return video_cap


def get_video_info(video_cap):
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    numFrames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    print("video:width:{},height:{},fps:{},numFrames:{}".format(width, height, fps, numFrames))
    return width, height, numFrames, fps


def get_video_writer(save_path, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frameSize = (int(width), int(height))
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, frameSize)
    print("video:width:{},height:{},fps:{}".format(width, height, fps))
    return video_writer


class CVVideo():
    def __init__(self):
        pass

    def start_capture(self, video_path, save_video=None, detect_freq=1):
        """
        start capture video
        :param video_path: *.avi,*.mp4,...
        :param save_video: *.avi
        :param detect_freq:
        :return:
        """
        # cv2.moveWindow("test", 1000, 100)
        video_cap = get_video_capture(video_path)
        width, height, numFrames, fps = get_video_info(video_cap)
        if save_video:
            self.video_writer = get_video_writer(save_video, width, height, fps)
        # freq = int(fps / detect_freq)
        count = 0
        while True:
            isSuccess, frame = video_cap.read()
            if not isSuccess:
                break
            if count % detect_freq == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.task(frame)
            if save_video:
                self.write_video(frame)
            count += 1
        video_cap.release()

    def write_video(self, frame):
        self.video_writer.write(frame)

    def task(self, frame):
        # TODO
        cv2.imshow("image", frame)
        cv2.moveWindow("image", 0, 0)
        cv2.waitKey(10)
        return frame


if __name__ == "__main__":
    image_path = "../dataset/dataset/A/test1.jpg"
    image = read_image(image_path, 400, 400)
    rects = [[100, 50, 180, 50]]
    print(rects)
    rect_image = show_image_rects("rect", image, rects, waitKey=1)
    rects2 = get_square_rects(rects, fixed="h")
    # red is square_rects
    rect_image = show_image_rects("rect", rect_image, rects2, color=(255, 0, 0), waitKey=0)
