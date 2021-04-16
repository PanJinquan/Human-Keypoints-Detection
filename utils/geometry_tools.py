# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : geometry_tools.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-11 09:23:21
"""
# -*- coding: utf-8 -*-

import numpy as np
import copy
import cv2


def compute_distance(vector1, vector2):
    d = np.sqrt(np.sum(np.square(vector1 - vector2)))
    # d = np.linalg.norm(vector1 - vector2)
    return d


def compute_point2area_distance(area_point, target_point):
    point1 = area_point[0, :]
    point2 = area_point[1, :]
    point3 = area_point[2, :]
    point4 = target_point
    d = point2area_distance(point1, point2, point3, point4)
    return d


def compute_point2point_distance(area_point, target_point):
    # point1 = area_point[0, :]
    # point2 = area_point[1, :]
    # point3 = area_point[2, :]
    mean_point = np.mean(area_point, axis=0)
    d = np.sqrt(np.sum(np.square(mean_point - target_point)))
    # d = np.linalg.norm(point1 - target_point)
    return d


def define_area(point1, point2, point3):
    """
    法向量    ：n={A,B,C}
    空间上某点：p={x0,y0,z0}
    点法式方程：A(x-x0)+B(y-y0)+C(z-z0)=Ax+By+Cz-(Ax0+By0+Cz0)
    https://wenku.baidu.com/view/12b44129af45b307e87197e1.html
    :param point1:
    :param point2:
    :param point3:
    :param point4:
    :return:（Ax, By, Cz, D）代表：Ax + By + Cz + D = 0
    """
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    point3 = np.asarray(point3)
    AB = np.asmatrix(point2 - point1)
    AC = np.asmatrix(point3 - point1)
    N = np.cross(AB, AC)  # 向量叉乘，求法向量
    # Ax+By+Cz
    Ax = N[0, 0]
    By = N[0, 1]
    Cz = N[0, 2]
    D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])
    return Ax, By, Cz, D


def define_line(point1, point2):
    '''
    y-y1=k(x-x1),k=(y2-y1)/(x2-x1)=>
    kx-y+(y1-kx1)=0 <=> Ax+By+C=0
    => A=K=(y2-y1)/(x2-x1)
    => B=-1
    => C=(y1-kx1)
    :param point1:
    :param point2:
    :return:
    '''
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    A = (y2 - y1) / (x2 - x1)  # K
    B = -1
    C = y1 - A * x1
    return A, B, C


def point2line_distance(point1, point2, target_point):
    '''
    :param point1: line point1
    :param point2: line point2
    :param target_point: target_point
    :return:
    '''
    A, B, C = define_line(point1, point2)
    mod_d = A * target_point[0] + B * target_point[1] + C
    mod_sqrt = np.sqrt(np.sum(np.square([A, B])))
    d = abs(mod_d) / mod_sqrt
    return d


def point2area_distance(point1, point2, point3, point4):
    """
    :param point1:数据框的行切片，三维
    :param point2:
    :param point3:
    :param point4:
    :return:点到面的距离
    """
    Ax, By, Cz, D = define_area(point1, point2, point3)
    mod_d = Ax * point4[0] + By * point4[1] + Cz * point4[2] + D
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    d = abs(mod_d) / mod_area
    return d


def create_vector(point1, point2):
    '''
    P12 = point2-point1
    :param point1:
    :param point2:
    :return:
    '''
    if not isinstance(point1, np.ndarray):
        point1 = np.asarray(point1, dtype=np.float32)
    if not isinstance(point2, np.ndarray):
        point2 = np.asarray(point2, dtype=np.float32)
    return point2 - point1


def create_2vectors(P1, P2, Q1, Q2):
    '''
    P12 = P2-P1
    Q21 = Q2-Q1
    :param P1:
    :param P2:
    :param Q1:
    :param Q2:
    :return:
    '''
    v1 = create_vector(P1, P2)
    v2 = create_vector(Q1, Q2)
    return v1, v2


def radian2angle(radian):
    '''弧度->角度'''
    angle = radian * (180 / np.pi)
    return angle


def angle2radian(angle):
    '''角度 ->弧度'''
    radian = angle * np.pi / 180.0
    return radian


def compute_point_angle(P1, P2, Q1, Q2, minangle=True):
    x, y = create_2vectors(P1, P2, Q1, Q2)
    angle = compute_vector_angle(x, y, minangle=minangle)
    return angle


def compute_horizontal_angle(P1, P2):
    """
    计算逆时针，水平角度大小
    :param P1:
    :param P2:
    :param Q1:
    :param Q2:
    :param minangle:
    :return:
    """
    Q1 = (0, 0)
    Q2 = (1, 0)
    x, y = create_2vectors(P1, P2, Q1, Q2)
    angle = compute_vector_angle(x, y, minangle=False)
    if x[1] < 0:
        # 在第三和四象限
        angle = -angle
    return angle


def compute_vector_angle(a, b, minangle=True):
    '''
    cosφ = u·v/|u||v|
    https://wenku.baidu.com/view/301a6ba1250c844769eae009581b6bd97f19bca3.html?from=search
    :param a:
    :param b:
    :return:
    '''
    # 两个向量
    x = np.array(a)
    y = np.array(b)
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    value = x.dot(y) / ((Lx * Ly) + 1e-6)  # cosφ = u·v/|u||v|
    radian = np.arccos(value)
    angle = radian2angle(radian)
    if minangle:
        # angle = np.where(angle > 90, 180 - angle, angle)
        angle = angle if angle < 90 else 180 - angle
    return angle


def line_test():
    '''
    angle: 56.789092174788685
    radian: 0.9911566376686096
    cosφ = u·v/|u||v|
    :return:
    '''
    # 两个向量
    point1 = np.array([1, 1, 0.5], dtype=np.float32)
    point2 = np.array([0.5, 0, 1], dtype=np.float32)
    point3 = np.array([1, 0, 0], dtype=np.float32)
    point4 = np.array([0.5, 0, 1], dtype=np.float32)
    angle = compute_point_angle(point1, point2, point3, point4)
    radian = angle2radian(angle)
    print("angle:", angle)
    print("radian:", radian)


def image2plane_coordinates(image_points, height=0):
    """
    将图像坐标转换到平面坐标或者将图像坐标(x,y)转换到平面坐标(x`,y`)：
    :param image_points:
    :param height: height=0 (默认值)： 表示平面原点在图像左上角
    :param         height=image.height： 表示平面原点在图像左上下角
    :return:
    """
    if not isinstance(image_points, np.ndarray):
        points = np.asarray(image_points)
    else:
        points = image_points.copy()
    points[:, 1] = height - points[:, 1]
    return points


def rotate_point(point1, point2, angle, height):
    """
    点point1绕点point2旋转angle后的点
    ======================================
    在平面坐标上，任意点P(x1,y1)，绕一个坐标点Q(x2,y2)旋转θ角度后,新的坐标设为(x, y)的计算公式：
    x= (x1 - x2)*cos(θ) - (y1 - y2)*sin(θ) + x2 ;
    y= (x1 - x2)*sin(θ) + (y1 - y2)*cos(θ) + y2 ;
    ======================================
    将图像坐标(x,y)转换到平面坐标(x`,y`)：
    x`=x
    y`=height-y
    :param point1:
    :param point2: base point (基点)
    :param angle: 旋转角度，正：表示逆时针，负：表示顺时针
    :param height: 图像的height
    :return:
    """
    x1, y1 = point1
    x2, y2 = point2
    # 将图像坐标转换到平面坐标
    y1 = height - y1
    y2 = height - y2
    x = (x1 - x2) * np.cos(np.pi / 180.0 * angle) - (y1 - y2) * np.sin(np.pi / 180.0 * angle) + x2
    y = (x1 - x2) * np.sin(np.pi / 180.0 * angle) + (y1 - y2) * np.cos(np.pi / 180.0 * angle) + y2
    # 将平面坐标转换到图像坐标
    y = height - y
    return (x, y)


def rotate_points(points, centers, angle, height):
    """
    eg.:
    height, weight, d = image.shape
    point1 = [[300, 200],[50, 200]]
    point1 = np.asarray(point1)
    center = [[200, 200]]
    point3 = rotate_points(point1, center, angle=30, height=height)
    :param points:
    :param centers:
    :param angle:
    :param height:
    :return:
    """
    if not isinstance(points, np.ndarray):
        points = np.asarray(points)
    if not isinstance(centers, np.ndarray):
        centers = np.asarray(centers)
    dst_points = points.copy()
    # 将图像坐标转换到平面坐标
    dst_points[:, 1] = height - dst_points[:, 1]
    centers[:, 1] = height - centers[:, 1]
    x = (dst_points[:, 0] - centers[:, 0]) * np.cos(np.pi / 180.0 * angle) - (
            dst_points[:, 1] - centers[:, 1]) * np.sin(np.pi / 180.0 * angle) + centers[:, 0]
    y = (dst_points[:, 0] - centers[:, 0]) * np.sin(np.pi / 180.0 * angle) + (
            dst_points[:, 1] - centers[:, 1]) * np.cos(np.pi / 180.0 * angle) + centers[:, 1]
    # 将平面坐标转换到图像坐标
    y = height - y
    dst_points[:, 0] = x
    dst_points[:, 1] = y
    return dst_points


def demo_for_rotate_point():
    import cv2
    from utils import image_processing

    image_path = "4.jpg"
    image = cv2.imread(image_path)
    image = image_processing.resize_image(image, resize_width=800, resize_height=800)
    height, weight, d = image.shape
    point1 = [[300, 200],[50, 200]]
    point1 = np.asarray(point1)
    # center = [[200, 200]]
    center = [(weight/2.0, height/2.0)]
    for i in range(360):
        point2 = rotate_points(point1, center, angle=i, height=height)
        image_vis = image_processing.draw_points_text(image, center, texts=["center"], drawType="simple")
        image_vis = image_processing.draw_points_text(image_vis, point1, texts=[str(i)]*len(point1), drawType="simple")
        image_vis = image_processing.draw_points_text(image_vis, point2, texts=[str(i)]*len(point1), drawType="simple")
        image_processing.cv_show_image("image", image_vis)


def line_test():
    '''
    angle: 56.789092174788685
    radian: 0.9911566376686096
    cosφ = u·v/|u||v|
    :return:
    '''
    # 两个向量
    point1 = np.array([1, 1, 0.5], dtype=np.float32)
    point2 = np.array([0.5, 0, 1], dtype=np.float32)
    point3 = np.array([1, 0, 0], dtype=np.float32)
    point4 = np.array([0.5, 0, 1], dtype=np.float32)
    angle = compute_point_angle(point1, point2, point3, point4)
    radian = angle2radian(angle)
    print("angle:", angle)
    print("radian:", radian)

def line_test2():
    '''
    angle: 56.789092174788685
    radian: 0.9911566376686096
    cosφ = u·v/|u||v|
    :return:
    '''
    # 两个向量
    point1 = np.array([0, 0], dtype=np.float32)
    point2 = np.array([1, 1], dtype=np.float32)
    point3 = np.array([0, 0], dtype=np.float32)
    point4 = np.array([1, 1], dtype=np.float32)
    v1 = create_vector(point1, point2)
    v2 = create_vector(point3, point4)
    angle = compute_vector_angle(v1, v2, minangle=True)
    print("angle:", angle)




if __name__ == '__main__':
    line_test()
    line_test2()
