# -*-coding: utf-8 -*-
"""
    @Project: PyKinect2-OpenCV
    @File   : posture.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-10-09 11:55:11
"""
from utils import geometry_tools


class Posture(object):
    '''张手(左右，双手)，敬礼，捂胸口，叉腰，弯腰，歪头'''
    mpii_joints = {"r_ankle": 0, "r_knee": 1, "r_hip": 2, "l_hip": 3, "l_knee": 4, "l_ankle": 5, "pelvis": 6,
                   "thorax": 7, "upper_neck": 8, "head_top": 9, "r_wrist": 10, "r_elbow": 11, "r_shoulder": 12,
                   "l_shoulder": 13, "l_elbow": 14, "l_wrist": 15}

    def __init__(self):
        self.pos_label = PostureLabel()
        self.key_points = None
        origin = [0, 0]
        self.X_coordinates = geometry_tools.create_vector(origin, point2=[1, 0])
        self.Y_coordinates = geometry_tools.create_vector(origin, point2=[0, 1])
        self.joints = self.mpii_joints
        self.scale = 1.0
        self.base_body_width = 0.4  # 0.4m
        self.epsilon = 1e-5

        self.key_points = None
        self.r_ankle = None
        self.r_knee = None
        self.r_hip = None
        self.l_hip = None
        self.l_knee = None
        self.l_ankle = None
        self.pelvis = None
        self.thorax = None
        self.upper_neck = None
        self.head_top = None
        self.r_wrist = None
        self.r_elbow = None
        self.r_shoulder = None
        self.l_shoulder = None
        self.l_elbow = None
        self.l_wrist = None

    def update_key_points(self, key_points):
        self.key_points = key_points
        self.r_ankle = key_points[0]
        self.r_knee = key_points[1]
        self.r_hip = key_points[2]
        self.l_hip = key_points[3]
        self.l_knee = key_points[4]
        self.l_ankle = key_points[5]
        self.pelvis = key_points[6]
        self.thorax = key_points[7]
        self.upper_neck = key_points[8]
        self.head_top = key_points[9]
        self.r_wrist = key_points[10]
        self.r_elbow = key_points[11]
        self.r_shoulder = key_points[12]
        self.l_shoulder = key_points[13]
        self.l_elbow = key_points[14]
        self.l_wrist = key_points[15]
        body_width = geometry_tools.compute_distance(self.l_shoulder, self.r_shoulder)
        self.scale = self.base_body_width / (body_width + self.epsilon)
        print("scale:{},body_width/base_body_width:{}/{}".
              format(self.scale, body_width, self.base_body_width))

    def __body_vector(self):
        '''胸部->骨盆'''
        v = geometry_tools.create_vector(self.thorax, self.pelvis)
        return v

    def __head_vector(self):
        '''head->thorax'''
        v = geometry_tools.create_vector(self.head_top, self.thorax)
        return v

    def __left_arm_vector(self):
        '''l_shoulder->l_wrist'''
        v = geometry_tools.create_vector(self.l_shoulder, self.l_wrist)
        return v

    def __right_arm_vector(self):
        v = geometry_tools.create_vector(self.r_shoulder, self.r_wrist)
        return v

    def body_bent(self):
        '''
        站立,弯腰,(左负 右正)=>left_angle<0,right_angle>0
        :return:
        '''
        body_vector = self.__body_vector()
        angle = geometry_tools.compute_vector_angle(body_vector,
                                                    self.Y_coordinates,
                                                    minangle=False)
        bias = False if body_vector[0] > 0 else True
        if bias:
            angle = -angle
        return angle

    def hand_expand(self):
        '''
        张手(左右，双手)
        :param key_points:
        :return:
        '''
        left_arm_vector = self.__left_arm_vector()
        right_arm_vector = self.__right_arm_vector()
        l_arm_angle = geometry_tools.compute_vector_angle(left_arm_vector,
                                                          self.Y_coordinates,
                                                          minangle=True)
        r_arm_angle = geometry_tools.compute_vector_angle(right_arm_vector,
                                                          self.Y_coordinates,
                                                          minangle=True)
        return l_arm_angle, r_arm_angle

    def hand_salute(self):
        '''
        敬礼
        :return:
        '''
        l_wrist_head_dist = geometry_tools.compute_distance(self.l_wrist,
                                                            self.head_top) * self.scale
        r_wrist_head_dist = geometry_tools.compute_distance(self.r_wrist,
                                                            self.head_top) * self.scale
        return l_wrist_head_dist, r_wrist_head_dist

    def hand_thorax(self):
        '''
        捂胸口
        :return:
        '''
        l_wrist_thorax_dist = geometry_tools.compute_distance(self.l_wrist,
                                                              self.thorax) * self.scale
        r_wrist_thorax_dist = geometry_tools.compute_distance(self.r_wrist,
                                                              self.thorax) * self.scale
        return l_wrist_thorax_dist, r_wrist_thorax_dist

    def hand_akimbo(self):
        '''
        叉腰
        target_point:手腕->(胸部,骨盆)
        :return:
        '''
        l_wrist_body_dist = geometry_tools.point2line_distance(self.thorax,
                                                               self.pelvis,
                                                               self.l_wrist) * self.scale
        r_wrist_body_dist = geometry_tools.point2line_distance(self.thorax,
                                                               self.pelvis,
                                                               self.r_wrist) * self.scale

        l_elbow_body_dist = geometry_tools.point2line_distance(self.thorax,
                                                               self.pelvis,
                                                               self.l_elbow) * self.scale
        r_elbow_body_dist = geometry_tools.point2line_distance(self.thorax,
                                                               self.pelvis,
                                                               self.r_elbow) * self.scale
        return l_wrist_body_dist, l_elbow_body_dist, r_wrist_body_dist, r_elbow_body_dist

    def head_status(self):
        '''
        pitch:是围绕X轴旋转，也叫做俯仰角，点头 上负下正
        yaw:  是围绕Y轴旋转，也叫偏航角，摇头 左正右负
        roll: 是围绕Z轴旋转，也叫翻滚角，摆头（歪头）左负 右正
        :return:
        '''
        head_vector = self.__head_vector()
        head_roll = geometry_tools.compute_vector_angle(head_vector,
                                                        self.Y_coordinates,
                                                        minangle=False)
        bias = False if head_vector[0] > 0 else True
        if bias:
            head_roll = -head_roll
        return head_roll

    def get_posture(self):
        status = {}
        status["head_roll"] = self.head_status()
        status["body_angle"] = self.body_bent()

        l_arm_angle, r_arm_angle = self.hand_expand()
        status["l_arm_angle"] = l_arm_angle
        status["r_arm_angle"] = r_arm_angle

        l_wrist_thorax_dist, r_wrist_thorax_dist = self.hand_thorax()
        status["l_wrist_thorax_dist"] = l_wrist_thorax_dist
        status["r_wrist_thorax_dist"] = r_wrist_thorax_dist

        l_wrist_head_dist, r_wrist_head_dist = self.hand_salute()
        status["l_wrist_head_dist"] = l_wrist_head_dist
        status["r_wrist_head_dist"] = r_wrist_head_dist

        # l_wrist_body_dist, l_elbow_body_dist,r_wrist_body_dist, r_elbow_body_dist
        akimbo = self.hand_akimbo()
        status["l_wrist_body_dist"] = akimbo[0]
        status["l_elbow_body_dist"] = akimbo[1]
        status["r_wrist_body_dist"] = akimbo[2]
        status["r_elbow_body_dist"] = akimbo[3]
        self.pos_label.update_status(status)
        labels = self.pos_label.get_labels()
        posture_result = {}
        posture_result["status"] = status
        posture_result["labels"] = labels
        return posture_result

    def decode_label(self, status):
        pass


class PostureLabel(object):
    def __init__(self):
        self.status = None
        self.result = {}

    def update_status(self, status):
        self.status = status
        self.set_hand_expand()
        self.set_hand_thorax()
        self.set_body_bent()
        self.set_hand_salute()
        self.set_hand_akimbo()
        self.set_head_status()

    def set_body_bent(self, angle_th=15):
        '''
        (左负 右正)=>left_angle<0,right_angle>0
        :param angle_th:
        :return:
        '''
        body_angle = self.status["body_angle"]
        self.result["l_body_bent"] = False if body_angle > -angle_th else True
        self.result["r_body_bent"] = False if body_angle < angle_th else True

    def set_hand_expand(self, angle_th=25):
        '''
        :param angle_th: 30°
        :return:
        '''
        l_arm_angle = self.status["l_arm_angle"]
        r_arm_angle = self.status["r_arm_angle"]
        self.result["l_hand_expand"] = False if l_arm_angle < angle_th else True
        self.result["r_hand_expand"] = False if r_arm_angle < angle_th else True

    def set_hand_salute(self, dist_th=0.30):
        '''
        敬礼
        :param dist_th:
        :return:
        '''
        l_wrist_head_dist = self.status["l_wrist_head_dist"]
        r_wrist_head_dist = self.status["r_wrist_head_dist"]
        self.result["l_salute"] = False if l_wrist_head_dist > dist_th else True
        self.result["r_salute"] = False if r_wrist_head_dist > dist_th else True

    def set_hand_thorax(self, dist_th=0.30):
        '''
        捂胸口
        :param dist_th: 0.30m
        :return:
        '''
        l_wrist_thorax_dist = self.status["l_wrist_thorax_dist"]
        r_wrist_thorax_dist = self.status["r_wrist_thorax_dist"]
        self.result["l_hand_thorax"] = False if l_wrist_thorax_dist > dist_th else True
        self.result["r_hand_thorax"] = False if r_wrist_thorax_dist > dist_th else True

    def set_hand_akimbo(self, wrist_th=0.28, elbow_th=0.35):
        '''
        叉腰
        手腕->(胸部,骨盆)
        :param wrist_th:
        :param elbow_th:
        :return:
        '''
        l_wrist_body_dist = self.status["l_wrist_body_dist"]
        l_elbow_body_dist = self.status["l_elbow_body_dist"]
        r_wrist_body_dist = self.status["r_wrist_body_dist"]
        r_elbow_body_dist = self.status["r_elbow_body_dist"]
        self.result["l_akimbo"] = True if l_wrist_body_dist < wrist_th and l_elbow_body_dist > elbow_th else False
        self.result["r_akimbo"] = True if r_wrist_body_dist < wrist_th and r_elbow_body_dist > elbow_th else False

    def set_head_status(self, angle_th=15):
        '''
        pitch:是围绕X轴旋转，也叫做俯仰角，点头 上负下正
        yaw:  是围绕Y轴旋转，也叫偏航角，摇头 左正右负
        roll: 是围绕Z轴旋转，也叫翻滚角，摆头（歪头）左负 右正
        :param angle_th:
        :return:
        '''
        head_roll = self.status["head_roll"]
        self.result["l_head_roll"] = False if head_roll > -angle_th else True
        self.result["r_head_roll"] = False if head_roll < angle_th else True

    def get_labels(self):
        return self.result


if __name__ == "__main__":
    save_dir = "../data/data01"
