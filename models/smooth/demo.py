# -*-coding: utf-8 -*-
"""
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2021-01-20 09:46:40
"""
import numpy as np
import cv2
import kalman_filter
import mean_filter


class FilterDemo():
    def __init__(self):
        # self.filter = kalman_filter.KalmanFilter()
        self.filter = mean_filter.MeanFilter(win_size=3, decay=0.9)
        self.curr_mes = np.array([-1, -1])
        self.curr_pre = np.array([-1, -1])

        self.last_mes = np.array([-1, -1])
        self.last_pre = np.array([-1, -1])
        self.frame = np.zeros((800, 800, 3), np.uint8)

    def mouseEvent(self, event, x, y, s, p):
        self.curr_mes = np.array([x, y], dtype=np.float32)

    def task(self):
        self.filter.update(self.curr_mes)
        self.curr_pre = self.filter.predict()

        curr_mes = (int(self.curr_mes[0]), int(self.curr_mes[1]))
        last_mes = (int(self.last_mes[0]), int(self.last_mes[1]))
        curr_pre = (int(self.curr_pre[0]), int(self.curr_pre[1]))
        last_pre = (int(self.last_pre[0]), int(self.last_pre[1]))
        # 绘制测量值轨迹（绿色）
        cv2.line(self.frame, last_mes, curr_mes, (0, 255, 0))
        # 绘制预测值轨迹（红色）
        cv2.line(self.frame, last_pre, curr_pre, (0, 0, 255))
        print("last_pre:{},curr_pre:{}".format(last_pre, curr_pre))
        self.last_mes = self.curr_mes.copy()
        self.last_pre = self.curr_pre.copy()

    def demo(self):
        cv2.namedWindow("Kalman")
        cv2.setMouseCallback("Kalman", self.mouseEvent)
        while (True):
            # 绘制预测值轨迹（蓝色）
            self.task()
            cv2.imshow('Kalman', self.frame)
            key = cv2.waitKey(100)
            if key == 27:  # ESC对应的ASCII码是27
                break
            elif key == ord("c") or key == ord("C"):
                self.frame = np.zeros((800, 800, 3), np.uint8)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    fd = FilterDemo()
    fd.demo()
