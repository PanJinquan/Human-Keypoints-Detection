"""
@author: Songfang Han
@license: (C) Copyright Dark matter AI.
@contact: hansongfang@dm-ai.cn.
@time: 2019/12/19 下午3:56

show of colored joint and skeleton. Support COCO and MPII format.
Original code from alphapose with moderate editing.

TODO:
    1. add prediction score threshold
"""
import torch
import cv2
import numpy as np
import math

'''Constant color variable'''
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)


def vis_frame_fast(frame, im_res, format='coco'):
    """Modified skeleton show.

    Args:
        frame: (height, width, 3)
        im_res: dictionary key 'result' keeping each human pose estimation
        format: coco or mpii

    Returns:
        img: (height, width ,3)

    """

    if format == 'coco':
        # limb_pair
        l_pair = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (17, 11), (17, 12), (11, 13),
                  (12, 14), (13, 15), (14, 16)]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255),
                   (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77), (204, 77, 255),
                   (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), (77, 255, 222), (77, 196, 255),
                      (77, 135, 255), (191, 255, 77), (77, 255, 77), (77, 222, 255), (255, 156, 127), (0, 127, 255),
                      (255, 127, 77), (0, 77, 255), (255, 77, 36)]
    elif format == 'mpii':
        l_pair = [(8, 9), (11, 12), (11, 10), (2, 1), (1, 0), (13, 14), (14, 15), (3, 4), (4, 5), (8, 7), (7, 6),
                  (6, 2), (6, 3), (8, 12), (8, 13)]
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
    else:
        raise NotImplementedError

    img = frame
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']

        # Add new key point = mean(key[5], key[6])
        if isinstance(kp_preds, np.ndarray):
            kp_preds = np.concatenate((kp_preds, np.expand_dims((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
            kp_scores = np.concatenate((kp_scores, np.expand_dims((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))
        else:
            kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
            kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))

        # Draw key points
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.05:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            cv2.circle(img, (cor_x, cor_y), 4, p_color[n], -1)

        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, line_color[i], 2 * (kp_scores[start_p] + kp_scores[end_p]) + 1)
    return img


def vis_frame(frame, im_res, format='coco'):
    """Modified skeleton show. Draw each limb as ellipse2Poly. Add transparency effect.

        Args:
            frame: (height, width, 3)
            im_res: dictionary key 'result' keeping each human pose estimation
            format: coco or mpii

        Returns:
            img: (height, width ,3)

    """

    if format == 'coco':
        l_pair = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (17, 11), (17, 12), (11, 13),
                  (12, 14), (13, 15), (14, 16)]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255),
                   (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77), (204, 77, 255),
                   (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), (77, 255, 222), (77, 196, 255),
                      (77, 135, 255), (191, 255, 77), (77, 255, 77), (77, 222, 255), (255, 156, 127), (0, 127, 255),
                      (255, 127, 77), (0, 77, 255), (255, 77, 36)]
    elif format == 'mpii':
        l_pair = [(8, 9), (11, 12), (11, 10), (2, 1), (1, 0), (13, 14), (14, 15), (3, 4), (4, 5), (8, 7), (7, 6),
                  (6, 2), (6, 3), (8, 12), (8, 13)]
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
        line_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
    else:
        raise NotImplementedError

    img = frame
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width / 2), int(height / 2)))
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints']
        kp_scores = human['kp_score']

        # mean key[5] + key[6] -> new key
        kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
        kp_scores = torch.cat((kp_scores, torch.unsqueeze((kp_scores[5, :] + kp_scores[6, :]) / 2, 0)))

        # Draw key points
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.05:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (int(cor_x / 2), int(cor_y / 2))
            bg = img.copy()
            cv2.circle(bg, (int(cor_x / 2), int(cor_y / 2)), 2, p_color[n], -1)

            # Now create a mask of logo and create its inverse mask also
            transparency = max(0, min(1, kp_scores[n]))
            img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)

        # Draw proposal score on the head
        middle_eye = (kp_preds[1] + kp_preds[2]) / 4
        middle_cor = int(middle_eye[0]) - 10, int(middle_eye[1]) - 12
        cv2.putText(img, f"{human['proposal_score'].item():.2f}", middle_cor, cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 0, 255))

        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                stick_width = (kp_scores[start_p] + kp_scores[end_p]) + 1
                polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stick_width), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(bg, polygon, line_color[i])
                transparency = max(0, min(1, 0.5 * (kp_scores[start_p] + kp_scores[end_p])))
                img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img
