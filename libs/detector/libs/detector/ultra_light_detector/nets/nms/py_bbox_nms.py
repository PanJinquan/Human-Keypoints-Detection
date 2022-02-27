# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-05 11:01:49
# --------------------------------------------------------
"""

import numpy as np


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def per_class_nms(boxes, scores, prob_threshold, iou_threshold, top_k=200, keep_top_k=100):
    """
    :param boxes: (num_boxes, 4)
    :param scores:(num_boxes,)
    :param landms:(num_boxes, 10)
    :param prob_threshold:
    :param iou_threshold:
    :param top_k: keep top_k results. If k <= 0, keep all the results.
    :param keep_top_k: keep_top_k<=top_k
    :return: dets:shape=(num_bboxes,5),[xmin,ymin,xmax,ymax,scores]
             landms:(num_bboxes,10),[x0,y0,x1,y1,...,x4,y4]
    """
    # ignore low scores
    inds = np.where(scores > prob_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]
    # keep top-K before NMS
    if top_k >= 0:
        order = scores.argsort()[::-1][:top_k]
    else:
        order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, iou_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    return dets


def bboxes_nms(input_bboxes, input_scores, prob_threshold, iou_threshold, top_k=200, keep_top_k=100):
    """
    :param input_bboxes: (num_boxes, 4)
    :param input_scores: (num_boxes,num_class)
    :param prob_threshold:
    :param iou_threshold:
    :param top_k: keep top_k results. If k <= 0, keep all the results.
    :param keep_top_k: keep_top_k<=top_k
    :return: dets:shape=(num_bboxes,5),[xmin,ymin,xmax,ymax,scores]
    """
    if not isinstance(input_bboxes, np.ndarray):
        input_bboxes = np.asarray(input_bboxes)
    if not isinstance(input_scores, np.ndarray):
        input_scores = np.asarray(input_scores)

    picked_boxes_probs = []
    picked_labels = []
    for class_index in range(0, input_scores.shape[1]):
        probs = input_scores[:, class_index]
        index = probs > prob_threshold
        subset_probs = probs[index]
        if probs.shape[0] == 0:
            continue
        subset_boxes = input_bboxes[index, :]
        sub_boxes_probs = per_class_nms(subset_boxes,
                                        subset_probs,
                                        prob_threshold=prob_threshold,
                                        iou_threshold=iou_threshold,
                                        top_k=top_k,
                                        keep_top_k=keep_top_k)
        picked_boxes_probs.append(sub_boxes_probs)
        picked_labels += [class_index] * sub_boxes_probs.shape[0]

    if len(picked_boxes_probs) == 0:
        return np.asarray([]), np.asarray([]), np.asarray([])
    picked_boxes_probs = np.concatenate(picked_boxes_probs)
    boxes = picked_boxes_probs[:, :4]
    probs = picked_boxes_probs[:, 4]
    labels = np.asarray(picked_labels)
    return boxes, labels, probs
