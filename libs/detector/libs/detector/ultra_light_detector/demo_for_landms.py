# -*- coding: utf-8 -*-

"""
This code is used to batch detect images in a folder.
"""
import sys
import os

sys.path.append(os.getcwd())
import cv2
import argparse
import torch
import numpy as np
import demo
from utils.nms import py_bbox_nms
from models import box_utils
from utils import image_processing, file_processing, debug

# class_names = ["BACKGROUND", "person"]
class_names = ["BACKGROUND", "face"]


# class_names = ["BACKGROUND", "face", "person"]

def get_parser():
    input_size = [416, 416]  # [W,H]
    priors_type = "face"
    # priors_type = "face_person"
    model_path = "/home/dm/data3/FaceDetector/Ultra-Light-Fast-Generic-Face-Detector-1MB/work_space/RFB_landms_augm/RFB_landms_face/RFB_landms1.0_face_416_416_wider_face_add_lm_10_10_20210203180543/model/best_model_RFB_landms_195_loss7.2719.pth"
    # model_path = "work_space/face_person/RFB_landms_face_person/RFB_landms_face_person_640_360_wider_face_add_lm_10_10_20200728150506/model/best_model_RFB_landms_019_loss14.2951.pth"
    # priors_type = "face"
    # model_path = "data/pretrained/version-slim-640.pth"
    # image_dir = "data/test_images/10.jpg"
    # image_dir = "data/nvr"
    # image_dir = "data/person"
    # image_dir = "data/test_images"
    # image_dir = "libs/release_person/test.jpg"
    # image_dir = "data/dmai/img_andy.jpg"
    # image_dir = "/media/dm/dm/git/python-learning-notes/dataset/dmai_demo/face_person_landmark/landmark"
    # image_dir = "test.jpg"
    image_dir = "data/test_images"
    parser = argparse.ArgumentParser(description='detect_imgs')
    parser.add_argument('--model_path', default=model_path, type=str, help='model_path')
    parser.add_argument('--net_type', default="RFB_landms", type=str,
                        help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
    parser.add_argument('--input_size', nargs='+', help="--input size 112 112", type=int, default=input_size)
    parser.add_argument('--prob_threshold', default=0.5, type=float, help='score threshold')
    parser.add_argument('--iou_threshold', default=0.3, type=float, help='iou_threshold')
    parser.add_argument('--candidate_size', default=200, type=int, help='nms candidate size')
    parser.add_argument('--image_dir', default=image_dir, type=str, help='directory or image path')
    parser.add_argument('--device', default="cuda:0", type=str, help='cuda:0 or cpu')
    parser.add_argument('--priors_type', default=priors_type, type=str, help='priors type:face or person')
    args = parser.parse_args()
    return args


class Detector(demo.Detector):
    """
    Ultra Light Person Detector
    """

    def __init__(self,
                 model_path,
                 net_type,
                 input_size,
                 class_names,
                 priors_type,
                 candidate_size=200,
                 prob_threshold=0.5,
                 iou_threshold=0.1,
                 freeze_header=False,
                 device="cuda:0"):
        """
        :param model_path:  path to model(*.pth) file
        :param net_type:  "RFB" (higher precision) or "slim" (faster)'
        :param input_size: model input size
        :param priors_type: face or person
        :param candidate_size:nms candidate size
        :param prob_threshold: 置信度分数
        :param iou_threshold:  NMS IOU阈值
        :param device: GPU Device
        """
        super(Detector, self).__init__(model_path,
                                       net_type,
                                       input_size,
                                       priors_type=priors_type,
                                       class_names=class_names,
                                       candidate_size=candidate_size,
                                       prob_threshold=prob_threshold,
                                       iou_threshold=iou_threshold,
                                       freeze_header=freeze_header,
                                       device=device)

    @debug.run_time_decorator("forward")
    def forward(self, image_tensor):
        """
        :param image_tensor:
        :return: scores: shape=([1, num_bboxes, num_class])
                 boxes:  shape=([1, num_bboxes, 4]),boxes=[[xmin,ymin,xmax,ymax]]
        """
        with torch.no_grad():
            scores, boxes, ldmks = self.net.forward(image_tensor)
            if not self.prior_boxes.freeze_header:
                # scores = F.softmax(scores, dim=2)
                # boxes = locations  # this line should be added.
                boxes = box_utils.convert_locations_to_boxes(boxes,
                                                             self.priors,
                                                             self.center_variance,
                                                             self.size_variance)
                boxes = box_utils.center_form_to_corner_form(boxes)
                # landms = box_utils.decode_landm(landms, self.priors,
                #                                 variances=[self.center_variance, self.size_variance])
                ldmks = box_utils.decode_landms(ldmks, self.priors,
                                                variances=[self.center_variance,
                                                           self.size_variance])
        return scores, boxes, ldmks

    @debug.run_time_decorator("post_process")
    def post_process1(self, boxes, scores, landms, width, height, top_k, prob_threshold, iou_threshold):
        """
        :param boxes:
        :param scores:
        :param width: orig image width
        :param height:orig image height
        :param top_k: keep top_k results. If k <= 0, keep all the results.
        :param prob_threshold:
        :param iou_threshold:
        :return: boxes, labels, probs

        """
        # this version of nms is slower on GPU, so recommend move data to CPU.
        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes)
        if isinstance(scores, np.ndarray):
            scores = torch.from_numpy(scores)
        picked_boxes_probs = []
        picked_labels = []
        picked_landms = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            index = probs > prob_threshold
            sub_probs = probs[index]
            if sub_probs.size(0) == 0:
                continue
            sub_boxes = boxes[index, :]
            sub_landm = landms[index]
            sub_boxes_probs = torch.cat([sub_boxes, sub_probs.reshape(-1, 1)], dim=1)
            sub_boxes_probs, sub_landm = box_utils.boxes_landms_nms(sub_boxes_probs,
                                                                    landms=sub_landm,
                                                                    nms_method="soft",
                                                                    # nms_method=None,
                                                                    score_threshold=prob_threshold,
                                                                    iou_threshold=iou_threshold,
                                                                    top_k=top_k,
                                                                    candidate_size=self.candidate_size)
            picked_boxes_probs.append(sub_boxes_probs)
            picked_labels += [class_index] * sub_boxes_probs.size(0)
            picked_landms.append(sub_landm)

        if len(picked_boxes_probs) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_boxes_probs = torch.cat(picked_boxes_probs)
        picked_landms = torch.cat(picked_landms)
        boxes = picked_boxes_probs[:, :4]
        probs = picked_boxes_probs[:, 4]
        # conver normalized coordinates to image coordinates
        # bboxes_scale = [width, height, width, height]
        image_size = [width, height]
        bboxes_scale = torch.tensor(image_size * 2, dtype=torch.float32)
        landms_scale = torch.tensor(image_size * 5, dtype=torch.float32)
        boxes = boxes * bboxes_scale
        landms = picked_landms * landms_scale
        landms = landms.reshape(shape=(len(landms), -1, 2))
        labels = torch.tensor(picked_labels)
        return boxes, labels, probs, landms

    @debug.run_time_decorator("post_process")
    def post_process(self, boxes, scores, landms, width, height, top_k, prob_threshold, iou_threshold):
        """
        :param boxes:
        :param scores:
        :param width: orig image width
        :param height:orig image height
        :param top_k: keep top_k results. If k <= 0, keep all the results.
        :param prob_threshold:
        :param iou_threshold:
        :return: boxes, labels, probs

        """
        # this version of nms is slower on GPU, so recommend move data to CPU.
        if not isinstance(boxes, np.ndarray):
            boxes = boxes.data.cpu().numpy()
        if not isinstance(scores, np.ndarray):
            scores = scores.data.cpu().numpy()
        if not isinstance(landms, np.ndarray):
            landms = landms.data.cpu().numpy()
        picked_boxes_probs = []
        picked_labels = []
        picked_landms = []
        for class_index in range(1, scores.shape[1]):
            probs = scores[:, class_index]
            index = probs > prob_threshold
            subset_probs = probs[index]
            if probs.shape[0] == 0 or len(subset_probs) == 0:
                continue
            subset_boxes = boxes[index, :]
            subset_landms = landms[index, :]
            sub_boxes_probs, sub_landm = self.nms_process(subset_boxes,
                                                          subset_probs,
                                                          subset_landms,
                                                          prob_threshold=prob_threshold,
                                                          iou_threshold=iou_threshold,
                                                          top_k=top_k,
                                                          keep_top_k=self.candidate_size)
            picked_boxes_probs.append(sub_boxes_probs)
            picked_labels += [class_index] * sub_boxes_probs.shape[0]
            picked_landms.append(sub_landm)

        if len(picked_boxes_probs) == 0:
            return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
        picked_boxes_probs = np.concatenate(picked_boxes_probs)
        picked_landms = np.concatenate(picked_landms)
        boxes = picked_boxes_probs[:, :4]
        probs = picked_boxes_probs[:, 4]
        # conver normalized coordinates to image coordinates
        # bboxes_scale = [width, height, width, height]
        image_size = [width, height]
        bboxes_scale = np.asarray(image_size * 2, dtype=np.float32)
        landms_scale = np.asarray(image_size * 5, dtype=np.float32)
        boxes = boxes * bboxes_scale
        landms = picked_landms * landms_scale
        landms = landms.reshape(len(landms), -1, 2)
        labels = np.asarray(picked_labels)
        return boxes, labels, probs, landms

    @staticmethod
    @debug.run_time_decorator("detect-nms_process")
    def nms_process(boxes, scores, landms, prob_threshold, iou_threshold, top_k, keep_top_k):
        """
        :param boxes: (num_boxes, 4)
        :param scores:(num_boxes,)
        :param landms:(num_boxes, 10)
        :param prob_threshold:
        :param iou_threshold:
        :param top_k: keep top_k results. If k <= 0, keep all the results.
        :param keep_top_k:
        :return: dets:shape=(num_bboxes,5),[xmin,ymin,xmax,ymax,scores]
                 landms:(num_bboxes,10),[x0,y0,x1,y1,...,x4,y4]
        """
        if not isinstance(boxes, np.ndarray):
            boxes = boxes.data.cpu().numpy()
        if not isinstance(scores, np.ndarray):
            scores = scores.data.cpu().numpy()
        if not isinstance(landms, np.ndarray):
            landms = landms.data.cpu().numpy()
        # ignore low scores
        inds = np.where(scores > prob_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        # keep top-K before NMS
        if top_k >= 0:
            order = scores.argsort()[::-1][:top_k]
        else:
            order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_bbox_nms.py_cpu_nms(dets, iou_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        return dets, landms

    @debug.run_time_decorator("predict")
    def predict(self, rgb_image, top_k=-1, prob_threshold=None, iou_threshold=None):
        """
        :param rgb_image: RGB Image
        :param top_k: keep top_k results. If k <= 0, keep all the results.
        :param prob_threshold:
        :param iou_threshold:
        :return:
        """
        height, width, _ = rgb_image.shape
        image_tensor = self.pre_process(rgb_image)
        image_tensor = torch.from_numpy(image_tensor)
        image_tensor = image_tensor.to(self.device)
        scores, boxes, landms = self.forward(image_tensor)
        boxes = boxes[0].cpu()
        scores = scores[0].cpu()
        landms = landms[0].cpu()
        if not prob_threshold:
            prob_threshold = self.prob_threshold
        if not iou_threshold:
            iou_threshold = self.iou_threshold
        boxes, labels, probs, landms = self.post_process(boxes, scores, landms, width, height, top_k, prob_threshold,
                                                         iou_threshold)
        # boxes, scores, landms = self.adapter_bbox_score_landmarks(dets, landms)
        if len(boxes) == 0:
            return np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
        face_index = labels == self.class_names.index("face")
        landms[~face_index, :] = 0
        return boxes, labels, probs, landms

    # @debug.run_time_decorator("detect_image")
    def detect_image(self, rgb_image, isshow=True):
        """
        :param rgb_image:  input RGB Image
        :param isshow:
        :return:
        """
        boxes, labels, probs, landms = self.predict(rgb_image,
                                                    iou_threshold=self.iou_threshold,
                                                    prob_threshold=self.prob_threshold)
        if not isinstance(boxes, np.ndarray):
            boxes = boxes.detach().cpu().numpy()
        if not isinstance(labels, np.ndarray):
            labels = labels.detach().cpu().numpy()
        if not isinstance(probs, np.ndarray):
            probs = probs.detach().cpu().numpy()
        if not isinstance(landms, np.ndarray):
            landms = landms.detach().cpu().numpy()
        if isshow:
            self.show_landmark_boxes("Det", rgb_image, boxes, labels, probs, landms)
        return boxes, labels, probs, landms
        # return boxes, labels, probs

    def detect_image_dir(self, image_dir, isshow=True):
        """
        :param image_dir: directory or image file path
        :param isshow:<bool>
        :return:
        """
        image_list = file_processing.read_files_lists(image_dir)
        for img_path in image_list:
            # img_path="data/mpii_test/000004812.jpg"
            orig_image = cv2.imread(img_path)
            rgb_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            # rgb_image = image_processing.resize_image(rgb_image, resize_height=800)
            boxes, labels, scores, landms = self.detect_image(rgb_image, isshow=isshow)
            print(boxes, scores, labels)
            print("--" * 20)

    @staticmethod
    def show_landmark_boxes(win_name, image, boxes, labels, probs, landms):
        '''
        显示landmark和boxes
        :param win_name:
        :param image:
        :param landmarks_list: [[x1, y1], [x2, y2]]
        :param bboxes: [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
        :return:
        '''
        boxes_name = ["{}:{:3.4f}".format(l, s) for l, s in zip(labels, probs)]
        rgb_image = image_processing.draw_landmark(image, landms, color=(255, 0, 0), vis_id=False)
        rgb_image = image_processing.draw_image_detection_bboxes(rgb_image, boxes, probs, labels, color=(0, 0, 255))
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.namedWindow(win_name, flags=cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, bgr_image)
        cv2.imwrite("result.jpg", bgr_image)
        cv2.waitKey(0)

    @staticmethod
    def show_landmark_boxes1(win_name, image, boxes, labels, probs, landms):
        '''
        显示landmark和boxes
        :param win_name:
        :param image:
        :param landmarks_list: [[x1, y1], [x2, y2]]
        :param bboxes: [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
        :return:
        '''
        boxes_name = ["{}:{:3.4f}".format(l, s) for l, s in zip(labels, probs)]
        for i in range(len(boxes)):
            print(landms[i])
            rgb_image = image_processing.draw_landmark(image, [landms[i]], vis_id=True)
            rgb_image = image_processing.draw_image_detection_bboxes(rgb_image,
                                                                     [boxes[i]],
                                                                     [probs[i]],
                                                                     [labels[i]])
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.namedWindow(win_name, flags=cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, image)
            cv2.imwrite("result.jpg", image)
            cv2.waitKey(0)

    def start_capture(self, video_path, save_video=None, detect_freq=1):
        """
        start capture video
        :param video_path: *.avi,*.mp4,...
        :param save_video: *.avi
        :param detect_freq:
        :return:
        """
        # cv2.moveWindow("test", 1000, 100)
        video_cap = image_processing.get_video_capture(video_path)
        width, height, numFrames, fps = image_processing.get_video_info(video_cap)
        if save_video:
            self.video_writer = image_processing.get_video_writer(save_video, width, height, fps)
        # freq = int(fps / detect_freq)
        count = 0
        while True:
            isSuccess, frame = video_cap.read()
            if not isSuccess:
                break
            if count % detect_freq == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = self.task(frame)
                # rgb_image = image_processing.resize_image(rgb_image, resize_height=800)
                boxes, labels, scores, landms = self.detect_image(frame, isshow=True)
            if save_video:
                self.video_writer.write(frame)
            count += 1
        video_cap.release()


if __name__ == "__main__":
    args = get_parser()
    print(args)
    net_type = args.net_type
    input_size = args.input_size
    priors_type = args.priors_type
    device = args.device
    # model_path = "RFB-person.pth"
    image_dir = args.image_dir
    model_path = args.model_path
    candidate_size = args.candidate_size
    prob_threshold = args.prob_threshold
    iou_threshold = args.iou_threshold
    det = Detector(model_path,
                   net_type=net_type,
                   input_size=input_size,
                   class_names=class_names,
                   priors_type=priors_type,
                   candidate_size=candidate_size,
                   iou_threshold=iou_threshold,
                   prob_threshold=prob_threshold,
                   device=device)

    det.detect_image_dir(image_dir, isshow=True)
    # video_path = "/media/dm/dm/FaceRecognition/face-recognition-cpp/data/honghe/video/TV_CAM_20191030_141355.mp4"
    # save_video = "data/video/SMTC-group2-result5.mp4"
    # det.start_capture(video_path, save_video=save_video)
