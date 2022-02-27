from collections import namedtuple
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets import box_utils

GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])


class SSD(nn.Module):
    def __init__(self, num_classes: int,
                 base_net: nn.ModuleList,
                 source_layer_indexes: List[int],
                 extras: nn.ModuleList,
                 classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList,
                 landms_headers: nn.ModuleList,
                 is_test=False,
                 prior_boxes=None,
                 device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.landms_headers = landms_headers
        self.is_test = is_test

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.center_variance = prior_boxes.center_variance
            self.size_variance = prior_boxes.size_variance
            self.priors = prior_boxes.priors.to(self.device)
            self.freeze_header = prior_boxes.freeze_header

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        landms = []
        start_layer_index = 0
        header_index = 0
        end_layer_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location, landm = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
            landms.append(landm)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location, landm = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
            landms.append(landm)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        landms = torch.cat(landms, 1)

        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            # boxes = locations  # this line should be added.
            if self.freeze_header:
                locations = box_utils.convert_locations_to_boxes(locations,
                                                                 self.priors,
                                                                 self.center_variance,
                                                                 self.size_variance)
                locations = box_utils.center_form_to_corner_form(locations)
                # landms = box_utils.decode_landm(landms, self.priors,
                #                                 variances=[self.center_variance, self.size_variance])
                landms = box_utils.decode_landms(landms, self.priors,
                                                 variances=[self.center_variance,
                                                            self.size_variance])
            return confidences, locations, landms
        else:
            return confidences, locations, landms

    def compute_header(self, i, x):
        """
        x=torch.Size([24, 64, 30, 40]),location:torch.Size([24, 12, 30, 40])
        x=torch.Size([24, 128, 15, 20]),location:torch.Size([24, 8, 15, 20])
        x=torch.Size([24, 256, 8, 10]),location:torch.Size([24, 8, 8, 10])
        x=torch.Size([24, 256, 4, 5]),location:torch.Size([24, 12, 4, 5])
        :param i:
        :param x:
        :return:
        """
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        # print("x={},location:{}".format(x.shape, location.shape))
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        # print("location-view:{}".format(location.shape))

        landm = self.landms_headers[i](x)
        # print("x={},location:{}".format(x.shape, location.shape))
        landm = landm.permute(0, 2, 3, 1).contiguous()
        landm = landm.view(landm.size(0), -1, 10)
        # print("location-view:{}".format(location.shape))
        return confidence, location, landm

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)
        self.landms_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if
                      not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)
        self.landms_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)
        self.landms_headers.apply(_xavier_init_)

    def load(self, model, strict=True):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=strict)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPriorLandms(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        """
        :param center_form_priors: priors [cx,cy,w,h]
        :param center_variance:
        :param size_variance:
        :param iou_threshold:
        """
        self.center_form_priors = center_form_priors  # [cx,cy,w,h]
        # [cx,cy,w,h]->[xmin,ymin,xmax,ymax]
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels, gt_landms):
        """
        if landms=-1,preproc will set landms=0, labels=-1
        :param gt_boxes:
        :param gt_labels:
        :param gt_landms:
        :return:
        """
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        if type(gt_landms) is np.ndarray:
            gt_landms = torch.from_numpy(gt_landms)
        boxes, labels, landms = box_utils.assign_priors_landms(gt_boxes,
                                                               gt_labels,
                                                               gt_landms,
                                                               self.corner_form_priors,
                                                               self.iou_threshold)
        # [xmin,ymin,xmax,ymax]-> [cx,cy,w,h]
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes,
                                                         self.center_form_priors,
                                                         self.center_variance,
                                                         self.size_variance)
        landms = box_utils.encode_landm(landms,
                                        self.center_form_priors,
                                        variances=[self.center_variance, self.size_variance])
        return locations, labels, landms


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
