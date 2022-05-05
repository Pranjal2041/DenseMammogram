from typing import List, OrderedDict, Tuple
import warnings
import numpy as np
import pandas as pd
import cv2
import os
from torch.nn.modules.conv import Conv2d
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
from torchvision import models
import detection.transforms as transforms
import torchvision.transforms as T
import detection.utils as utils
import torch.nn.functional as F
import shutil
import json
from detection.engine import train_one_epoch, evaluate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.multiprocessing
import copy
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import RoIHeads




# First we will create the FRCNN model
def get_FRCNN_model(num_classes=1):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True,trainable_backbone_layers=3,min_size=1800,max_size=3600,image_std=(1.0,1.0,1.0),box_score_thresh=0.001)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1) 
    return model

# Some utility heads for Bilateral Model

class RoIpool(nn.Module):

    def __init__(self,pool):
        super().__init__()
        self.box_roi_pool1 = copy.deepcopy(pool)
        self.box_roi_pool2 = copy.deepcopy(pool)
        

    def forward(self,features,proposals,image_shapes):
        x = self.box_roi_pool1(features[0],proposals,image_shapes)
        y = self.box_roi_pool2(features[1],proposals,image_shapes)
        z = torch.cat((x,y),dim=1)
        return z

class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels=None, representation_size=None):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

# Next the bilateral model

class Bilateral_model(nn.Module):

    def __init__(self,frcnn_model):
        super().__init__()
        self.frcnn = frcnn_model
        self.transform = copy.deepcopy(frcnn_model.transform)
        self.backbone1 = copy.deepcopy(frcnn_model.backbone)
        self.backbone2 = copy.deepcopy(frcnn_model.backbone)
        self.rpn = copy.deepcopy(frcnn_model.rpn)
        for param in self.rpn.parameters():
            param.requires_grad = False
        for param in self.backbone1.parameters():
            param.requires_grad = False
        for param in self.backbone2.parameters():
           param.requires_grad = False
        box_roi_pool = RoIpool(frcnn_model.roi_heads.box_roi_pool)
        box_head = TwoMLPHead(512*7*7,1024)
        box_predictor = copy.deepcopy(frcnn_model.roi_heads.box_predictor)
        box_score_thresh=0.001
        box_nms_thresh=0.5
        box_detections_per_img=100
        box_fg_iou_thresh=0.5
        box_bg_iou_thresh=0.5
        box_batch_size_per_image=512
        box_positive_fraction=0.25
        bbox_reg_weights=None
        self.roi_heads =  RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )
    
    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        if self.training:
            return losses

        return detections
    
    
    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor(tuples)]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
                else:
                    raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img[0].shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        images1 = [img[0] for img in images]
        images2 = [img[1] for img in images]
        targets2 = copy.deepcopy(targets)
        #print(images1.shape)
        #print(images2.shape)
        images1, targets = self.transform(images1, targets)
        images2, targets2 = self.transform(images2, targets2)
        
        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        features1 = self.backbone1(images1.tensors)
        features2 = self.backbone2(images2.tensors)
        #print(self.backbone1.out_channels)
        if isinstance(features1, torch.Tensor):
            features1 = OrderedDict([("0", features1)])
        if isinstance(features2, torch.Tensor):
            features2 = OrderedDict([("0", features2)])
        proposals, proposal_losses = self.rpn(images1, features1, targets)
        features = {0:features1,1:features2}
        detections, detector_losses = self.roi_heads(features, proposals, images1.image_sizes, targets)
        detections = self.transform.postprocess(detections, images1.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
