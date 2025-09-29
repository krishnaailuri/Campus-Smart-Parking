import torch
import torch.nn as nn
import numpy as np

class ParkingSpaceModel(nn.Module):
    def __init__(self):
        super(ParkingSpaceModel, self).__init__()
        self.backbone = nn.Sequential(
            self._make_conv_block(3, 64),
            self._make_conv_block(64, 128),
            self._make_conv_block(128, 256),
            self._make_conv_block(256, 512),
        )
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 5, kernel_size=1)  # 4 bbox coords, 1 occupancy
        )

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.detection_head(features)
        return output  # (B, 5, H, W)

def nms(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep
