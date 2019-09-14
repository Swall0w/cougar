import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple


class BoxPredictor(nn.Module):
    def __init__(self,
                 num_classes: int,
                 boxes_per_locations: list = [4, 6, 6, 6, 4, 4],
                 backbone_out_channels: list = [512, 1024, 512, 256, 256, 256],
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        self.boxes_per_locations = boxes_per_locations
        self.backbone_out_channels = backbone_out_channels
        assert len(self.boxes_per_locations) == len(self.backbone_out_channels)

        for level, (boxes_per_location, out_channels) in enumerate(
                zip(self.boxes_per_locations, self.backbone_out_channels)):

            self.cls_headers.append(self.cls_block(level, out_channels, boxes_per_location))
            self.reg_headers.append(self.reg_block(level, out_channels, boxes_per_location))
        self.reset_parameters()

    def cls_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reg_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: OrderedDict) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_logits = list()
        bbox_pred = list()

        for feature, cls_header, reg_header in zip(features.values(), self.cls_headers, self.reg_headers):
            print(feature.shape)
            cls_logits.append(cls_header(feature).permute(0, 2, 3, 1).contiguous())
            bbox_pred.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(
            batch_size, -1, self.num_classes)
        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred], dim=1).view(batch_size, -1, 4)

        # When num_classes is 21, the shape of cls_logits is torch.Size([1, 8732, 21])
        return cls_logits, bbox_pred


class SSDBoxPredictor(BoxPredictor):
    def cls_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * self.num_classes, kernel_size=3, stride=1, padding=1)

    def reg_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)
