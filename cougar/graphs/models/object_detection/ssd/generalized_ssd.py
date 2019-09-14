from collections import OrderedDict
import torch
from torch import nn


class GeneralizedSSD(nn.Module):
    """
    Main class for Generalized Single Shot Multibox Detector.
    Arguments:
        backbone (nn.Module):
        heads (nn.Module): takes the features and computes detections.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone: nn.Module, box_heads: nn.Module, transform):
        super(GeneralizedSSD, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.box_heads = box_heads

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        detections, detector_losses = self.box_heads(features, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)

        if self.training:
            return losses

        return detections
