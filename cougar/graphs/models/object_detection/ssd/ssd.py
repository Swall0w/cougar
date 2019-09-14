from cougar.graphs.models.object_detection.ssd import MultiBoxLoss
from cougar.graphs.models.object_detection.ssd import SSDBoxPredictor
from cougar.graphs.models.object_detection.ssd import SSDBoxHead
from cougar.graphs.models.object_detection.ssd import GeneralizedSSD


class SSD(GeneralizedSSD):
    def __init__(self, backbone, num_classes=None,
                 #transform parameters
                 image_size=300,
                 image_mean=None, image_std=None,
                 # Box parameters
                 box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 ):

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        if box_predictor is None:
            box_predictor = SSDBoxPredictor(num_classes)

        if box_head is None:
            box_head = SSDBoxHead(
                box_predictor=box_predictor,
                multibox_loss=MultiBoxLoss(neg_pos_ratio=3)
            )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]

        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

#        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        transform = None

        super(SSD, self).__init__(backbone, box_head, transform)

