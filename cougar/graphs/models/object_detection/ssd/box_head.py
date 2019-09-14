import torch
import torch.nn as nn


class SSDBoxHead(nn.Module):
    def __init__(self,
                 box_predictor: nn.Module,
                 multibox_loss: nn.Module,
                 ):
        super().__init__()
        self.predictor = box_predictor
        self.loss_evaluator = multibox_loss
#        self.post_processor = PostProcessor(cfg)
#        self.priors = None

    def forward(self, features, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'

        class_logits, box_regression = self.predictor(features)

        result, losses = dict(), dict()

        if self.training:
            gt_boxes, gt_labels = targets['boxes'], targets['labels']
            reg_loss, cls_loss = self.loss_evaluator(class_logits, box_regression, gt_labels, gt_boxes)
            losses = dict(
                reg_loss=reg_loss,
                cls_loss=cls_loss,
            )

        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                    )
                )

        return result, losses

    def postprocess_detections(self, class_logits, box_regression, image_shapes):
        pass

#    def _forward_test(self, cls_logits, bbox_pred):
#        if self.priors is None:
#            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
#        scores = F.softmax(cls_logits, dim=2)
#        boxes = box_utils.convert_locations_to_boxes(
#            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
#        )
#        boxes = box_utils.center_form_to_corner_form(boxes)
#        detections = (scores, boxes)
#        detections = self.post_processor(detections)
#        return detections, {}