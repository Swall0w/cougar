from cougar.graphs.models.object_detection import ssd
from cougar.graphs.models.object_detection.ssd import SSD
from cougar.graphs.models.object_detection.yolo import Darknet


__all__ = ['SSD',
           'Darknet',
           ]


def build_detection_model(config, num_classes):
    if 'TwoStage' == config['model']['type']:
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    else:
        pass

    return model
