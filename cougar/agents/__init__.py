from cougar.agents.baseagent import BaseAgent
from cougar.agents.object_detection_agent import ObjectDetectionAgent
from collections import OrderedDict
from cougar.agents.yolo_agent import YOLOAgent


__all__ = ['BaseAgent', 'ObjectDetectionAgent',
           'build_agent',
           'YOLOAgent',
          ]


def build_agent(config: OrderedDict):
    return eval('{}Agent'.format(config['experiment']['task']))
