from cougar.agents.baseagent import BaseAgent
from cougar.agents.object_detection_agent import ObjectDetectionAgent
from collections import OrderedDict


__all__ = ['BaseAgent', 'ObjectDetectionAgent',
           'build_agent'
          ]


def build_agent(config: OrderedDict):
    return eval('{}Agent'.format(config['experiment']['task']))
