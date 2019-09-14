from cougar.agents import BaseAgent
from collections import OrderedDict


class ObjectDetectionAgent(BaseAgent):
    def __init__(self, config: OrderedDict):
        super().__init__(config)

    def load_checkpoint(self, file_name: str):
        pass

    def save_checkpoint(self, file_name: str = "checkpoint.pth", is_best: int = 0):
        pass

    def run(self):
        pass

    def train(self):
        pass

    def validate(self):
        pass

    def finalize(self):
        pass
