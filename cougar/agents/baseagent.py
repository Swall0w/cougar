"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
from collections import OrderedDict
import logging
from pathlib import Path


class BaseAgent(object):
    def __init__(self, config: OrderedDict, **kwargs):
        self.config = config
        self.logger = logging.getLogger('{}.Agent'.format(
            config['experiment']['name']
        ))

        if 'tensorboard' in config['trainer']:
            import tensorboardX
            self.summary_writer = tensorboardX.SummaryWriter(log_dir=str(
                Path(config['experiment']['output_dir']) / config['experiment']['name'] / 'tf_logs/'
            ))
        else:
            self.summary_writer = None

    def load_checkpoint(self, file_name: str):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name: str = "checkpoint.pth", is_best: int = 0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def run(self):
        """
        The main operator
        :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError