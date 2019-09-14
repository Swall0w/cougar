from cougar.agents import BaseAgent
from collections import OrderedDict
from argparse import ArgumentParser
import torch
import logging
from pathlib import Path
from cougar.common import read_labels
from cougar.common import comm
from cougar.graphs.models.object_detection import build_detection_model
from cougar.solver import make_optimizer, make_lr_scheduler
from cougar.common import CheckPointer
from cougar.data import make_data_loader
from cougar.engine import do_iter_train


class ObjectDetectionAgent(BaseAgent):
    def __init__(self, config: OrderedDict, args: ArgumentParser):
        super().__init__(config)
        self.args = args
        self.labels = read_labels(self.config['dataset']['labels'])

    def load_checkpoint(self, file_name: str):
        pass

    def save_checkpoint(self, file_name: str = "checkpoint.pth", is_best: int = 0):
        pass

    def run(self):
        try:
            self.train()
        except KeyboardInterrupt:
            self.logger.info('You have entered CTRL+C.. Wait to finalize')
        pass

    def train(self):
        # logger
        logger = logging.getLogger('{}.trainer'.format(
            self.config['experiment']['name']
        ))
        model = build_detection_model(self.config, len(self.labels))
        device = torch.device(self.config["model"]["device"])
        model.to(device)

        lr = self.config['trainer']['optimizer']['args']['lr'] * self.args.num_gpus # Scale by num gpus
        optimizer = make_optimizer(self.config, model, lr)
        milestones = [step // self.args.num_gpus for step in self.config['trainer']['lr_scheduler']['args']['milestones']]
        scheduler = make_lr_scheduler(self.config, optimizer, milestones)

        # Initialize mixed-precision training
        #use_mixed_precision = cfg.DTYPE == "float16"
        #amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        #model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

        if self.args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.args.local_rank], output_device=self.args.local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
            )

        arguments = dict()
        arguments["iteration"] = 0
        output_dir = str(Path(self.config['experiment']['output_dir']) / self.config['experiment']['name'])
        save_to_disk = comm.get_rank() == 0
        checkpointer = CheckPointer(model, optimizer, scheduler, output_dir, save_to_disk, logger)
        extra_checkpoint_data = checkpointer.load()
        arguments.update(extra_checkpoint_data)

        max_iter = self.config['trainer']['max_iter'] // self.args.num_gpus
        train_loader = make_data_loader(
            self.config,
            is_train=True,
            distributed=self.args.distributed,
            max_iter=max_iter,
            start_iter=arguments['iteration'],
        )

        self.model = do_iter_train(
            self.config,
            model,
            train_loader,
            optimizer,
            scheduler,
            checkpointer,
            device,
            arguments,
            self.args,
            self.summary_writer
        )

    def validate(self):
        pass

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        if self.summary_writer:
            self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
            self.summary_writer.close()
#        self.data_loader.finalize()
