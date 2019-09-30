import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cougar.common import read_labels
from cougar.graphs.models.object_detection import Darknet
from cougar.data.datasets import ListDataset


def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class YOLOAgent(pl.LightningModule):
    def __init__(self, args):
        super(YOLOAgent, self).__init__()
        data_config = parse_data_config(args.data_config)
        self.train_path = data_config["train"]
        self.valid_path = data_config["valid"]
        self.class_names = read_labels(data_config["names"])
        self.model = Darknet(args.model_def)
        self.model.apply(weights_init_normal)
        self.multiscale_training = args.multiscale_training
        self.batch_size = args.batch_size
        self.num_workers = args.n_cpu

        if args.pretrained_weights.endswith(".pth"):
            self.model.load_state_dict(torch.load(args.pretrained_weights))
        else:
            self.model.load_darknet_weights(args.pretrained_weights)

    def forward(self, input, targets):
        loss, outputs = self.model(input, targets)
        return loss, outputs

    def training_step(self, batch, batch_nb):
        _, imgs, targets = batch
        loss, outputs = self.forward(imgs, targets)
        self.model.seen += imgs.size(0)
        return {'loss': loss, 'progress': {}}

    def validation_step(self, batch, batch_nb):
        _, imgs, targets = batch
        loss, outputs = self.forward(imgs, targets)
        self.model.seen += imgs.size(0)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.model.parameters())]

    @pl.data_loader
    def train_dataloader(self):
        dataset = ListDataset(self.train_path, augment=True, multiscale=self.multiscale_training)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        dataset = ListDataset(self.valid_path, augment=False, multiscale=self.multiscale_training)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
        return dataloader



