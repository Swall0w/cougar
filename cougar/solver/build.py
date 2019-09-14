import torch

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model, lr):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=cfg['trainer']['optimizer']['args']['momentum'],
        weight_decay=cfg['trainer']['optimizer']['args']['weight_decay'],
    )
    return optimizer


def make_lr_scheduler(cfg, optimizer, milestones=None):
    scheduler = WarmupMultiStepLR(
        optimizer=optimizer,
        milestones=milestones,
        gamma=cfg['trainer']['lr_scheduler']['args']['gamma'],
        warmup_factor=(1.0 / cfg['trainer']['lr_scheduler']['args']['warmup_factor']),
        warmup_iters=cfg['trainer']['lr_scheduler']['args']['warmup_iters'],
    )

    return scheduler