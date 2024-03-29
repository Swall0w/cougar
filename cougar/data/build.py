import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from cougar.data import samplers
from cougar.data.datasets import build_dataset
from cougar.data.transforms import build_transforms
from distutils.util import strtobool
from cougar.structures.image_list import to_image_list


class BatchCollator(object):
    def __init__(self, is_train=True, size_divisible=0):
        self.is_train = is_train
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
#        images = to_image_list(transposed_batch[0], self.size_divisible)
        # TODO: reshape same size of images
        img_ids = default_collate(transposed_batch[2])
        targets = transposed_batch[1]
#        img_ids = transposed_batch[2]

        return images, targets, img_ids


def make_data_loader(cfg, is_train=True, distributed=False, max_iter=None, start_iter=0):
    train_transform = build_transforms(cfg, is_train=is_train)

    dataset_list = cfg['dataset']['train_dataset'] if is_train else cfg['dataset']['test_dataset']

    datasets = build_dataset(
        cfg,
        dataset_list,
        transform=train_transform,
        is_train=is_train
    )

    shuffle = is_train or distributed

    data_loaders = []

    for dataset in datasets:
        if distributed:
            sampler = samplers.DistributedSampler(dataset, shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        batch_size = cfg['trainer']['batch_size'] if is_train else cfg['tester']['batch_size']
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False
        )

        if max_iter is not None:
            batch_sampler = samplers.IterationBasedBatchSampler(
                batch_sampler,
                num_iterations=max_iter,
                start_iter=start_iter
            )

        data_loader = DataLoader(dataset,
                                 num_workers=cfg["data_loader"]["train"]["num_workers"],
                                 batch_sampler=batch_sampler,
                                 pin_memory=strtobool(cfg['data_loader']['train']['pin_memory']),
                                 collate_fn=BatchCollator(is_train),
                                 )
        data_loaders.append(data_loader)

    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]

    return data_loaders