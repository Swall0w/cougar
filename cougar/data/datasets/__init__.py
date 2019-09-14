from cougar.data.datasets.voc import get_voc
from torch.utils.data import ConcatDataset


_DATASETS = {
    'VOCDataset': get_voc,
}


def build_dataset(config, dataset_list, transform=None, is_train=True):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        dataset = _DATASETS[config['dataset']['type']](
            root=config['dataset']['root'],
            image_set=dataset_name,
            transforms=transform
        )
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]


__all__ = ['get_voc', 'build_dataset',
           ]