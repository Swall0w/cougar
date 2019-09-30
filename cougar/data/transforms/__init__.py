from cougar.data.transforms.transforms import Compose, RandomHorizontalFlip, ToTensor, horisontal_flip


def build_transforms(config, is_train):
    transforms = []
    transforms.append(ToTensor())
    if is_train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


__all__ = ['Compose', 'RandomHorizontalFlip', 'ToTensor',
           'build_transforms',
          ]