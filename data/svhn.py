"""
SVHN data loader.
"""

import torch

from torchvision import datasets
from torchvision import transforms


dataset_dict = {
    'svhn': datasets.SVHN
}


def get_loaders(batch_size, dataset='svhn', train=False, num_workers=4, pin_memory=True, norm_transform=True, **kwargs):

    # define transforms
    if train:
        transform_list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    else:
        transform_list = [transforms.ToTensor()]
    
    if norm_transform:
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],))

    transform = transforms.Compose(transform_list)

    # load the dataset
    data_dir = "./datasets"

    dataset = dataset_dict[dataset](root=data_dir, split='train' if train else 'test', download=True, transform=transform,)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
    )

    return loader