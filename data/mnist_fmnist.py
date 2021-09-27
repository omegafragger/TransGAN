"""
MNIST and Fashion MNIST data loader.
"""

import torch

from torchvision import datasets
from torchvision import transforms


dataset_dict = {
    'mnist': datasets.MNIST,
    'fashion_mnist': datasets.FashionMNIST
}


def get_loaders(batch_size, dataset='mnist', train=False, num_workers=4, pin_memory=True, norm_transform=True, subclass=0, **kwargs):

    # define transforms
    if train:
        # transform_list = [transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        transform_list = [transforms.ToTensor()]
    else:
        transform_list = [transforms.ToTensor()]
    
    if norm_transform:
        # transform_list.append(transforms.Normalize((0.1307,), (0.3081,),))
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    transform = transforms.Compose(transform_list)

    # load the dataset
    data_dir = "./datasets"

    dataset = dataset_dict[dataset](root=data_dir, train=train, download=True, transform=transform,)

    if (subclass is not None):
        subset_indices = []
        for i in range(len(test_dataset)):
            if test_dataset[i][1] == subclass:
                subset_indices.append(i)

    subset = torch.utils.data.Subset(test_dataset, torch.tensor(subset_indices))
    subsets = [subset for i in range(10)]

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
    )

    return loader