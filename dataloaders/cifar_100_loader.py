"""
CIFAR-100 Long-Tail Loader
===========================
Loads CIFAR-100 and creates a long-tailed version with configurable
imbalance ratio using exponential decay class distribution.
"""

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import defaultdict


def get_cifar100_loaders(batch_size=128, data_dir='./datasets/CIFAR100',
                          imbalance_factor=100, num_workers=4):
    """
    Get CIFAR-100-LT data loaders.

    Args:
        batch_size: Batch size for DataLoaders
        data_dir: Where to download/find CIFAR-100
        imbalance_factor: Long-tail imbalance ratio (max_samples / min_samples)
        num_workers: DataLoader workers

    Returns:
        train_loader, val_loader, samples_per_class (list of ints)
    """
    num_classes = 100

    # CIFAR-100 proper statistics (computed from training set)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761])
    ])

    # Load datasets (auto-download)
    train_set = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    val_set = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # Create long-tailed version
    samples_per_class = None
    if imbalance_factor > 1:
        train_set, samples_per_class = _create_imbalanced(
            train_set, num_classes, imbalance_factor
        )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, samples_per_class


def _create_imbalanced(dataset, num_classes, imbalance_ratio):
    """Create long-tailed version using exponential decay: n_k = n_max * mu^k."""
    targets = dataset.targets if hasattr(dataset, 'targets') else [
        dataset[i][1] for i in range(len(dataset))
    ]

    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    # Exponential decay: n_k = n_max * (ratio ^ (-k / (C-1)))
    img_max = max(len(indices) for indices in class_indices.values())
    img_num_per_cls = []
    for cls_idx in range(num_classes):
        num = img_max * (imbalance_ratio ** (-cls_idx / (num_classes - 1)))
        img_num_per_cls.append(int(num))

    selected_indices = []
    for cls_idx in range(num_classes):
        indices = class_indices[cls_idx]
        np.random.seed(42)
        np.random.shuffle(indices)
        take = min(img_num_per_cls[cls_idx], len(indices))
        selected_indices.extend(indices[:take])
        img_num_per_cls[cls_idx] = take

    return Subset(dataset, selected_indices), img_num_per_cls
