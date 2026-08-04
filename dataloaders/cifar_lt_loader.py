"""
Long-tailed CIFAR-10 / CIFAR-100 loaders.

Single entry point for the imbalanced CIFAR benchmarks so that this pipeline and the CSL
baseline it extends see exactly the same data. The class-count profile and the per-dataset
transforms are taken from the CSL dataloaders in `dataloaders/Legacy dataloaders/`.

Alongside the usual train and test loaders this returns a deterministic view of the *training*
subset: the memory bank, exemplar selection and feature extraction all need the training
images without random crops and flips, otherwise the features they compare are drawn from
different augmentations of the same image.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Resolution and normalization are taken per-dataset from the legacy CSL dataloaders rather
# than made uniform: CIFAR-10 was trained at the native 32x32 with CIFAR-10 channel
# statistics, CIFAR-100 at ImageNet resolution with a plain (0.5, 0.5) rescale. Reported
# accuracies are only comparable to the CSL baseline if the inputs match, so these are
# deliberately not "corrected" to the textbook per-dataset statistics.
CIFAR10_STATS = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
CIFAR100_STATS = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

DATASET_SPECS = {
    'cifar10': {
        'cls': datasets.CIFAR10,
        'num_classes': 10,
        'stats': CIFAR10_STATS,
        'image_size': 32,
    },
    'cifar100': {
        'cls': datasets.CIFAR100,
        'num_classes': 100,
        'stats': CIFAR100_STATS,
        'image_size': 224,
    },
}


def _normalize_name(name):
    key = str(name).lower().replace('-', '').replace('_', '').replace(' ', '')
    if key in ('cifar10', 'cifar10lt'):
        return 'cifar10'
    if key in ('cifar100', 'cifar100lt'):
        return 'cifar100'
    raise ValueError(f"Unsupported dataset {name!r}; expected 'cifar10' or 'cifar100'")


def get_normalization(dataset):
    """Normalization mean and std used for one CIFAR variant."""
    return DATASET_SPECS[_normalize_name(dataset)]['stats']


def get_num_classes(dataset):
    """Number of classes in one CIFAR variant."""
    return DATASET_SPECS[_normalize_name(dataset)]['num_classes']


def get_default_image_size(dataset):
    """Input resolution the CSL baseline used for one CIFAR variant."""
    return DATASET_SPECS[_normalize_name(dataset)]['image_size']


def build_transforms(dataset, image_size=None):
    """
    Training and evaluation transforms for one CIFAR variant.

    At the native 32x32 the standard long-tail recipe applies (pad-and-crop plus horizontal
    flip). At ImageNet resolution the crop has to resize as well, and brightness jitter is
    added, which is what the CSL CIFAR-100 setup does.
    """
    name = _normalize_name(dataset)
    if image_size is None:
        image_size = DATASET_SPECS[name]['image_size']
    mean, std = DATASET_SPECS[name]['stats']
    normalize = transforms.Normalize(mean, std)

    if image_size == 32:
        train_ops = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        eval_ops = []
    else:
        train_ops = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1),
        ]
        eval_ops = [transforms.Resize((image_size, image_size))]

    transform_train = transforms.Compose(train_ops + [transforms.ToTensor(), normalize])
    transform_eval = transforms.Compose(eval_ops + [transforms.ToTensor(), normalize])
    return transform_train, transform_eval


def compute_samples_per_class(num_classes, img_max, imbalance_ratio):
    """
    Exponential long-tail profile: n_k = n_max * ratio^(-k / (C-1)).

    Class 0 keeps all its images and the last class keeps 1/ratio of them, which is the
    convention the CSL results are reported against.
    """
    if imbalance_ratio <= 1:
        return [img_max] * num_classes

    return [
        max(1, int(img_max * (imbalance_ratio ** (-cls_idx / (num_classes - 1)))))
        for cls_idx in range(num_classes)
    ]


def make_imbalanced_indices(targets, num_classes, imbalance_ratio, seed=42):
    """Pick the indices that make up the long-tailed training subset."""
    targets = np.asarray(targets)
    class_indices = {cls: np.flatnonzero(targets == cls) for cls in range(num_classes)}

    img_max = max(len(idx) for idx in class_indices.values())
    counts = compute_samples_per_class(num_classes, img_max, imbalance_ratio)

    rng = np.random.RandomState(seed)
    selected = []
    for cls_idx in range(num_classes):
        available = class_indices[cls_idx]
        take = min(counts[cls_idx], len(available))
        selected.extend(rng.choice(available, size=take, replace=False).tolist())
        counts[cls_idx] = take

    return selected, counts


def get_cifar_lt_loaders(dataset='cifar10', batch_size=128, imbalance_ratio=100,
                         num_workers=4, data_dir='./data', image_size=None, seed=42):
    """
    Build long-tailed CIFAR loaders.

    Args:
        dataset: 'cifar10' or 'cifar100'
        batch_size: Batch size for every loader
        imbalance_ratio: n_max / n_min; 1 or less leaves the dataset balanced
        num_workers: DataLoader worker processes
        data_dir: Download/cache location
        image_size: None uses the resolution the CSL baseline used for this dataset
        seed: Controls which images each class keeps, so runs are reproducible

    Returns:
        dict with train_loader, train_eval_loader, test_loader, samples_per_class,
        class_names, selected_indices, num_classes and image_size
    """
    name = _normalize_name(dataset)
    spec = DATASET_SPECS[name]
    dataset_cls = spec['cls']
    num_classes = spec['num_classes']
    if image_size is None:
        image_size = spec['image_size']

    transform_train, transform_eval = build_transforms(name, image_size)

    train_set = dataset_cls(root=data_dir, train=True, download=True,
                            transform=transform_train)
    # Same underlying images, deterministic transform. Used for feature extraction and
    # exemplar selection, where augmentation noise would corrupt the comparison.
    train_set_eval = dataset_cls(root=data_dir, train=True, download=True,
                                 transform=transform_eval)
    test_set = dataset_cls(root=data_dir, train=False, download=True,
                           transform=transform_eval)

    selected_indices, samples_per_class = make_imbalanced_indices(
        train_set.targets, num_classes, imbalance_ratio, seed=seed
    )

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
    }

    train_loader = DataLoader(Subset(train_set, selected_indices), shuffle=True,
                              **loader_kwargs)
    train_eval_loader = DataLoader(Subset(train_set_eval, selected_indices), shuffle=False,
                                   **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return {
        'train_loader': train_loader,
        'train_eval_loader': train_eval_loader,
        'test_loader': test_loader,
        'samples_per_class': samples_per_class,
        'class_names': list(train_set.classes),
        'selected_indices': selected_indices,
        'num_classes': num_classes,
        'image_size': image_size,
    }
