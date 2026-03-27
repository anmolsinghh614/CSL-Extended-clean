"""
Dataset Configuration Registry
================================
Per-dataset metadata matching the CSL paper (Section IV: Implementation).
Each config contains training hyperparameters, normalization stats,
and architecture choices exactly as described in the paper.
"""

DATASET_CONFIGS = {
    'cifar10': {
        'name': 'CIFAR-10-LT',
        'num_classes': 10,
        'image_size': 32,
        'norm_mean': [0.4914, 0.4822, 0.4465],
        'norm_std': [0.2023, 0.1994, 0.2010],
        'backbone': 'ResNet32',       # ResNet-32 (wraps resnet34)
        'epochs': 200,
        'lr': 0.1,
        'lr_decay_epochs': [160, 180],
        'lr_decay_factor': 0.01,
        'weight_decay': 5e-4,
        'momentum': 0.9,
        'batch_size': 128,
        'imbalance_ratios': [50, 100],
        'class_names': [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ],
        'auto_download': True,
    },

    'cifar100': {
        'name': 'CIFAR-100-LT',
        'num_classes': 100,
        'image_size': 32,
        'norm_mean': [0.5071, 0.4867, 0.4408],
        'norm_std': [0.2675, 0.2565, 0.2761],
        'backbone': 'ResNet32',
        'epochs': 200,
        'lr': 0.1,
        'lr_decay_epochs': [160, 180],
        'lr_decay_factor': 0.01,
        'weight_decay': 5e-4,
        'momentum': 0.9,
        'batch_size': 128,
        'imbalance_ratios': [50, 100, 200],
        'class_names': None,  # Too many — use numeric indices
        'auto_download': True,
    },

    'tiny_imagenet': {
        'name': 'Tiny ImageNet',
        'num_classes': 200,
        'image_size': 64,
        'norm_mean': [0.485, 0.456, 0.406],
        'norm_std': [0.229, 0.224, 0.225],
        'backbone': 'ResNet18',       # Paper uses ResNet-18
        'epochs': 100,
        'lr': 0.1,
        'lr_decay_epochs': [50, 90],
        'lr_decay_factor': 0.1,
        'weight_decay': 2e-4,
        'momentum': 0.9,
        'batch_size': 128,
        'imbalance_ratios': [100],
        'class_names': None,
        'auto_download': True,
    },

    'imagenet_lt': {
        'name': 'ImageNet-LT',
        'num_classes': 1000,
        'image_size': 224,
        'norm_mean': [0.485, 0.456, 0.406],
        'norm_std': [0.229, 0.224, 0.225],
        'backbone': 'ResNet50',
        'epochs': 120,
        'lr': 0.1,
        'lr_decay_epochs': [60, 80],
        'lr_decay_factor': 0.1,
        'weight_decay': 2e-4,
        'momentum': 0.9,
        'batch_size': 256,
        'imbalance_ratios': [None],   # Natural long-tail (Pareto)
        'class_names': None,
        'auto_download': False,
        'train_txt': 'dataloaders/ImageNet_LT/ImageNet_LT_train.txt',
        'val_txt': 'dataloaders/ImageNet_LT/ImageNet_LT_val.txt',
        'test_txt': 'dataloaders/ImageNet_LT/ImageNet_LT_test.txt',
    },

    'inaturalist': {
        'name': 'iNaturalist-2018',
        'num_classes': 8142,
        'image_size': 224,
        'norm_mean': [0.466, 0.471, 0.380],
        'norm_std': [0.195, 0.194, 0.192],
        'backbone': 'ResNet50',
        'epochs': 200,
        'lr': 0.05,
        'lr_decay_epochs': [160, 180],
        'lr_decay_factor': 0.1,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'batch_size': 256,
        'imbalance_ratios': [None],   # Natural long-tail
        'class_names': None,
        'auto_download': False,
        'train_txt': 'dataloaders/Inaturalist18/iNaturalist18_train.txt',
        'val_txt': 'dataloaders/Inaturalist18/iNaturalist18_val.txt',
        'test_txt': 'dataloaders/Inaturalist18/iNaturalist18_test.txt',
    },
}


def get_dataset_config(dataset_name):
    """Get config for a dataset by name."""
    if dataset_name not in DATASET_CONFIGS:
        available = ', '.join(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    return DATASET_CONFIGS[dataset_name]


def get_available_datasets():
    """Get list of datasets that can auto-download."""
    return [k for k, v in DATASET_CONFIGS.items() if v.get('auto_download', False)]
