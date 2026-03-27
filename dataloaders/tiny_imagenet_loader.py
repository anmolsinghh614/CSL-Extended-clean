"""
Tiny ImageNet Loader
====================
Auto-downloads Tiny ImageNet (200 classes, 64×64 images) and 
creates a long-tailed version with configurable imbalance ratio.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import urllib.request
import zipfile
import shutil


class TinyImageNetDataset(Dataset):
    """Tiny ImageNet dataset."""

    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.image_paths = []
        self.labels = []

        if split == 'train':
            self._load_train()
        elif split == 'val':
            self._load_val()
        else:
            raise ValueError(f"Unknown split: {split}")

    def _load_train(self):
        train_dir = os.path.join(self.root_dir, 'train')
        # Build class-to-id mapping
        class_dirs = sorted(os.listdir(train_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_dirs)}

        for cls_name in class_dirs:
            cls_dir = os.path.join(train_dir, cls_name, 'images')
            if not os.path.isdir(cls_dir):
                continue
            cls_idx = self.class_to_idx[cls_name]
            for img_name in os.listdir(cls_dir):
                if img_name.endswith('.JPEG'):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(cls_idx)

    def _load_val(self):
        val_dir = os.path.join(self.root_dir, 'val')
        annotations_file = os.path.join(val_dir, 'val_annotations.txt')

        # Build class-to-id mapping from train directory
        train_dir = os.path.join(self.root_dir, 'train')
        class_dirs = sorted(os.listdir(train_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_dirs)}

        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                cls_name = parts[1]
                img_path = os.path.join(val_dir, 'images', img_name)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def download_tiny_imagenet(data_dir='./datasets/tiny-imagenet-200'):
    """Download and extract Tiny ImageNet if not present."""
    if os.path.exists(os.path.join(data_dir, 'train')):
        print(f"  Tiny ImageNet already exists at {data_dir}")
        return data_dir

    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = os.path.join(os.path.dirname(data_dir), 'tiny-imagenet-200.zip')

    os.makedirs(os.path.dirname(data_dir), exist_ok=True)

    print(f"  Downloading Tiny ImageNet from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    print(f"  Extracting to {data_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(data_dir))

    os.remove(zip_path)
    print(f"  ✓ Tiny ImageNet ready at {data_dir}")
    return data_dir


def get_tiny_imagenet_loaders(batch_size=128, data_dir='./datasets/tiny-imagenet-200',
                               imbalance_factor=100, num_workers=4):
    """
    Get Tiny ImageNet data loaders with optional long-tail imbalance.

    Returns: train_loader, val_loader, samples_per_class
    """
    num_classes = 200

    # Download if needed
    data_dir = download_tiny_imagenet(data_dir)

    # Transforms matching CSL paper (64×64 images)
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = TinyImageNetDataset(data_dir, split='train', transform=transform_train)
    val_dataset = TinyImageNetDataset(data_dir, split='val', transform=transform_test)

    # Create imbalanced version
    samples_per_class = None
    if imbalance_factor > 1:
        train_dataset, samples_per_class = _create_imbalanced(
            train_dataset, num_classes, imbalance_factor
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, samples_per_class


def _create_imbalanced(dataset, num_classes, imbalance_ratio):
    """Create long-tailed version using exponential decay (same as CIFAR-LT)."""
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label].append(idx)

    # Exponential decay: n_k = n_max * (ratio ^ (-k / (C-1)))
    img_max = max(len(indices) for indices in class_indices.values())
    img_num_per_cls = []
    for cls_idx in range(num_classes):
        num = img_max * (imbalance_ratio ** (-cls_idx / (num_classes - 1)))
        img_num_per_cls.append(int(num))

    selected_indices = []
    for cls_idx in range(num_classes):
        if cls_idx in class_indices:
            indices = class_indices[cls_idx]
            np.random.seed(42)
            np.random.shuffle(indices)
            take = min(img_num_per_cls[cls_idx], len(indices))
            selected_indices.extend(indices[:take])
            img_num_per_cls[cls_idx] = take

    return Subset(dataset, selected_indices), img_num_per_cls
