import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import defaultdict

class CIFAR100WithLabels(datasets.CIFAR100):
    def get_unique_labels(self):
        return list(set(self.targets))

def calculate_class_distribution(num_classes, max_samples, mu):
    return [int(max_samples * (mu ** i)) for i in range(num_classes)]

def get_class_indices(dataset):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    return class_indices

def sample_class_indices(class_indices, target_counts):
    selected_indices = []
    for class_idx, count in enumerate(target_counts):
        if class_idx in class_indices:
            available = len(class_indices[class_idx])
            take = min(count, available)
            selected_indices.extend(np.random.choice(class_indices[class_idx], take, replace=False))
    return selected_indices

def get_cifar100_loaders(batch_size, data_dir='./datasets/CIFAR100', imbalance_factor=1.0, num_workers=4):
    num_classes = 100

    # Precompute transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load datasets
    train_set = CIFAR100WithLabels(root=data_dir, train=True, download=True, transform=transform_train)
    val_set = CIFAR100WithLabels(root=data_dir, train=False, download=True, transform=transform_test)

    # Apply imbalance only to training set
    if imbalance_factor != 1.0:
        train_class_indices = get_class_indices(train_set)
        train_max_samples = max(len(indices) for indices in train_class_indices.values())
        mu = (1 / imbalance_factor) ** (1 / (num_classes - 1))
        target_counts = calculate_class_distribution(num_classes, train_max_samples, mu)
        selected_indices = sample_class_indices(train_class_indices, target_counts)
        train_set = Subset(train_set, selected_indices)

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, None  # Returning val_loader as test_loader if needed
