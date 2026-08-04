import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=128, num_workers=4, root='./data'):
    """
    Get CIFAR-10 train and validation dataloaders.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes
        root: Root directory for data storage
    
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    
    # CIFAR-10 specific transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download and load CIFAR-10
    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train
    )
    
    valset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_val
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader

def get_cifar10_class_names():
    """Get CIFAR-10 class names."""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

