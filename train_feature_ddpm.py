"""
Train Feature-Space DDPM Model
Trains on features extracted from ResNet32 memory bank
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import json
import numpy as np
from tqdm import tqdm
import argparse

from phase3_feature_ddpm import FeatureDDPM
from models import ResNet32, ResNet50
from torchvision import datasets, transforms


class FeatureDataset(Dataset):
    """Dataset of features extracted from images"""
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def extract_features_from_dataset(model, dataloader, device, num_classes=10):
    """
    Extract features from all images in dataset
    
    Args:
        model: Trained ResNet model
        dataloader: DataLoader with images
        device: cuda/cpu
        num_classes: Number of classes
    
    Returns:
        features_dict: Dict[class_id -> features tensor]
    """
    model.eval()
    
    features_dict = {i: [] for i in range(num_classes)}
    
    print("\nExtracting features from dataset...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting"):
            images = images.to(device)
            
            # Extract features
            _, features = model(images, return_features=True)
            features = features.cpu()
            
            # Group by class
            for feat, label in zip(features, labels):
                features_dict[label.item()].append(feat)
    
    # Convert lists to tensors
    for class_id in features_dict:
        if len(features_dict[class_id]) > 0:
            features_dict[class_id] = torch.stack(features_dict[class_id])
            print(f"Class {class_id}: {len(features_dict[class_id])} features")
        else:
            features_dict[class_id] = torch.empty(0, 512)
    
    return features_dict


def load_features_from_memory_bank(memory_bank_path, device='cpu'):
    """
    Load features from saved memory bank checkpoint
    
    Args:
        memory_bank_path: Path to memory bank .pth file
        device: Device to load to
    
    Returns:
        features_dict: Dict[class_id -> features tensor]
    """
    print(f"\nLoading features from memory bank: {memory_bank_path}")
    
    checkpoint = torch.load(memory_bank_path, map_location=device)
    
    # Extract features from memory bank
    if 'memory' in checkpoint:
        memory = checkpoint['memory']
        features_dict = {}
        
        for class_id, class_memory in memory.items():
            if 'features' in class_memory and len(class_memory['features']) > 0:
                features_dict[class_id] = class_memory['features']
                print(f"Class {class_id}: {len(features_dict[class_id])} features")
            else:
                features_dict[class_id] = torch.empty(0, 512)
        
        return features_dict
    else:
        raise ValueError("Invalid memory bank format")


def create_training_dataset(features_dict, tail_classes=[6, 7, 8, 9], 
                           augment_factor=5):
    """
    Create training dataset from features
    
    Args:
        features_dict: Dict[class_id -> features tensor]
        tail_classes: Which classes to train on
        augment_factor: How many times to repeat small classes
    
    Returns:
        train_dataset: Dataset for training
    """
    all_features = []
    all_labels = []
    
    for class_id in tail_classes:
        if class_id not in features_dict or len(features_dict[class_id]) == 0:
            print(f"Warning: No features for class {class_id}, skipping")
            continue
        
        features = features_dict[class_id]
        
        # Augment small classes by repeating
        if len(features) < 100:
            repeat = min(augment_factor, 100 // len(features) + 1)
            features = features.repeat(repeat, 1)
            print(f"Class {class_id}: Augmented {len(features_dict[class_id])} -> {len(features)} samples")
        
        all_features.append(features)
        all_labels.extend([class_id] * len(features))
    
    if len(all_features) == 0:
        raise ValueError("No features available for training!")
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    
    return FeatureDataset(all_features, all_labels)


def train_feature_ddpm(model, train_loader, optimizer, device, epoch):
    """Single training epoch"""
    model.train()
    
    epoch_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for features, labels in pbar:
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        loss = model(features, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = epoch_loss / num_batches
    return avg_loss


def validate_generation(model, features_dict, tail_classes, device):
    """
    Validate by generating features and comparing to real
    
    Returns quality metrics
    """
    model.eval()
    
    # Compute class prototypes
    class_prototypes = {}
    for class_id in tail_classes:
        if class_id in features_dict and len(features_dict[class_id]) > 0:
            class_prototypes[class_id] = features_dict[class_id].mean(dim=0)
    
    if len(class_prototypes) == 0:
        return {}
    
    # Generate synthetic features
    num_samples = 10
    class_ids = torch.tensor(tail_classes * num_samples, device=device)
    
    synthetic_features, confidences = model.sample_with_confidence(
        class_ids, class_prototypes, device
    )
    
    # Compute metrics
    metrics = {
        'mean_confidence': confidences.mean().item(),
        'std_confidence': confidences.std().item(),
        'min_confidence': confidences.min().item(),
        'max_confidence': confidences.max().item()
    }
    
    return metrics


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ========== Load Features ==========
    if args.memory_bank_path:
        # Option 1: Load from memory bank
        features_dict = load_features_from_memory_bank(args.memory_bank_path, device)
    else:
        # Option 2: Extract from dataset
        print("\nLoading model and extracting features...")
        
        # Load trained model
        if args.model_name == 'resnet32':
            model = ResNet32(num_classes=10).to(device)
        elif args.model_name == 'resnet50':
            model = ResNet50(num_classes=10).to(device)
        else:
            raise ValueError(f"Unknown model: {args.model_name}")
        
        checkpoint = torch.load(args.model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {args.model_checkpoint}")
        
        # Load dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                         download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=128, 
                                 shuffle=False, num_workers=2)
        
        # Extract features
        features_dict = extract_features_from_dataset(model, train_loader, device)
    
    # ========== Create Training Dataset ==========
    train_dataset = create_training_dataset(
        features_dict, 
        tail_classes=args.tail_classes,
        augment_factor=args.augment_factor
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"\nTraining dataset: {len(train_dataset)} samples")
    
    # ========== Initialize DDPM ==========
    ddpm_model = FeatureDDPM(
        feature_dim=args.feature_dim,
        num_classes=10,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule
    ).to(device)
    
    optimizer = optim.AdamW(
        ddpm_model.parameters(), 
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs
    )
    
    # ========== Training Loop ==========
    print("\n" + "="*70)
    print("TRAINING FEATURE-SPACE DDPM")
    print("="*70)
    
    best_loss = float('inf')
    train_losses = []
    
    for epoch in range(args.num_epochs):
        # Train
        train_loss = train_feature_ddpm(ddpm_model, train_loader, optimizer, device, epoch + 1)
        train_losses.append(train_loss)
        
        # Validate
        if (epoch + 1) % 10 == 0:
            metrics = validate_generation(ddpm_model, features_dict, args.tail_classes, device)
            
            print(f"\nEpoch {epoch+1}/{args.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if metrics:
                print(f"  Mean Confidence: {metrics['mean_confidence']:.4f}")
                print(f"  Confidence Range: [{metrics['min_confidence']:.4f}, {metrics['max_confidence']:.4f}]")
        
        scheduler.step()
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, 'feature_ddpm_best.pth')
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ddpm_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': {
                    'feature_dim': args.feature_dim,
                    'num_classes': 10,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_timesteps': args.num_timesteps,
                    'beta_schedule': args.beta_schedule
                }
            }, checkpoint_path)
            
            print(f"  ✓ Best model saved: {checkpoint_path}")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'feature_ddpm_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ddpm_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    # ========== Final Save ==========
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    final_path = os.path.join(args.output_dir, 'feature_ddpm_final.pth')
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': ddpm_model.state_dict(),
        'train_losses': train_losses,
        'best_loss': best_loss,
        'config': {
            'feature_dim': args.feature_dim,
            'num_classes': 10,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'num_timesteps': args.num_timesteps,
            'beta_schedule': args.beta_schedule
        }
    }, final_path)
    
    print(f"\n✓ Final model saved: {final_path}")
    print(f"✓ Best loss: {best_loss:.4f}")
    
    # Save training summary
    summary = {
        'num_epochs': args.num_epochs,
        'best_loss': best_loss,
        'final_loss': train_losses[-1],
        'tail_classes': args.tail_classes,
        'num_training_samples': len(train_dataset),
        'config': {
            'feature_dim': args.feature_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'num_timesteps': args.num_timesteps,
            'beta_schedule': args.beta_schedule,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate
        }
    }
    
    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Training summary: {summary_path}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Feature-Space DDPM')
    
    # Data arguments
    parser.add_argument('--memory_bank_path', type=str, default=None,
                       help='Path to memory bank checkpoint')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                       help='Path to trained ResNet (if not using memory bank)')
    parser.add_argument('--model_name', type=str, default='resnet32',
                       choices=['resnet32', 'resnet50'])
    parser.add_argument('--tail_classes', type=int, nargs='+', default=[6, 7, 8, 9],
                       help='Tail classes to train on')
    parser.add_argument('--augment_factor', type=int, default=5,
                       help='Repeat factor for small classes')
    
    # Model arguments
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='cosine',
                       choices=['linear', 'cosine'])
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='./feature_ddpm_checkpoints')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.memory_bank_path is None and args.model_checkpoint is None:
        raise ValueError("Must provide either --memory_bank_path or --model_checkpoint")
    
    main(args)