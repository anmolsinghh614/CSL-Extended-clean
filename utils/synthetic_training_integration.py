import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import numpy as np
import json
from pathlib import Path


class SyntheticImageDataset(Dataset):
    """
    Dataset class for synthetic tail class images.
    """
    
    def __init__(self, 
                 dataset_file: str, 
                 transform: Optional[transforms.Compose] = None,
                 confidence_scores: Optional[Dict[str, float]] = None):
        """
        Initialize synthetic image dataset.
        
        Args:
            dataset_file: Path to dataset file (image_path class_id format)
            transform: Image transformations
            confidence_scores: Optional confidence scores for synthetic images
        """
        self.dataset_file = dataset_file
        self.transform = transform or self._default_transform()
        self.confidence_scores = confidence_scores or {}
        
        # Load dataset entries
        self.samples = []
        with open(dataset_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        image_path = ' '.join(parts[:-1])  # Handle paths with spaces
                        class_id = int(parts[-1])
                        self.samples.append((image_path, class_id))
        
        print(f"Loaded {len(self.samples)} synthetic samples from {dataset_file}")
    
    def _default_transform(self):
        """Default image transformations for synthetic images (CIFAR-10)."""
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                               std=[0.2023, 0.1994, 0.2010])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, class_id = self.samples[idx]
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get confidence score if available
        confidence = self.confidence_scores.get(image_path, 1.0)
        
        return image, class_id, confidence


class ConfidenceAdaptiveCSL(nn.Module):
    """
    Enhanced CSL loss that incorporates confidence weighting for synthetic features.
    Extends the original CSL to handle synthetic image integration.
    """
    
    def __init__(self, 
                 target_class_index: List[int], 
                 num_classes: int,
                 synthetic_weight: float = 0.5,
                 confidence_threshold: float = 0.3):
        """
        Initialize confidence-adaptive CSL.
        
        Args:
            target_class_index: List of target class indices (typically tail classes)
            num_classes: Total number of classes
            synthetic_weight: Weight for synthetic samples vs real samples
            confidence_threshold: Minimum confidence to include synthetic sample
        """
        super().__init__()
        
        from utils.csl_loss import CSLLossFunc  # Import your existing CSL
        self.base_csl = CSLLossFunc(target_class_index, num_classes)
        
        self.target_class_index = target_class_index
        self.num_classes = num_classes
        self.synthetic_weight = synthetic_weight
        self.confidence_threshold = confidence_threshold
        
        # Track synthetic vs real sample statistics
        self.synthetic_stats = {
            'used_samples': 0,
            'total_synthetic': 0,
            'confidence_filtered': 0
        }
    
    def forward(self, 
                y_true: torch.Tensor, 
                y_pred: torch.Tensor, 
                epoch: int,
                sample_confidences: Optional[torch.Tensor] = None,
                is_synthetic: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with confidence-adaptive weighting.
        
        Args:
            y_true: True labels [batch_size]
            y_pred: Predicted logits [batch_size, num_classes]
            epoch: Current epoch number
            sample_confidences: Confidence scores [batch_size] (1.0 for real, <1.0 for synthetic)
            is_synthetic: Boolean mask [batch_size] indicating synthetic samples
            
        Returns:
            Weighted CSL loss
        """
        # Base CSL loss
        base_loss = self.base_csl(y_true, y_pred, epoch)
        
        # If no synthetic samples, return base loss
        if sample_confidences is None or is_synthetic is None:
            return base_loss
        
        # Separate synthetic and real samples
        synthetic_mask = is_synthetic.bool()
        real_mask = ~synthetic_mask
        
        # Update statistics
        self.synthetic_stats['total_synthetic'] += synthetic_mask.sum().item()
        
        # Apply confidence filtering for synthetic samples
        if synthetic_mask.any():
            # Get synthetic sample confidences
            synthetic_confidences = sample_confidences[synthetic_mask]
            
            # Filter low-confidence synthetic samples
            high_conf_synthetic = synthetic_confidences >= self.confidence_threshold
            filtered_synthetic_mask = synthetic_mask.clone()
            filtered_synthetic_mask[synthetic_mask] = high_conf_synthetic
            
            # Update statistics
            self.synthetic_stats['confidence_filtered'] += (~high_conf_synthetic).sum().item()
            self.synthetic_stats['used_samples'] += high_conf_synthetic.sum().item()
            
            if filtered_synthetic_mask.any():
                # Compute weighted loss for high-confidence synthetic samples
                synthetic_loss = self._compute_synthetic_loss(
                    y_true[filtered_synthetic_mask], 
                    y_pred[filtered_synthetic_mask],
                    sample_confidences[filtered_synthetic_mask]
                )
                
                # Combine losses with weighting
                if real_mask.any():
                    real_loss = self.base_csl(y_true[real_mask], y_pred[real_mask], epoch)
                    total_loss = (real_loss + self.synthetic_weight * synthetic_loss) / (1 + self.synthetic_weight)
                else:
                    total_loss = synthetic_loss
                
                return total_loss
        
        return base_loss
    
    def _compute_synthetic_loss(self, 
                              synthetic_labels: torch.Tensor,
                              synthetic_preds: torch.Tensor, 
                              confidences: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence-weighted loss for synthetic samples.
        
        Args:
            synthetic_labels: Labels for synthetic samples
            synthetic_preds: Predictions for synthetic samples  
            confidences: Confidence scores for synthetic samples
            
        Returns:
            Confidence-weighted synthetic loss
        """
        # Base cross-entropy loss
        ce_loss = nn.functional.cross_entropy(synthetic_preds, synthetic_labels, reduction='none')
        
        # Weight by confidence scores
        weighted_loss = ce_loss * confidences
        
        return weighted_loss.mean()
    
    def get_synthetic_stats(self) -> Dict:
        """Get statistics about synthetic sample usage."""
        total = self.synthetic_stats['total_synthetic']
        if total > 0:
            usage_rate = self.synthetic_stats['used_samples'] / total
            filter_rate = self.synthetic_stats['confidence_filtered'] / total
        else:
            usage_rate = 0.0
            filter_rate = 0.0
            
        return {
            **self.synthetic_stats,
            'usage_rate': usage_rate,
            'filter_rate': filter_rate
        }


class HybridDataLoader:
    """
    Data loader that combines real and synthetic data with proper balancing.
    """
    
    def __init__(self, 
                 real_loader: DataLoader,
                 synthetic_dataset: SyntheticImageDataset,
                 synthetic_ratio: float = 0.3,
                 batch_size: int = 256):
        """
        Initialize hybrid data loader.
        
        Args:
            real_loader: DataLoader for real images
            synthetic_dataset: Dataset containing synthetic images
            synthetic_ratio: Ratio of synthetic to real samples per batch
            batch_size: Total batch size
        """
        self.real_loader = real_loader
        self.synthetic_dataset = synthetic_dataset
        self.synthetic_ratio = synthetic_ratio
        self.batch_size = batch_size
        
        # Calculate split
        self.synthetic_batch_size = int(batch_size * synthetic_ratio)
        self.real_batch_size = batch_size - self.synthetic_batch_size
        
        # Create synthetic loader
        self.synthetic_loader = DataLoader(
            synthetic_dataset, 
            batch_size=self.synthetic_batch_size,
            shuffle=True,
            drop_last=True
        )
        
        print(f"Hybrid loader: {self.real_batch_size} real + {self.synthetic_batch_size} synthetic per batch")
    
    def __iter__(self):
        """Iterate through combined batches of real and synthetic data."""
        real_iter = iter(self.real_loader)
        synthetic_iter = iter(self.synthetic_loader)
        
        while True:
            try:
                # Get real batch
                real_batch = next(real_iter)
                real_images, real_labels = real_batch[:2]  # Handle different return formats
                
                # Truncate real batch to desired size
                if len(real_images) > self.real_batch_size:
                    real_images = real_images[:self.real_batch_size]
                    real_labels = real_labels[:self.real_batch_size]
                
                # Get synthetic batch
                try:
                    synthetic_images, synthetic_labels, synthetic_confidences = next(synthetic_iter)
                except StopIteration:
                    # Restart synthetic iterator
                    synthetic_iter = iter(self.synthetic_loader)
                    synthetic_images, synthetic_labels, synthetic_confidences = next(synthetic_iter)
                
                # Combine batches
                combined_images = torch.cat([real_images, synthetic_images], dim=0)
                combined_labels = torch.cat([real_labels, synthetic_labels], dim=0)
                
                # Create confidence and synthetic masks
                real_confidences = torch.ones(len(real_images))
                combined_confidences = torch.cat([real_confidences, synthetic_confidences], dim=0)
                
                is_synthetic = torch.cat([
                    torch.zeros(len(real_images), dtype=torch.bool),
                    torch.ones(len(synthetic_images), dtype=torch.bool)
                ], dim=0)
                
                yield combined_images, combined_labels, combined_confidences, is_synthetic
                
            except StopIteration:
                break
    
    def __len__(self):
        return len(self.real_loader)


def integrate_synthetic_training(original_main_function,
                               memory_manager,
                               feature_to_text_mapper,
                               class_names: Dict[int, str],
                               synthetic_dataset_path: Optional[str] = None,
                               synthetic_ratio: float = 0.2) -> None:
    """
    Modify existing training to include synthetic tail class images.
    
    Args:
        original_main_function: Your existing main training function
        memory_manager: Initialized MemoryManager
        feature_to_text_mapper: Initialized FeatureToTextMapper
        class_names: Dictionary mapping class_id to class_name
        synthetic_dataset_path: Path to synthetic dataset file (if None, generate new)
        synthetic_ratio: Ratio of synthetic samples in each batch
    """
    
    # Generate synthetic dataset if not provided
    if synthetic_dataset_path is None:
        print("Generating synthetic tail class dataset...")
        from image_diffusion_pipeline import main_generation_pipeline
        synthetic_dataset_path, _ = main_generation_pipeline(
            memory_manager=memory_manager,
            feature_to_text_mapper=feature_to_text_mapper,
            class_names=class_names,
            output_dir="./synthetic_tail_dataset"
        )
    
    # Create synthetic dataset
    synthetic_dataset = SyntheticImageDataset(synthetic_dataset_path)
    
    print(f"Integrating {len(synthetic_dataset)} synthetic samples into training")
    print(f"Synthetic ratio: {synthetic_ratio}")
    
    # The integration would modify your main.py training loop
    # This is a template for how to modify your existing training
    
    return synthetic_dataset


def modified_train_function(model, criterion, optimizer, scheduler, 
                          real_train_loader, val_loader, device, epoch, 
                          memory_manager=None, synthetic_dataset=None,
                          synthetic_ratio=0.2):
    """
    Modified training function that incorporates synthetic images.
    This replaces your existing train() function in main.py.
    """
    model.train()
    
    # Create hybrid data loader if synthetic dataset is provided
    if synthetic_dataset is not None:
        train_loader = HybridDataLoader(
            real_loader=real_train_loader,
            synthetic_dataset=synthetic_dataset,
            synthetic_ratio=synthetic_ratio,
            batch_size=real_train_loader.batch_size
        )
        print(f"Using hybrid training with {synthetic_ratio:.1%} synthetic samples")
    else:
        train_loader = real_train_loader
    
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        if len(batch_data) == 4:  # Hybrid batch with synthetic data
            inputs, labels, confidences, is_synthetic = batch_data
            inputs, labels = inputs.to(device), labels.to(device)
            confidences, is_synthetic = confidences.to(device), is_synthetic.to(device)
        else:  # Regular batch
            inputs, labels = batch_data[:2]
            inputs, labels = inputs.to(device), labels.to(device)
            confidences, is_synthetic = None, None
        
        optimizer.zero_grad()
        
        # Forward pass with feature extraction
        if memory_manager is not None:
            # Only update memory with real samples
            real_mask = ~is_synthetic if is_synthetic is not None else torch.ones(len(labels), dtype=torch.bool)
            if real_mask.any():
                real_inputs = inputs[real_mask]
                real_labels = labels[real_mask]
                outputs, features = model(inputs, return_features=True)
                # Update memory bank with real samples only
                memory_manager.update_memory(real_inputs, real_labels)
            else:
                outputs = model(inputs)
        else:
            outputs = model(inputs)
        
        # Compute loss with confidence weighting
        if isinstance(criterion, ConfidenceAdaptiveCSL):
            loss = criterion(labels, outputs, epoch, confidences, is_synthetic)
        else:
            loss = criterion(labels, outputs, epoch)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print statistics every 100 batches
        if batch_idx % 100 == 0:
            if isinstance(criterion, ConfidenceAdaptiveCSL):
                synthetic_stats = criterion.get_synthetic_stats()
                print(f"Batch {batch_idx}: Synthetic usage rate: {synthetic_stats['usage_rate']:.2%}")
            if memory_manager is not None:
                print(f"Memory updates: {memory_manager.update_stats['total_updates']}")
    
    train_loss /= len(train_loader.dataset) if hasattr(train_loader, 'dataset') else total
    train_accuracy = 100. * correct / total
    
    # Validation remains the same
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(labels, outputs, epoch) 
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / total
    scheduler.step()
    
    return train_loss, train_accuracy, val_loss, val_accuracy