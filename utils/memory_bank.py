import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class MemoryBank:
    """
    Memory bank that stores features per class using EMA + Reservoir sampling.
    Supports tail class identification for later semantic prompt generation.
    """
    
    def __init__(self, 
                 num_classes: int, 
                 feature_dim: int, 
                 capacity_per_class: int = 256,
                 alpha_base: float = 0.1,
                 tail_threshold_percentile: float = 20.0,
                 device: str = 'cpu'):
        """
        Initialize the memory bank.
        
        Args:
            num_classes: Number of classes in the dataset
            feature_dim: Dimension of feature vectors
            capacity_per_class: Maximum number of features to store per class (Reservoir)
            alpha_base: Base learning rate for EMA updates
            tail_threshold_percentile: Percentile below which classes are considered "tail"
            device: Device to store tensors on
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.capacity_per_class = capacity_per_class
        self.alpha_base = alpha_base
        self.tail_threshold_percentile = tail_threshold_percentile
        self.device = device
        
        # Initialize EMA prototypes for each class
        self.ema_prototypes = torch.zeros(num_classes, feature_dim, device=device)
        self.ema_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        
        # Initialize Reservoir buffers for each class
        self.reservoir_buffers = {i: [] for i in range(num_classes)}
        self.reservoir_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        
        # Class frequency tracking for adaptive alpha
        self.class_frequencies = torch.zeros(num_classes, dtype=torch.long, device=device)
        self.total_samples = 0
        
        # Tail class identification
        self.tail_classes = set()
        self.head_classes = set()
        self.medium_classes = set()
        
        # Statistics tracking
        self.update_history = defaultdict(list)
        
    def _normalize_feature(self, feature: torch.Tensor) -> torch.Tensor:
        """Normalize feature vector to unit L2 norm."""
        if isinstance(feature, np.ndarray):
            feature = torch.from_numpy(feature).to(self.device)
        
        # Ensure feature is 1D
        if feature.dim() > 1:
            feature = feature.flatten()
            
        # L2 normalize
        norm = torch.norm(feature, p=2)
        if norm > 0:
            feature = feature / norm
        else:
            # Handle zero features
            feature = torch.zeros_like(feature)
            
        return feature
    
    def _get_adaptive_alpha(self, class_id: int) -> float:
        """Get adaptive alpha based on class frequency."""
        if self.class_frequencies[class_id] == 0:
            return self.alpha_base
            
        # Calculate class frequency ratio
        freq_ratio = self.class_frequencies[class_id] / max(self.total_samples, 1)
        
        # Adaptive alpha: lower for head classes (more stable), higher for tail classes (faster adaptation)
        adaptive_alpha = self.alpha_base * (1.0 / (freq_ratio + 1e-6))
        
        # Clamp to reasonable bounds
        return torch.clamp(adaptive_alpha, 0.01, 0.5).item()
    
    def update(self, class_id: int, feature: torch.Tensor) -> None:
        """
        Update memory bank with a new feature for the given class.
        
        Args:
            class_id: Class ID (0 to num_classes-1)
            feature: Feature vector to store
        """
        if not (0 <= class_id < self.num_classes):
            return
            
        # Normalize feature
        feature = self._normalize_feature(feature)
        
        # Update class frequency
        self.class_frequencies[class_id] += 1
        self.total_samples += 1
        
        # Update EMA prototype
        alpha = self._get_adaptive_alpha(class_id)
        if self.ema_counts[class_id] == 0:
            # First feature for this class
            self.ema_prototypes[class_id] = feature
        else:
            # EMA update: M_c ← (1 - α) * M_c + α * feature
            self.ema_prototypes[class_id] = (1 - alpha) * self.ema_prototypes[class_id] + alpha * feature
        
        self.ema_counts[class_id] += 1
        
        # Update Reservoir buffer
        self._update_reservoir(class_id, feature)
        
        # Update tail class identification
        self._update_tail_classification()
        
        # Track update history for analysis
        self.update_history[class_id].append({
            'step': self.total_samples,
            'alpha': alpha,
            'feature_norm': torch.norm(feature).item()
        })
    
    def _update_reservoir(self, class_id: int, feature: torch.Tensor) -> None:
        """Update Reservoir buffer for a class using Reservoir sampling."""
        self.reservoir_counts[class_id] += 1
        n = self.reservoir_counts[class_id]
        
        if len(self.reservoir_buffers[class_id]) < self.capacity_per_class:
            # Buffer not full yet, just append
            self.reservoir_buffers[class_id].append(feature.detach().cpu())
        else:
            # Reservoir sampling: replace with probability K/n
            if torch.rand(1).item() < self.capacity_per_class / n:
                # Randomly select position to replace
                replace_idx = torch.randint(0, self.capacity_per_class, (1,)).item()
                self.reservoir_buffers[class_id][replace_idx] = feature.detach().cpu()
    
    def _update_tail_classification(self) -> None:
        """Update tail/head/medium class classification based on current frequencies."""
        if self.total_samples == 0:
            return
            
        # Calculate class frequencies as percentages
        class_percentages = (self.class_frequencies / self.total_samples * 100).cpu().numpy()
        
        # Sort classes by frequency
        sorted_indices = np.argsort(class_percentages)[::-1]  # Descending order
        
        # Determine thresholds
        head_threshold = np.percentile(class_percentages, 100 - self.tail_threshold_percentile)
        tail_threshold = np.percentile(class_percentages, self.tail_threshold_percentile)
        
        # Classify classes
        self.head_classes = set(sorted_indices[class_percentages[sorted_indices] >= head_threshold])
        self.tail_classes = set(sorted_indices[class_percentages[sorted_indices] <= tail_threshold])
        self.medium_classes = set(range(self.num_classes)) - self.head_classes - self.tail_classes
    
    def get_prototype(self, class_id: int) -> torch.Tensor:
        """Get EMA prototype for a class."""
        if not (0 <= class_id < self.num_classes):
            return torch.zeros(self.feature_dim, device=self.device)
        return self.ema_prototypes[class_id].clone()
    
    def get_prototypes(self) -> torch.Tensor:
        """Get all EMA prototypes."""
        return self.ema_prototypes.clone()
    
    def get_class_exemplars(self, class_id: int, num_exemplars: int = 5) -> torch.Tensor:
        """
        Get k exemplar features for a class.
    
        Args:
        class_id: Class ID
        k: Number of exemplars to return
    
        Returns:
        exemplars: Tensor of shape (k, feature_dim) or fewer if not enough samples
        """
        if not (0 <= class_id < self.num_classes):
            return torch.empty(0, self.feature_dim, device=self.device)
    
     # Get features from reservoir buffer
        buffer = self.reservoir_buffers.get(class_id, [])
    
        if not buffer:
        # No samples in reservoir, return prototype repeated k times
            prototype = self.ema_prototypes[class_id]
            return prototype.unsqueeze(0).repeat(num_exemplars, 1)
    
    # Sample k features from buffer (or all if fewer than k)
        n_samples = min(num_exemplars, len(buffer))
        indices = np.random.choice(len(buffer), n_samples, replace=False)
    
        exemplars = torch.stack([buffer[i].to(self.device) for i in indices])
        return exemplars
    
    
    def sample_features(self, class_id: int, k: Optional[int] = None) -> List[torch.Tensor]:
        """Sample features from Reservoir buffer for a class."""
        if not (0 <= class_id < self.num_classes):
            return []
            
        buffer = self.reservoir_buffers[class_id]
        if k is None:
            return buffer.copy()
        
        # Sample k features (or all if buffer is smaller)
        k = min(k, len(buffer))
        if k == 0:
            return []
            
        indices = torch.randperm(len(buffer))[:k]
        return [buffer[i] for i in indices]
    
    def get_class_statistics(self, class_id: int) -> Dict:
        """Get comprehensive statistics for a class."""
        if not (0 <= class_id < self.num_classes):
            return {}
            
        buffer = self.reservoir_buffers[class_id]
        prototype = self.ema_prototypes[class_id]
        
        stats = {
            'class_id': class_id,
            'ema_count': self.ema_counts[class_id].item(),
            'reservoir_count': len(buffer),
            'total_samples': self.reservoir_counts[class_id].item(),
            'class_frequency': self.class_frequencies[class_id].item(),
            'class_percentage': (self.class_frequencies[class_id] / max(self.total_samples, 1) * 100).item(),
            'is_tail': class_id in self.tail_classes,
            'is_head': class_id in self.head_classes,
            'is_medium': class_id in self.medium_classes,
            'prototype_norm': torch.norm(prototype).item(),
        }
        
        if buffer:
            buffer_tensor = torch.stack(buffer)
            stats.update({
                'buffer_mean_norm': torch.mean(torch.norm(buffer_tensor, dim=1)).item(),
                'buffer_std_norm': torch.std(torch.norm(buffer_tensor, dim=1)).item(),
                'prototype_buffer_similarity': torch.mean(
                    torch.cosine_similarity(prototype.unsqueeze(0), buffer_tensor, dim=1)
                ).item()
            })
        
        return stats
    
    def get_tail_classes(self) -> List[int]:
        """Get list of tail class IDs."""
        return list(self.tail_classes)
    
    def get_head_classes(self) -> List[int]:
        """Get list of head class IDs."""
        return list(self.head_classes)
    
    def get_medium_classes(self) -> List[int]:
        """Get list of medium class IDs."""
        return list(self.medium_classes)
    
    def get_class_distribution(self) -> Dict[str, List[int]]:
        """Get distribution of classes across tail/head/medium categories."""
        return {
            'tail': list(self.tail_classes),
            'medium': list(self.medium_classes),
            'head': list(self.head_classes)
        }
    
    def get_tail_class_features(self, k_per_class: int = 10) -> Dict[int, List[torch.Tensor]]:
        """Get features from tail classes for semantic prompt generation."""
        tail_features = {}
        for class_id in self.tail_classes:
            features = self.sample_features(class_id, k_per_class)
            if features:
                tail_features[class_id] = features
        return tail_features
    
    def get_tail_class_prototypes(self) -> Dict[int, torch.Tensor]:
        """Get prototypes for tail classes."""
        return {class_id: self.get_prototype(class_id) for class_id in self.tail_classes}
    
    def compute_class_similarity_matrix(self) -> torch.Tensor:
        """Compute cosine similarity matrix between all class prototypes."""
        prototypes = self.get_prototypes()  # [num_classes, feature_dim]
        # Normalize prototypes
        prototypes_norm = prototypes / (torch.norm(prototypes, dim=1, keepdim=True) + 1e-8)
        # Compute similarity matrix
        similarity_matrix = torch.mm(prototypes_norm, prototypes_norm.t())
        return similarity_matrix
    
    def visualize_class_distribution(self, save_path: Optional[str] = None) -> None:
        """Visualize class distribution and memory bank statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class frequency distribution
        class_percentages = (self.class_frequencies / max(self.total_samples, 1) * 100).cpu().numpy()
        axes[0, 0].hist(class_percentages, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.percentile(class_percentages, self.tail_threshold_percentile), 
                           color='red', linestyle='--', label=f'{self.tail_threshold_percentile}th percentile')
        axes[0, 0].axvline(np.percentile(class_percentages, 100 - self.tail_threshold_percentile), 
                           color='green', linestyle='--', label=f'{100 - self.tail_threshold_percentile}th percentile')
        axes[0, 0].set_xlabel('Class Frequency (%)')
        axes[0, 0].set_ylabel('Number of Classes')
        axes[0, 0].set_title('Class Frequency Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Tail/Head/Medium class counts
        distribution = self.get_class_distribution()
        categories = ['Tail', 'Medium', 'Head']
        counts = [len(distribution['tail']), len(distribution['medium']), len(distribution['head'])]
        colors = ['red', 'orange', 'green']
        axes[0, 1].bar(categories, counts, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Number of Classes')
        axes[0, 1].set_title('Class Distribution by Frequency')
        for i, count in enumerate(counts):
            axes[0, 1].text(i, count + 0.5, str(count), ha='center', va='bottom')
        
        # Memory buffer utilization
        buffer_utilization = [len(self.reservoir_buffers[i]) / self.capacity_per_class * 100 
                             for i in range(self.num_classes)]
        axes[1, 0].hist(buffer_utilization, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].set_xlabel('Buffer Utilization (%)')
        axes[1, 0].set_ylabel('Number of Classes')
        axes[1, 0].set_title('Reservoir Buffer Utilization')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Prototype norms
        prototype_norms = torch.norm(self.ema_prototypes, dim=1).cpu().numpy()
        axes[1, 1].hist(prototype_norms, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].set_xlabel('Prototype L2 Norm')
        axes[1, 1].set_ylabel('Number of Classes')
        axes[1, 1].set_title('EMA Prototype Norms')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save(self, filepath: str) -> None:
        """Save memory bank to disk."""
        def convert_numpy_types(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            else:
                return obj
        
        save_data = {
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'capacity_per_class': self.capacity_per_class,
            'alpha_base': self.alpha_base,
            'tail_threshold_percentile': self.tail_threshold_percentile,
            'ema_prototypes': self.ema_prototypes.cpu().numpy().tolist(),
            'ema_counts': self.ema_counts.cpu().numpy().tolist(),
            'reservoir_counts': self.reservoir_counts.cpu().numpy().tolist(),
            'class_frequencies': self.class_frequencies.cpu().numpy().tolist(),
            'total_samples': self.total_samples,
            'tail_classes': list(self.tail_classes),
            'head_classes': list(self.head_classes),
            'medium_classes': list(self.medium_classes),
            'update_history': dict(self.update_history)
        }
        
        # Convert reservoir buffers to lists for JSON serialization
        save_data['reservoir_buffers'] = {}
        for class_id, buffer in self.reservoir_buffers.items():
            if buffer:
                save_data['reservoir_buffers'][str(class_id)] = [f.cpu().numpy().tolist() for f in buffer]
            else:
                save_data['reservoir_buffers'][str(class_id)] = []
        
        # Convert all numpy types to Python native types
        save_data = convert_numpy_types(save_data)
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load memory bank from disk."""
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        # Restore basic parameters
        self.num_classes = save_data['num_classes']
        self.feature_dim = save_data['feature_dim']
        self.capacity_per_class = save_data['capacity_per_class']
        self.alpha_base = save_data['alpha_base']
        self.tail_threshold_percentile = save_data['tail_threshold_percentile']
        
        # Convert lists back to numpy arrays and then to tensors
        self.ema_prototypes = torch.from_numpy(np.array(save_data['ema_prototypes'])).to(self.device)
        self.ema_counts = torch.from_numpy(np.array(save_data['ema_counts'])).to(self.device)
        self.reservoir_counts = torch.from_numpy(np.array(save_data['reservoir_counts'])).to(self.device)
        self.class_frequencies = torch.from_numpy(np.array(save_data['class_frequencies'])).to(self.device)
        self.total_samples = save_data['total_samples']
        
        # Restore sets
        self.tail_classes = set(save_data['tail_classes'])
        self.head_classes = set(save_data['head_classes'])
        self.medium_classes = set(save_data['medium_classes'])
        
        # Restore update history
        self.update_history = defaultdict(list)
        for class_id, history in save_data['update_history'].items():
            self.update_history[int(class_id)] = history
        
        # Restore reservoir buffers
        self.reservoir_buffers = {i: [] for i in range(self.num_classes)}
        for class_id_str, buffer_list in save_data['reservoir_buffers'].items():
            class_id = int(class_id_str)
            if len(buffer_list) > 0:
                self.reservoir_buffers[class_id] = [torch.from_numpy(np.array(f)) for f in buffer_list]
            else:
                self.reservoir_buffers[class_id] = []
    
    def get_memory_usage(self) -> Dict:
        """Get memory usage statistics."""
        total_features = sum(len(buffer) for buffer in self.reservoir_buffers.values())
        
        # Calculate memory usage for reservoir buffers
        reservoir_memory = 0
        for buffer in self.reservoir_buffers.values():
            if buffer:
                # Each feature in buffer is a tensor
                for feature in buffer:
                    reservoir_memory += feature.numel() * 4  # float32 = 4 bytes
        
        total_memory_bytes = (
            self.ema_prototypes.numel() * 4 +  # float32
            reservoir_memory  # float32 features from reservoir
        )
        
        return {
            'total_features_stored': total_features,
            'total_memory_mb': total_memory_bytes / (1024 * 1024),
            'buffer_utilization': total_features / (self.num_classes * self.capacity_per_class),
            'classes_with_data': sum(1 for buffer in self.reservoir_buffers.values() if buffer),
            'empty_classes': sum(1 for buffer in self.reservoir_buffers.values() if not buffer)
        }
