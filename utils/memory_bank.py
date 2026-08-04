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
                 device: str = 'cpu',
                 tail_refresh_interval: int = 512,
                 history_limit: int = 200):
        """
        Initialize the memory bank.
        
        Args:
            num_classes: Number of classes in the dataset
            feature_dim: Dimension of feature vectors
            capacity_per_class: Maximum number of features to store per class (Reservoir)
            alpha_base: Base learning rate for EMA updates
            tail_threshold_percentile: Percentile below which classes are considered "tail"
            device: Device to store tensors on
            tail_refresh_interval: Number of samples between tail/head reclassifications.
                Reclassification is a percentile computation over all classes, so doing it
                once per sample dominates training time on large datasets.
            history_limit: Maximum number of history entries retained per class. Without a
                cap this grows to one dict per observed sample, which makes long runs leak
                memory and produces unusably large checkpoints.
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.capacity_per_class = capacity_per_class
        self.alpha_base = alpha_base
        self.tail_threshold_percentile = tail_threshold_percentile
        self.device = device
        self.tail_refresh_interval = max(1, tail_refresh_interval)
        self.history_limit = max(0, history_limit)
        
        # Initialize EMA prototypes for each class
        self.ema_prototypes = torch.zeros(num_classes, feature_dim, device=device)
        
        # Counters stay on CPU: they are only ever read as Python scalars, and keeping them
        # on the accelerator forces a host sync on every training batch.
        self.ema_counts = torch.zeros(num_classes, dtype=torch.long)
        
        # Initialize Reservoir buffers for each class
        self.reservoir_buffers = {i: [] for i in range(num_classes)}
        self.reservoir_counts = torch.zeros(num_classes, dtype=torch.long)
        
        # Class frequency tracking for adaptive alpha
        self.class_frequencies = torch.zeros(num_classes, dtype=torch.long)
        self.total_samples = 0
        self._samples_since_refresh = 0
        
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
        frequency = int(self.class_frequencies[class_id])
        if frequency == 0:
            return self.alpha_base
            
        # Calculate class frequency ratio
        freq_ratio = frequency / max(self.total_samples, 1)
        
        # Adaptive alpha: lower for head classes (more stable), higher for tail classes (faster adaptation)
        adaptive_alpha = self.alpha_base * (1.0 / (freq_ratio + 1e-6))
        
        # Clamp to reasonable bounds
        return float(np.clip(adaptive_alpha, 0.01, 0.5))
    
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
        self._apply_ema(class_id, feature, alpha)
        
        # Update Reservoir buffer
        self._update_reservoir(class_id, feature.detach().cpu())
        
        self._record_history(class_id, alpha, float(torch.norm(feature)))
        self._maybe_refresh_tail_classification(1)
    
    def update_batch(self, labels: torch.Tensor, features: torch.Tensor) -> None:
        """
        Update the memory bank with a whole batch of features at once.
        
        This is the path used by the training loop. Updating sample-by-sample means one
        prototype write, one host transfer and one percentile recomputation per image,
        which costs more than the forward/backward pass it accompanies.
        
        Args:
            labels: Class labels [batch_size]
            features: Feature vectors [batch_size, feature_dim]
        """
        if features.dim() != 2 or features.shape[0] == 0:
            return
        
        features = features.detach()
        labels = labels.detach().to(torch.long)
        
        # Single L2 normalization for the whole batch
        normalized = torch.nn.functional.normalize(features, p=2, dim=1)
        
        # One host transfer for the batch instead of one per sample
        labels_cpu = labels.cpu()
        normalized_cpu = normalized.cpu()
        
        batch_size = labels_cpu.shape[0]
        counts = torch.bincount(labels_cpu, minlength=self.num_classes)
        
        # Frequencies must be current before alpha is derived from them
        self.class_frequencies += counts
        self.total_samples += batch_size
        
        for class_id in torch.nonzero(counts, as_tuple=False).flatten().tolist():
            if not (0 <= class_id < self.num_classes):
                continue
            
            mask = labels_cpu == class_id
            class_features = normalized[mask.to(normalized.device)]
            num_in_batch = int(counts[class_id])
            
            alpha = self._get_adaptive_alpha(class_id)
            # Applying the per-sample EMA num_in_batch times with the batch mean as the
            # incoming feature collapses to a single update with this effective rate.
            effective_alpha = 1.0 - (1.0 - alpha) ** num_in_batch
            self._apply_ema(class_id, class_features.mean(dim=0), effective_alpha,
                            increment=num_in_batch)
            
            for feature in normalized_cpu[mask]:
                self._update_reservoir(class_id, feature)
            
            self._record_history(class_id, alpha, float(torch.norm(class_features[0])))
        
        self._maybe_refresh_tail_classification(batch_size)
    
    def _apply_ema(self, class_id: int, feature: torch.Tensor, alpha: float,
                   increment: int = 1) -> None:
        """Blend a feature into a class prototype with the given EMA rate."""
        feature = feature.to(self.ema_prototypes.device)
        if int(self.ema_counts[class_id]) == 0:
            # First feature for this class
            self.ema_prototypes[class_id] = feature
        else:
            # EMA update: M_c ← (1 - α) * M_c + α * feature
            self.ema_prototypes[class_id] = (1 - alpha) * self.ema_prototypes[class_id] + alpha * feature
        
        self.ema_counts[class_id] += increment
    
    def _record_history(self, class_id: int, alpha: float, feature_norm: float) -> None:
        """Append a capped history entry for later analysis."""
        if self.history_limit == 0:
            return
        
        history = self.update_history[class_id]
        history.append({
            'step': self.total_samples,
            'alpha': alpha,
            'feature_norm': feature_norm
        })
        if len(history) > self.history_limit:
            del history[:-self.history_limit]
    
    def _maybe_refresh_tail_classification(self, num_samples: int) -> None:
        """Recompute tail/head/medium membership once enough new samples have arrived."""
        self._samples_since_refresh += num_samples
        if self._samples_since_refresh >= self.tail_refresh_interval or not self.tail_classes:
            self._samples_since_refresh = 0
            self._update_tail_classification()
    
    def _update_reservoir(self, class_id: int, feature: torch.Tensor) -> None:
        """Update Reservoir buffer for a class using Reservoir sampling.
        
        Expects `feature` to already live on the CPU.
        """
        self.reservoir_counts[class_id] += 1
        n = int(self.reservoir_counts[class_id])
        
        if len(self.reservoir_buffers[class_id]) < self.capacity_per_class:
            # Buffer not full yet, just append
            self.reservoir_buffers[class_id].append(feature)
        else:
            # Reservoir sampling: replace with probability K/n
            if torch.rand(1).item() < self.capacity_per_class / n:
                # Randomly select position to replace
                replace_idx = torch.randint(0, self.capacity_per_class, (1,)).item()
                self.reservoir_buffers[class_id][replace_idx] = feature
    
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
        
        # Classify classes (plain ints so downstream JSON serialization stays simple)
        self.head_classes = {int(c) for c in sorted_indices[class_percentages[sorted_indices] >= head_threshold]}
        self.tail_classes = {int(c) for c in sorted_indices[class_percentages[sorted_indices] <= tail_threshold]}
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
        """Save memory bank to disk.
        
        Uses torch serialization rather than JSON: the reservoir holds
        num_classes × capacity_per_class × feature_dim floats, which as text is tens of
        megabytes per checkpoint and seconds of encoding time per save.
        """
        save_data = {
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'capacity_per_class': self.capacity_per_class,
            'alpha_base': self.alpha_base,
            'tail_threshold_percentile': self.tail_threshold_percentile,
            'ema_prototypes': self.ema_prototypes.detach().cpu(),
            'ema_counts': self.ema_counts.cpu(),
            'reservoir_counts': self.reservoir_counts.cpu(),
            'class_frequencies': self.class_frequencies.cpu(),
            'total_samples': self.total_samples,
            'tail_classes': sorted(int(c) for c in self.tail_classes),
            'head_classes': sorted(int(c) for c in self.head_classes),
            'medium_classes': sorted(int(c) for c in self.medium_classes),
            'update_history': {int(k): v for k, v in self.update_history.items()},
            'reservoir_buffers': {
                int(class_id): (torch.stack(buffer) if buffer else torch.empty(0, self.feature_dim))
                for class_id, buffer in self.reservoir_buffers.items()
            }
        }
        
        torch.save(save_data, filepath)
    
    def save_summary(self, filepath: str) -> None:
        """Write a small human-readable JSON summary alongside the tensor checkpoint."""
        total = max(self.total_samples, 1)
        summary = {
            'total_samples': self.total_samples,
            'tail_classes': sorted(int(c) for c in self.tail_classes),
            'medium_classes': sorted(int(c) for c in self.medium_classes),
            'head_classes': sorted(int(c) for c in self.head_classes),
            'class_frequencies': [int(c) for c in self.class_frequencies],
            'class_percentages': [round(float(c) / total * 100, 4) for c in self.class_frequencies],
            'prototype_norms': [round(float(n), 4) for n in torch.norm(self.ema_prototypes, dim=1)],
            'reservoir_fill': {int(c): len(b) for c, b in self.reservoir_buffers.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load memory bank from disk."""
        save_data = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # Restore basic parameters
        self.num_classes = save_data['num_classes']
        self.feature_dim = save_data['feature_dim']
        self.capacity_per_class = save_data['capacity_per_class']
        self.alpha_base = save_data['alpha_base']
        self.tail_threshold_percentile = save_data['tail_threshold_percentile']
        
        # Prototypes live on the compute device; counters stay on the CPU (see __init__).
        self.ema_prototypes = save_data['ema_prototypes'].to(dtype=torch.float32, device=self.device)
        self.ema_counts = save_data['ema_counts'].to(torch.long)
        self.reservoir_counts = save_data['reservoir_counts'].to(torch.long)
        self.class_frequencies = save_data['class_frequencies'].to(torch.long)
        self.total_samples = save_data['total_samples']
        self._samples_since_refresh = 0
        
        # Restore sets
        self.tail_classes = {int(c) for c in save_data['tail_classes']}
        self.head_classes = {int(c) for c in save_data['head_classes']}
        self.medium_classes = {int(c) for c in save_data['medium_classes']}
        
        # Restore update history
        self.update_history = defaultdict(list)
        for class_id, history in save_data['update_history'].items():
            self.update_history[int(class_id)] = history
        
        # Restore reservoir buffers
        self.reservoir_buffers = {i: [] for i in range(self.num_classes)}
        for class_id, buffer in save_data['reservoir_buffers'].items():
            buffer = buffer.to(torch.float32)
            self.reservoir_buffers[int(class_id)] = [row for row in buffer] if buffer.numel() else []
    
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
