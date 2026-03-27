import torch
from typing import Dict, List, Optional, Tuple
from .memory_bank import MemoryBank
import os
import json


class MemoryManager:
    """
    Manages the memory bank integration with the training loop.
    Handles feature extraction, memory updates, and tail class identification.
    """
    
    def __init__(self, 
                 model, 
                 num_classes: int,
                 capacity_per_class: int = 256,
                 alpha_base: float = 0.1,
                 tail_threshold_percentile: float = 20.0,
                 device: str = 'cpu',
                 save_dir: str = './memory_checkpoints'):
        """
        Initialize the memory manager.
        
        Args:
            model: The trained model (must have get_feature_dim method)
            num_classes: Number of classes in the dataset
            capacity_per_class: Maximum features per class in reservoir
            alpha_base: Base learning rate for EMA updates
            tail_threshold_percentile: Percentile for tail class identification
            device: Device to use for computations
            save_dir: Directory to save memory checkpoints
        """
        self.model = model
        self.device = device
        self.save_dir = save_dir
        
        # Get feature dimension from model
        if hasattr(model, 'get_feature_dim'):
            feature_dim = model.get_feature_dim()
        else:
            # Fallback: try to infer from model structure
            feature_dim = self._infer_feature_dim(model)
        
        # Initialize memory bank
        self.memory_bank = MemoryBank(
            num_classes=num_classes,
            feature_dim=feature_dim,
            capacity_per_class=capacity_per_class,
            alpha_base=alpha_base,
            tail_threshold_percentile=tail_threshold_percentile,
            device=device
        )
        
        # Statistics tracking
        self.update_stats = {
            'total_updates': 0,
            'updates_per_class': {i: 0 for i in range(num_classes)},
            'last_save_step': 0
        }
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def _infer_feature_dim(self, model) -> int:
        """Infer feature dimension from model structure."""
        # Try to find the last layer before classification
        for name, module in model.named_modules():
            if 'fc' in name or 'classifier' in name:
                if hasattr(module, 'in_features'):
                    return module.in_features
        
        # Fallback: assume ResNet-like architecture
        return 2048
    
    def update_memory(self, inputs: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update memory bank with features from a batch.
        
        Args:
            inputs: Input images [batch_size, channels, height, width]
            labels: Class labels [batch_size]
        """
        with torch.no_grad():
            # Extract features and logits
            outputs, features = self.model(inputs, return_features=True)
            
            # Update memory bank for each sample
            for i in range(len(labels)):
                class_id = labels[i].item()
                feature = features[i]
                
                # Update memory bank
                self.memory_bank.update(class_id, feature)
                
                # Update statistics
                self.update_stats['total_updates'] += 1
                self.update_stats['updates_per_class'][class_id] += 1
    
    def get_tail_classes(self) -> List[int]:
        """Get list of tail class IDs."""
        return self.memory_bank.get_tail_classes()
    
    def get_head_classes(self) -> List[int]:
        """Get list of head class IDs."""
        return self.memory_bank.get_head_classes()
    
    def get_medium_classes(self) -> List[int]:
        """Get list of medium class IDs."""
        return self.memory_bank.get_medium_classes()
    
    def get_class_distribution(self) -> Dict[str, List[int]]:
        """Get distribution of classes across tail/head/medium categories."""
        return self.memory_bank.get_class_distribution()
    
    def get_tail_class_features(self, k_per_class: int = 10) -> Dict[int, List[torch.Tensor]]:
        """Get features from tail classes for semantic prompt generation."""
        return self.memory_bank.get_tail_class_features(k_per_class)
    
    def get_tail_class_prototypes(self) -> Dict[int, torch.Tensor]:
        """Get prototypes for tail classes."""
        return self.memory_bank.get_tail_class_prototypes()
    
    def get_class_statistics(self, class_id: int) -> Dict:
        """Get comprehensive statistics for a class."""
        return self.memory_bank.get_class_statistics(class_id)
    
    def get_memory_usage(self) -> Dict:
        """Get memory usage statistics."""
        return self.memory_bank.get_memory_usage()
    
    def get_training_statistics(self) -> Dict:
        """Get training-related statistics."""
        return {
            'total_updates': self.update_stats['total_updates'],
            'updates_per_class': dict(self.update_stats['updates_per_class']),
            'memory_usage': self.get_memory_usage(),
            'class_distribution': self.get_class_distribution(),
            'tail_classes_count': len(self.get_tail_classes()),
            'head_classes_count': len(self.get_head_classes()),
            'medium_classes_count': len(self.get_medium_classes())
        }
    
    def save_memory(self, step: int, prefix: str = "memory") -> str:
        """
        Save memory bank to disk.
        
        Args:
            step: Current training step
            prefix: Prefix for the filename
            
        Returns:
            Path to saved file
        """
        filename = f"{prefix}_step_{step}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        # Save memory bank
        self.memory_bank.save(filepath)
        
        # Save training statistics
        stats_filepath = filepath.replace('.json', '_stats.json')
        with open(stats_filepath, 'w') as f:
            json.dump(self.update_stats, f, indent=2)
        
        self.update_stats['last_save_step'] = step
        return filepath
    
    def load_memory(self, filepath: str) -> None:
        """Load memory bank from disk."""
        self.memory_bank.load(filepath)
        
        # Try to load training statistics
        stats_filepath = filepath.replace('.json', '_stats.json')
        if os.path.exists(stats_filepath):
            with open(stats_filepath, 'r') as f:
                self.update_stats = json.load(f)
    
    def load_latest_memory(self, prefix: str = "memory") -> Optional[str]:
        """Load the most recent memory checkpoint."""
        if not os.path.exists(self.save_dir):
            return None
        
        # Find all memory files
        memory_files = [f for f in os.listdir(self.save_dir) 
                       if f.startswith(prefix) and f.endswith('.json') and 'stats' not in f]
        
        if not memory_files:
            return None
        
        # Extract step numbers and find the latest
        step_numbers = []
        for filename in memory_files:
            try:
                step = int(filename.split('_step_')[1].split('.')[0])
                step_numbers.append((step, filename))
            except:
                continue
        
        if not step_numbers:
            return None
        
        # Load the latest
        latest_step, latest_filename = max(step_numbers, key=lambda x: x[0])
        latest_filepath = os.path.join(self.save_dir, latest_filename)
        
        self.load_memory(latest_filepath)
        return latest_filepath
    
    def visualize_memory(self, save_path: Optional[str] = None) -> None:
        """Visualize memory bank statistics."""
        self.memory_bank.visualize_class_distribution(save_path)
    
    def print_summary(self) -> None:
        """Print a summary of the memory bank status."""
        stats = self.get_training_statistics()
        
        print("\n" + "="*60)
        print("MEMORY BANK SUMMARY")
        print("="*60)
        print(f"Total updates: {stats['total_updates']}")
        print(f"Memory usage: {stats['memory_usage']['total_memory_mb']:.2f} MB")
        print(f"Buffer utilization: {stats['memory_usage']['buffer_utilization']*100:.1f}%")
        print(f"Classes with data: {stats['memory_usage']['classes_with_data']}")
        print(f"Empty classes: {stats['memory_usage']['empty_classes']}")
        
        print(f"\nClass Distribution:")
        print(f"  Tail classes: {stats['tail_classes_count']}")
        print(f"  Medium classes: {stats['medium_classes_count']}")
        print(f"  Head classes: {stats['head_classes_count']}")
        
        if stats['tail_classes_count'] > 0:
            print(f"\nTail classes (first 10): {list(self.get_tail_classes())[:10]}")
        
        print("="*60)
    
    def get_semantic_prompt_data(self, class_id: int, k_features: int = 5) -> Dict:
        """
        Get data needed for semantic prompt generation for a specific class.
        
        Args:
            class_id: Class ID to get prompt data for
            k_features: Number of features to sample
            
        Returns:
            Dictionary containing prototype and sampled features
        """
        if class_id not in range(self.memory_bank.num_classes):
            return {}
        
        # Get prototype
        prototype = self.memory_bank.get_prototype(class_id)
        
        # Sample features from reservoir
        features = self.memory_bank.sample_features(class_id, k_features)
        
        # Get class statistics
        stats = self.memory_bank.get_class_statistics(class_id)
        
        return {
            'class_id': class_id,
            'prototype': prototype,
            'features': features,
            'is_tail': stats.get('is_tail', False),
            'class_frequency': stats.get('class_frequency', 0),
            'class_percentage': stats.get('class_percentage', 0.0),
            'feature_count': len(features)
        }
    
    def get_all_tail_prompt_data(self, k_features_per_class: int = 5) -> Dict[int, Dict]:
        """
        Get semantic prompt data for all tail classes.
        
        Args:
            k_features_per_class: Number of features to sample per class
            
        Returns:
            Dictionary mapping class_id to prompt data
        """
        tail_classes = self.get_tail_classes()
        prompt_data = {}
        
        for class_id in tail_classes:
            prompt_data[class_id] = self.get_semantic_prompt_data(class_id, k_features_per_class)
        
        return prompt_data
    
    def export_tail_class_analysis(self, export_path: str) -> None:
        """
        Export detailed analysis of tail classes for external use.
        
        Args:
            export_path: Path to save the analysis JSON file
        """
        tail_classes = self.get_tail_classes()
        analysis = {}
        
        for class_id in tail_classes:
            # Convert numpy int64 to regular Python int for JSON serialization
            class_id_int = int(class_id)
            
            # Get comprehensive data
            prompt_data = self.get_semantic_prompt_data(class_id_int, k_features=10)
            stats = self.memory_bank.get_class_statistics(class_id_int)
            
            # Convert tensors to lists for JSON serialization
            if 'prototype' in prompt_data:
                prompt_data['prototype'] = prompt_data['prototype'].cpu().numpy().tolist()
            
            if 'features' in prompt_data:
                prompt_data['features'] = [f.cpu().numpy().tolist() for f in prompt_data['features']]
            
            analysis[class_id_int] = {
                'prompt_data': prompt_data,
                'statistics': stats,
                'memory_info': {
                    'ema_count': stats['ema_count'],
                    'reservoir_count': stats['reservoir_count'],
                    'prototype_norm': stats['prototype_norm']
                }
            }
        
        # Add overall summary
        analysis['summary'] = {
            'total_tail_classes': len(tail_classes),
            'tail_class_ids': [int(cid) for cid in tail_classes],  # Convert all to regular ints
            'total_samples': self.memory_bank.total_samples,
            'export_timestamp': str(torch.cuda.Event() if torch.cuda.is_available() else 'cpu')
        }
        
        with open(export_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Tail class analysis exported to: {export_path}")
