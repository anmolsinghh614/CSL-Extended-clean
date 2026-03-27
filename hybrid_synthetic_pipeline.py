"""
Hybrid Synthetic Feature Pipeline
Integrates Phase 3 (DDPM) with existing Phase 5 (SD Images) pipeline
"""

import torch
import torch.nn.functional as F
import os
from typing import Dict, List, Tuple, Optional
import numpy as np

from phase3_feature_ddpm import FeatureDDPM


class HybridSyntheticFeatureGenerator:
    """
    Manages both Phase 3 (DDPM features) and Phase 5 (SD image features)
    Can use either or both simultaneously
    """
    
    def __init__(self, 
                 memory_manager=None,
                 phase3_model_path: Optional[str] = None,
                 phase5_features_path: Optional[str] = None,
                 device='cuda'):
        """
        Args:
            memory_manager: Your existing MemoryManager instance
            phase3_model_path: Path to trained Feature DDPM (optional)
            phase5_features_path: Path to SD-extracted features (optional)
            device: cuda/cpu
        """
        self.device = device
        self.memory_manager = memory_manager
        
        # Phase 3: DDPM Model
        self.phase3_model = None
        self.phase3_available = False
        if phase3_model_path and os.path.exists(phase3_model_path):
            self.phase3_model = self._load_phase3_model(phase3_model_path)
            self.phase3_available = True
            print("✓ Phase 3 (DDPM) loaded")
        
        # Phase 5: Pre-extracted features from SD images
        self.phase5_features = None
        self.phase5_available = False
        if phase5_features_path and os.path.exists(phase5_features_path):
            self.phase5_features = self._load_phase5_features(phase5_features_path)
            self.phase5_available = True
            print("✓ Phase 5 (SD features) loaded")
        
        # Class prototypes (computed from memory bank)
        self.class_prototypes = {}
        if memory_manager is not None:
            self._compute_prototypes()
        
        print(f"\nHybrid Pipeline Status:")
        print(f"  Phase 3 (DDPM):      {'✓ Available' if self.phase3_available else '✗ Not available'}")
        print(f"  Phase 5 (SD):        {'✓ Available' if self.phase5_available else '✗ Not available'}")
        print(f"  Memory Bank:         {'✓ Available' if memory_manager else '✗ Not available'}")
    
    def _load_phase3_model(self, checkpoint_path):
        """Load Phase 3 DDPM model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        model = FeatureDDPM(
            feature_dim=config['feature_dim'],
            num_classes=config['num_classes'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_timesteps=config['num_timesteps'],
            beta_schedule=config.get('beta_schedule', 'cosine')
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _load_phase5_features(self, features_path):
        """Load Phase 5 pre-extracted features"""
        data = torch.load(features_path, map_location='cpu')
        
        phase5_features = {}
        for class_id_str, class_data in data['classes'].items():
            class_id = int(class_id_str)
            phase5_features[class_id] = {
                'features': class_data['features'],
                'confidences': class_data['confidences'],
                'mean_confidence': class_data['mean_confidence']
            }
        
        return phase5_features
    
    def _compute_prototypes(self):
        """Compute class prototypes from memory bank"""
        # Assuming your memory bank has a method to get class features
        # Adjust based on your actual MemoryManager API
        
        for class_id in range(10):  # CIFAR-10
            try:
                # Option 1: If your memory manager has get_prototype
                if hasattr(self.memory_manager, 'get_prototype'):
                    prototype = self.memory_manager.get_prototype(class_id)
                
                # Option 2: If it has get_class_features
                elif hasattr(self.memory_manager, 'get_class_features'):
                    features = self.memory_manager.get_class_features(class_id)
                    if len(features) > 0:
                        prototype = features.mean(dim=0)
                    else:
                        continue
                
                # Option 3: If it has memory attribute
                elif hasattr(self.memory_manager, 'memory'):
                    if class_id in self.memory_manager.memory:
                        features = self.memory_manager.memory[class_id]['features']
                        prototype = features.mean(dim=0)
                    else:
                        continue
                
                else:
                    # Fallback: random prototype
                    prototype = torch.randn(512)
                
                self.class_prototypes[class_id] = prototype.to(self.device)
                
            except Exception as e:
                print(f"Warning: Could not get prototype for class {class_id}: {e}")
    
    def generate_features(self,
                         tail_classes: List[int],
                         num_samples_per_class: int,
                         method: str = 'hybrid',
                         phase3_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate synthetic features using available methods
        
        Args:
            tail_classes: List of tail class IDs [6, 7, 8, 9]
            num_samples_per_class: How many features to generate per class
            method: 'phase3', 'phase5', or 'hybrid'
            phase3_ratio: If hybrid, what fraction from Phase 3 (rest from Phase 5)
        
        Returns:
            features: [total_samples, feature_dim]
            labels: [total_samples]
            confidences: [total_samples]
        """
        if method == 'phase3':
            return self._generate_phase3(tail_classes, num_samples_per_class)
        
        elif method == 'phase5':
            return self._generate_phase5(tail_classes, num_samples_per_class)
        
        elif method == 'hybrid':
            return self._generate_hybrid(tail_classes, num_samples_per_class, phase3_ratio)
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'phase3', 'phase5', or 'hybrid'")
    
    def _generate_phase3(self, tail_classes, num_samples_per_class):
        """Generate using Phase 3 DDPM only"""
        if not self.phase3_available:
            raise ValueError("Phase 3 model not available!")
        
        all_features = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for class_id in tail_classes:
                class_ids = torch.full((num_samples_per_class,), class_id, 
                                      dtype=torch.long, device=self.device)
                
                # Generate features with DDPM
                features, confidences = self.phase3_model.sample_with_confidence(
                    class_ids, self.class_prototypes, self.device
                )
                
                all_features.append(features)
                all_labels.extend([class_id] * num_samples_per_class)
                all_confidences.append(confidences)
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.tensor(all_labels, dtype=torch.long, device=self.device)
        all_confidences = torch.cat(all_confidences, dim=0)
        
        return all_features, all_labels, all_confidences
    
    def _generate_phase5(self, tail_classes, num_samples_per_class):
        """Sample from Phase 5 pre-extracted features"""
        if not self.phase5_available:
            raise ValueError("Phase 5 features not available!")
        
        all_features = []
        all_labels = []
        all_confidences = []
        
        for class_id in tail_classes:
            if class_id not in self.phase5_features:
                print(f"Warning: No Phase 5 features for class {class_id}, skipping")
                continue
            
            phase5_data = self.phase5_features[class_id]
            available_features = phase5_data['features']
            available_confidences = phase5_data['confidences']
            
            # Sample with replacement if needed
            num_available = len(available_features)
            if num_samples_per_class <= num_available:
                indices = torch.randperm(num_available)[:num_samples_per_class]
            else:
                indices = torch.randint(0, num_available, (num_samples_per_class,))
            
            sampled_features = available_features[indices].to(self.device)
            sampled_confidences = available_confidences[indices].to(self.device)
            
            all_features.append(sampled_features)
            all_labels.extend([class_id] * num_samples_per_class)
            all_confidences.append(sampled_confidences)
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.tensor(all_labels, dtype=torch.long, device=self.device)
        all_confidences = torch.cat(all_confidences, dim=0)
        
        return all_features, all_labels, all_confidences
    
    def _generate_hybrid(self, tail_classes, num_samples_per_class, phase3_ratio):
        """
        Generate using both Phase 3 and Phase 5
        
        Combines features from both methods for maximum diversity
        """
        if not self.phase3_available and not self.phase5_available:
            raise ValueError("Neither Phase 3 nor Phase 5 available!")
        
        # If only one method available, use that
        if not self.phase3_available:
            return self._generate_phase5(tail_classes, num_samples_per_class)
        if not self.phase5_available:
            return self._generate_phase3(tail_classes, num_samples_per_class)
        
        # Both available: split according to ratio
        num_phase3 = int(num_samples_per_class * phase3_ratio)
        num_phase5 = num_samples_per_class - num_phase3
        
        # Generate from Phase 3
        features_p3, labels_p3, conf_p3 = self._generate_phase3(tail_classes, num_phase3)
        
        # Generate from Phase 5
        features_p5, labels_p5, conf_p5 = self._generate_phase5(tail_classes, num_phase5)
        
        # Combine
        all_features = torch.cat([features_p3, features_p5], dim=0)
        all_labels = torch.cat([labels_p3, labels_p5], dim=0)
        all_confidences = torch.cat([conf_p3, conf_p5], dim=0)
        
        # Shuffle to mix Phase 3 and Phase 5 samples
        perm = torch.randperm(len(all_features))
        all_features = all_features[perm]
        all_labels = all_labels[perm]
        all_confidences = all_confidences[perm]
        
        return all_features, all_labels, all_confidences
    
    def get_statistics(self):
        """Get statistics about available features"""
        stats = {
            'phase3_available': self.phase3_available,
            'phase5_available': self.phase5_available,
            'memory_bank_available': self.memory_manager is not None,
            'class_prototypes': len(self.class_prototypes)
        }
        
        if self.phase5_available:
            stats['phase5_classes'] = list(self.phase5_features.keys())
            stats['phase5_total_features'] = sum(
                len(data['features']) for data in self.phase5_features.values()
            )
        
        return stats


def train_epoch_with_hybrid_features(model, criterion, optimizer, train_loader, 
                                     device, epoch, hybrid_generator,
                                     tail_classes=[6, 7, 8, 9],
                                     synthetic_ratio=0.3,
                                     generation_method='hybrid',
                                     phase3_ratio=0.5):
    """
    Training epoch with hybrid synthetic features
    
    Supports: Phase 3 only, Phase 5 only, or Hybrid
    
    Args:
        hybrid_generator: HybridSyntheticFeatureGenerator instance
        generation_method: 'phase3', 'phase5', or 'hybrid'
        phase3_ratio: If hybrid, fraction from Phase 3
    """
    model.train()
    
    train_loss = 0.0
    correct = 0
    total = 0
    
    real_samples = 0
    synthetic_samples = 0
    phase3_samples = 0
    phase5_samples = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Extract features from real images
        outputs, real_features = model(inputs, return_features=True)
        
        # Update memory bank
        if hybrid_generator.memory_manager is not None:
            hybrid_generator.memory_manager.update_memory(inputs, labels)
            # Update prototypes periodically
            if batch_idx % 100 == 0:
                hybrid_generator._compute_prototypes()
        
        real_samples += len(inputs)
        
        # Prepare for mixed batch
        all_outputs = [outputs]
        all_labels = [labels]
        real_confidences = torch.ones(len(labels), device=device)
        confidences_list = [real_confidences]
        is_synthetic_list = [torch.zeros(len(labels), dtype=torch.bool, device=device)]
        
        # ========== GENERATE SYNTHETIC FEATURES ==========
        if synthetic_ratio > 0:
            # Calculate budget
            total_synthetic_budget = int(len(inputs) * synthetic_ratio)
            num_syn_per_class = max(1, total_synthetic_budget // len(tail_classes))
            
            try:
                # Generate using hybrid generator
                syn_features, syn_labels, syn_confidences = hybrid_generator.generate_features(
                    tail_classes=tail_classes,
                    num_samples_per_class=num_syn_per_class,
                    method=generation_method,
                    phase3_ratio=phase3_ratio
                )
                
                # Pass through classifier
                syn_outputs = model.fc(syn_features)
                
                all_outputs.append(syn_outputs)
                all_labels.append(syn_labels)
                confidences_list.append(syn_confidences)
                is_synthetic_list.append(torch.ones(len(syn_labels), dtype=torch.bool, device=device))
                
                synthetic_samples += len(syn_labels)
                
                # Track which method was used
                if generation_method == 'phase3':
                    phase3_samples += len(syn_labels)
                elif generation_method == 'phase5':
                    phase5_samples += len(syn_labels)
                else:  # hybrid
                    phase3_samples += int(len(syn_labels) * phase3_ratio)
                    phase5_samples += int(len(syn_labels) * (1 - phase3_ratio))
                
            except Exception as e:
                print(f"Warning: Feature generation failed: {e}")
        # ========== END GENERATION ==========
        
        # Combine real and synthetic
        combined_outputs = torch.cat(all_outputs, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)
        combined_confidences = torch.cat(confidences_list, dim=0)
        combined_is_synthetic = torch.cat(is_synthetic_list, dim=0)
        
        # Compute loss
        optimizer.zero_grad()
        
        if hasattr(criterion, 'forward') and 'sample_confidences' in criterion.forward.__code__.co_varnames:
            loss = criterion(
                combined_labels, combined_outputs, epoch,
                sample_confidences=combined_confidences,
                is_synthetic=combined_is_synthetic
            )
        else:
            loss = criterion(combined_labels, combined_outputs, epoch)
        
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_loss += loss.item() * len(combined_labels)
        _, predicted = combined_outputs.max(1)
        total += len(combined_labels)
        correct += predicted.eq(combined_labels).sum().item()
        
        # Print progress
        if batch_idx % 50 == 0:
            method_str = f"P3:{phase3_samples} P5:{phase5_samples}" if generation_method == 'hybrid' else f"{generation_method.upper()}:{synthetic_samples}"
            print(f"  Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f} "
                  f"[Real: {real_samples}, Syn: {synthetic_samples} ({method_str})]")
    
    train_loss /= total
    train_accuracy = 100. * correct / total
    
    print(f"\nEpoch {epoch} Training Summary:")
    print(f"  Real samples: {real_samples}")
    print(f"  Synthetic samples: {synthetic_samples}")
    if generation_method == 'hybrid':
        print(f"    Phase 3 (DDPM): {phase3_samples}")
        print(f"    Phase 5 (SD):   {phase5_samples}")
    print(f"  Synthetic ratio: {synthetic_samples / (real_samples + synthetic_samples) * 100:.1f}%")
    
    return train_loss, train_accuracy


# ============================================================================
# USAGE EXAMPLES FOR YOUR EXISTING PIPELINE
# ============================================================================

def example_integrate_with_existing_pipeline():
    """
    Example showing how to integrate with your existing code
    """
    
    # Assuming you have these from your existing code:
    # - memory_manager: Your MemoryManager instance
    # - model: Your ResNet32 model
    # - train_loader: Your data loader
    # - criterion: Your loss function
    # - optimizer: Your optimizer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Step 1: Initialize hybrid generator
    hybrid_gen = HybridSyntheticFeatureGenerator(
        memory_manager=None,  # Your memory_manager here
        phase3_model_path='./feature_ddpm_checkpoints/feature_ddpm_best.pth',
        phase5_features_path='./synthetic_features_phase5.pth',  # From your SD pipeline
        device=device
    )
    
    # Step 2: Check what's available
    stats = hybrid_gen.get_statistics()
    print("\nPipeline Status:")
    print(f"  Phase 3: {stats['phase3_available']}")
    print(f"  Phase 5: {stats['phase5_available']}")
    
    # Step 3: Use in training
    # Replace your training loop with:
    
    # train_loss, train_acc = train_epoch_with_hybrid_features(
    #     model=model,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     train_loader=train_loader,
    #     device=device,
    #     epoch=epoch,
    #     hybrid_generator=hybrid_gen,
    #     tail_classes=[6, 7, 8, 9],
    #     synthetic_ratio=0.5,
    #     generation_method='hybrid',  # or 'phase3' or 'phase5'
    #     phase3_ratio=0.6  # 60% from Phase 3, 40% from Phase 5
    # )


if __name__ == "__main__":
    print("Hybrid Synthetic Feature Pipeline")
    print("="*70)
    print("\nThis module integrates Phase 3 (DDPM) with your existing")
    print("Phase 5 (Stable Diffusion) pipeline.")
    print("\nSee example_integrate_with_existing_pipeline() for usage.")