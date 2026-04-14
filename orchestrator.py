"""
==============================================
MEMORY-CONDITIONED DIFFUSION MODEL ORCHESTRATOR
==============================================
Complete Pipeline Integration for Improving Tail Class Accuracy
Combines all phases: Memory Bank Training, Prompt Generation, 
Image Synthesis, Feature Extraction, and Adaptive Training

Author: Memory-Conditioned Diffusion Framework
Version: 1.0
==============================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Import all your existing components
from models import ResNet32, ResNet50
from utils import CSLLossFunc, plot_loss_curve, plot_accuracy_curve
from utils.memory_manager import MemoryManager
from utils.memory_bank import MemoryBank
from utils.synthetic_training_integration import ConfidenceAdaptiveCSL
from utils.visual_exemplar_prompt_generator import VisualExemplarPromptGenerator

# Import Option 3 components
from option3_image_generator import Option3ImageGenerator
from phase3_feature_ddpm import FeatureDDPM
from train_feature_ddpm import extract_features_from_dataset, train_feature_ddpm
from hybrid_synthetic_pipeline import HybridSyntheticFeatureGenerator, train_epoch_with_hybrid_features


class MemoryConditionedOrchestrator:
    """
    Main orchestrator class that manages the entire pipeline:
    1. Creates imbalanced CIFAR-10 dataset
    2. Trains memory bank on imbalanced data
    3. Performs tail class analysis
    4. Generates prompts using Option 3 (BLIP+CLIP)
    5. Generates synthetic images from prompts
    6. Extracts features using DDPM-based method
    7. Trains with hybrid approach (real + synthetic + DDPM features)
    8. Iteratively improves tail class accuracy
    """
    
    def __init__(self, config=None):
        """Initialize the orchestrator with configuration."""
        self.config = self._get_default_config() if config is None else config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tracking metrics
        self.metrics = {
            'epoch_losses': [],
            'epoch_accuracies': [],
            'tail_class_accuracies': defaultdict(list),
            'head_class_accuracies': defaultdict(list),
            'generation_rounds': [],
            'synthetic_samples_generated': defaultdict(int)
        }
        
        # Class names for CIFAR-10
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Setup directories
        self._setup_directories()
        
        print("\n" + "="*80)
        print("MEMORY-CONDITIONED DIFFUSION MODEL ORCHESTRATOR INITIALIZED")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Configuration loaded:")
        for key, value in self.config.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        print("="*80 + "\n")
    
    def _get_default_config(self):
        """Get default configuration for the pipeline."""
        return {
            # Dataset configuration
            'dataset': {
                'name': 'CIFAR10',
                'imbalance_ratio': 10,       # Moderate imbalance for best results
                'num_classes': 10,
                'batch_size': 128,
                'num_workers': 4
            },
            
            # Model configuration
            'model': {
                'architecture': 'ResNet32',
                'feature_dim': 512,  # ResNet32 (actually ResNet34) uses 512-dim features
                'num_classes': 10
            },
            
            # Memory Bank configuration
            'memory_bank': {
                'capacity_per_class': 256,
                'alpha_base': 0.1,
                'tail_threshold_percentile': 30.0,
                'update_interval': 100
            },
            
            # Training configuration
            'training': {
                'initial_epochs': 200,        # ResNet-34 needs ~200 epochs on CIFAR-10
                'synthetic_epochs': 25,       # Enough to integrate synthetic features
                'lr': 0.1,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'scheduler_milestones': [160, 180],  # Decay at 80% and 90% of total
                'scheduler_gamma': 0.1
            },
            
            # Synthetic generation configuration
            'generation': {
                'num_prompts_per_tail_class': 50,
                'images_per_prompt': 4,
                'generation_rounds': 3,
                'tail_improvement_threshold': 5.0,  # 5 percentage points improvement to continue
                'option3_temperature': 0.8,
                'use_blip': True,
                'use_clip': True
            },
            
            # DDPM configuration
            'ddpm': {
                'enabled': True,
                'num_timesteps': 1000,
                'beta_schedule': 'cosine',
                'hidden_dim': 1024,           # 2x feature_dim for better representational capacity
                'num_layers': 4,
                'training_steps': 10000,      # More training = better feature quality
                'features_per_class': 300     # More synthetic features per tail class
            },
            
            # Paths
            'paths': {
                'checkpoint_dir': './checkpoints',
                'memory_dir': './memory_checkpoints',
                'prompts_dir': './prompts',
                'images_dir': './synthetic_images',
                'features_dir': './synthetic_features',
                'logs_dir': './logs',
                'results_dir': './results'
            }
        }
    
    def _setup_directories(self):
        """Create necessary directories."""
        for path_key, path_value in self.config['paths'].items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
        print("✓ Directories created")
    
    def step1_create_imbalanced_dataset(self):
        """Step 1: Create imbalanced CIFAR-10 dataset."""
        print("\n" + "="*80)
        print("STEP 1: CREATING IMBALANCED CIFAR-10 DATASET")
        print("="*80)
        
        # Data transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load datasets
        train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                        download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
        
        # Create imbalanced dataset
        imbalanced_dataset, samples_per_class = self._create_cifar10_lt(
            train_dataset, 
            imbalance_ratio=self.config['dataset']['imbalance_ratio']
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            imbalanced_dataset, 
            batch_size=self.config['dataset']['batch_size'],
            shuffle=True, 
            num_workers=self.config['dataset']['num_workers']
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['dataset']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers']
        )
        
        # Store class distribution
        self.samples_per_class = samples_per_class
        tail_percentile = self.config['memory_bank']['tail_threshold_percentile']
        head_percentile = 100 - tail_percentile
        threshold = np.percentile(samples_per_class, tail_percentile)
        self.tail_classes = [i for i, count in enumerate(samples_per_class) if count <= threshold]
        self.head_classes = [i for i, count in enumerate(samples_per_class) if count >= np.percentile(samples_per_class, head_percentile)]
        
        print(f"\n✓ Dataset created successfully")
        print(f"  Total samples: {sum(samples_per_class)}")
        print(f"  Tail classes: {[self.class_names[i] for i in self.tail_classes]}")
        print(f"  Head classes: {[self.class_names[i] for i in self.head_classes]}")
        
        return imbalanced_dataset, samples_per_class
    
    def _create_cifar10_lt(self, train_dataset, imbalance_ratio=100):
        """Create long-tail version of CIFAR-10."""
        class_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(train_dataset):
            class_indices[label].append(idx)
        
        img_max = len(class_indices[0])
        img_num_per_cls = []
        
        for cls_idx in range(10):
            num = img_max * (imbalance_ratio ** (-cls_idx / 9.0))
            img_num_per_cls.append(int(num))
        
        selected_indices = []
        np.random.seed(42)
        for cls_idx, num_samples in enumerate(img_num_per_cls):
            indices = class_indices[cls_idx]
            np.random.shuffle(indices)
            selected_indices.extend(indices[:num_samples])
        
        return Subset(train_dataset, selected_indices), img_num_per_cls
    
    def step2_train_memory_bank(self, epochs=20):
        """Step 2: Train model with memory bank on imbalanced dataset."""
        print("\n" + "="*80)
        print("STEP 2: TRAINING MEMORY BANK ON IMBALANCED DATA")
        print("="*80)
        
        # Initialize model
        if self.config['model']['architecture'] == 'ResNet32':
            self.model = ResNet32(num_classes=self.config['model']['num_classes']).to(self.device)
        else:
            self.model = ResNet50(num_classes=self.config['model']['num_classes']).to(self.device)
        
        # Initialize memory manager with memory bank
        self.memory_manager = MemoryManager(
            model=self.model,
            num_classes=self.config['model']['num_classes'],
          #  feature_dim=self.config['model']['feature_dim'],
            capacity_per_class=self.config['memory_bank']['capacity_per_class'],
            alpha_base=self.config['memory_bank']['alpha_base'],
            tail_threshold_percentile=self.config['memory_bank']['tail_threshold_percentile'],
            device=self.device,
            save_dir=self.config['paths']['memory_dir']
        )
        
        # Initialize CSL loss with class-balanced weights
        self.criterion = CSLLossFunc(
            target_class_index=self.tail_classes,
            num_classes=self.config['model']['num_classes'],
            samples_per_class=self.samples_per_class
        )
        
        # Initialize optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            momentum=self.config['training']['momentum'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.config['training']['scheduler_milestones'],
            gamma=self.config['training']['scheduler_gamma']
        )
        
        # Training loop
        best_acc = 0.0
        for epoch in range(epochs):
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            
            # Train
            train_loss, train_acc = self._train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, class_accuracies = self._validate_epoch(epoch)
            
            # Track metrics
            self.metrics['epoch_losses'].append(train_loss)
            self.metrics['epoch_accuracies'].append(val_acc)
            
            for cls_idx in self.tail_classes:
                self.metrics['tail_class_accuracies'][cls_idx].append(
                    class_accuracies.get(cls_idx, 0.0)
                )
            
            for cls_idx in self.head_classes:
                self.metrics['head_class_accuracies'][cls_idx].append(
                    class_accuracies.get(cls_idx, 0.0)
                )
            
            # Save checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                self._save_checkpoint(epoch, val_acc)
            
            # Save memory bank
            if (epoch + 1) % 5 == 0:
                self._save_memory_bank(epoch)
            
            self.scheduler.step()
        
        print(f"\n✓ Memory bank training completed")
        print(f"  Best validation accuracy: {best_acc:.2f}%")
        
        # Perform tail class analysis
        tail_analysis = self.memory_manager.memory_bank.get_tail_classes()
        tail_analysis = [int(x) for x in tail_analysis]
        self._save_tail_analysis(tail_analysis)
        
        return best_acc
    
    def _train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs, features = self.model(inputs, return_features=True)
            loss = self.criterion(labels, outputs, epoch)
            
            loss.backward()
            self.optimizer.step()
            
            # Update memory bank directly using already-extracted features (no extra forward pass)
            with torch.no_grad():
                for i in range(len(labels)):
                    self.memory_manager.memory_bank.update(labels[i].item(), features[i].detach())
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss/total,
                'acc': 100.*correct/total
            })
        
        return train_loss/total, 100.*correct/total
    
    def _validate_epoch(self, epoch):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(labels, outputs, epoch)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                for label, pred in zip(labels, predicted):
                    class_total[label.item()] += 1
                    if pred == label:
                        class_correct[label.item()] += 1
        
        val_acc = 100. * correct / total
        class_accuracies = {
            cls: 100. * class_correct[cls] / class_total[cls] 
            for cls in class_total
        }
        
        # Print class-wise accuracy
        print(f"\n  Overall Accuracy: {val_acc:.2f}%")
        print("  Class-wise Accuracy:")
        for cls_idx in range(self.config['model']['num_classes']):
            acc = class_accuracies.get(cls_idx, 0.0)
            class_type = "TAIL" if cls_idx in self.tail_classes else "HEAD" if cls_idx in self.head_classes else "MED"
            print(f"    {self.class_names[cls_idx]:10s} [{class_type}]: {acc:.2f}%")
        
        return val_loss/total, val_acc, class_accuracies
    
    def step3_generate_prompts(self):
        """Step 3: Generate prompts using Option 3 (BLIP+CLIP)."""
        print("\n" + "="*80)
        print("STEP 3: GENERATING PROMPTS USING OPTION 3 (BLIP+CLIP)")
        print("="*80)
        
        # Initialize Option 3 generator
        self.prompt_generator = VisualExemplarPromptGenerator(
            memory_manager=self.memory_manager,
            device=self.device,
        )
        
        # Generate prompts for tail classes
        all_prompts = {}
        
        for cls_idx in self.tail_classes:
            print(f"\nGenerating prompts for class {cls_idx} ({self.class_names[cls_idx]})...")
            
            # Get visual exemplars from memory bank
            exemplar_features = self.memory_manager.memory_bank.get_class_exemplars(
                cls_idx, 
                num_exemplars=5
            )
            
            # Generate semantic prompts
            prompts = self.prompt_generator.generate_prompts(
                class_idx=cls_idx,
                exemplar_features=exemplar_features,
                num_prompts=self.config['generation']['num_prompts_per_tail_class'],
                temperature=self.config['generation']['option3_temperature']
            )
            
            all_prompts[cls_idx] = prompts
            print(f"  Generated {len(prompts)} prompts")
            
            # Show sample prompts
            print("  Sample prompts:")
            for i, prompt in enumerate(prompts[:3]):
                print(f"    {i+1}. {prompt}")
        
        # Save prompts
        prompts_file = os.path.join(
            self.config['paths']['prompts_dir'],
            f"option3_prompts_round{len(self.metrics['generation_rounds'])}.json"
        )
        
        with open(prompts_file, 'w') as f:
            json.dump({
                str(k): v for k, v in all_prompts.items()
            }, f, indent=2)
        
        print(f"\n✓ Prompts saved to {prompts_file}")
        self.current_prompts = all_prompts
        
        return all_prompts
    
    def step4_generate_images(self, prompts):
        """Step 4: Generate synthetic images from prompts."""
        print("\n" + "="*80)
        print("STEP 4: GENERATING SYNTHETIC IMAGES")
        print("="*80)
        
        # Initialize image generator
        self.image_generator = Option3ImageGenerator(
            model_type="stable_diffusion",
            device=str(self.device),
            output_dir=self.config['paths']['images_dir']
        )
        
        generated_images = {}
        
        for cls_idx, class_prompts in prompts.items():
            cls_idx = int(cls_idx) if isinstance(cls_idx, str) else cls_idx
            print(f"\nGenerating images for class {cls_idx} ({self.class_names[cls_idx]})...")
            
            class_images = []
            for prompt_idx, prompt in enumerate(tqdm(class_prompts, desc="Generating")):
                # Generate multiple images per prompt
                images = self.image_generator.generate_batch(
                    prompts=[prompt] * self.config['generation']['images_per_prompt'],
                    class_idx=cls_idx,
                    batch_size=self.config['generation']['images_per_prompt']
                )
                class_images.extend(images)
                
                # Track synthetic samples
                self.metrics['synthetic_samples_generated'][cls_idx] += len(images)
            
            generated_images[cls_idx] = class_images
            print(f"  Generated {len(class_images)} images")
        
        print(f"\n✓ Image generation completed")
        self.current_images = generated_images
        
        return generated_images
    
    def step5_extract_features(self, images=None):
        """Step 5: Extract features from synthetic images and train DDPM."""
        print("\n" + "="*80)
        print("STEP 5: EXTRACTING FEATURES (INCLUDING DDPM)")
        print("="*80)
        
        synthetic_features = {}
        
        # Extract features from generated images
        if images:
            print("\nExtracting features from synthetic images...")
            for cls_idx, class_images in images.items():
                cls_idx = int(cls_idx) if isinstance(cls_idx, str) else cls_idx
                class_features = []
                
                for img in class_images:
                    # Convert image to tensor and extract features
                    img_tensor = self._preprocess_image(img).to(self.device)
                    
                    with torch.no_grad():
                        # Extract features using hook method
                        activation = {}
                        def get_activation(name):
                            def hook(model, input, output):
                                activation[name] = output.detach()
                            return hook
                        
                        # Use robust feature extraction
                        if hasattr(self.model, 'forward') and 'return_features' in self.model.forward.__code__.co_varnames:
                            _, features = self.model(img_tensor.unsqueeze(0), return_features=True)
                        elif hasattr(self.model, 'get_feature_vector'): # Fallback for other models
                            features = self.model.get_feature_vector(img_tensor.unsqueeze(0))
                        else:
                            # Default fallback (might still be logits if return_features not supported)
                            features = self.model(img_tensor.unsqueeze(0))
                            
                        # Ensure we have the feature vector, not logits
                        if features.dim() > 1:
                            features = features.squeeze(0)
                        
                        class_features.append(features.cpu())
                
                synthetic_features[cls_idx] = torch.stack(class_features)
                print(f"  Class {cls_idx}: {len(class_features)} features extracted")
        
        # Train DDPM for additional feature generation
        if self.config['ddpm']['enabled']:
            print("\nTraining DDPM for feature generation...")
            ddpm_features = self._train_and_generate_ddpm_features()
            
            # Combine features
            for cls_idx, features in ddpm_features.items():
                if cls_idx in synthetic_features:
                    # Ensure both tensors are on CPU before concatenation to avoid device mismatch
                    current_features = synthetic_features[cls_idx].cpu()
                    new_features = features.cpu()
                    synthetic_features[cls_idx] = torch.cat([
                        current_features, new_features
                    ])
                else:
                    synthetic_features[cls_idx] = features.cpu()
        
        # Save features
        features_file = os.path.join(
            self.config['paths']['features_dir'],
            f"synthetic_features_round{len(self.metrics['generation_rounds'])}.pt"
        )
        torch.save(synthetic_features, features_file)
        
        print(f"\n✓ Features saved to {features_file}")
        self.current_synthetic_features = synthetic_features
        
        return synthetic_features
    
    def _train_and_generate_ddpm_features(self):
        """Train DDPM model and generate additional features."""
        from train_feature_ddpm import FeatureDataset
        
        # Extract features from real data for training
        print("  Extracting real features for DDPM training...")
        real_features = extract_features_from_dataset(
            self.model, self.train_loader, self.device, 
            num_classes=self.config['model']['num_classes']
        )
        
        # Create training dataset for DDPM
        all_features = []
        all_labels = []
        
        for cls_idx in self.tail_classes:
            if cls_idx in real_features and len(real_features[cls_idx]) > 0:
                features = real_features[cls_idx]
                # Augment small classes
                if len(features) < 100:
                    repeat = min(5, 100 // len(features) + 1)
                    features = features.repeat(repeat, 1)
                all_features.append(features)
                all_labels.extend([cls_idx] * len(features))
        
        if len(all_features) == 0:
            print("  Warning: No features available for DDPM training")
            return {}
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.tensor(all_labels, dtype=torch.long)
        
        ddpm_dataset = FeatureDataset(all_features, all_labels)
        ddpm_loader = DataLoader(
            ddpm_dataset,
            batch_size=min(64, len(ddpm_dataset)),
            shuffle=True
        )
        
        # Initialize DDPM model — auto-detect feature_dim from the model
        actual_feature_dim = self.model.get_feature_dim() if hasattr(self.model, 'get_feature_dim') else self.config['model']['feature_dim']
        ddpm_model = FeatureDDPM(
            feature_dim=actual_feature_dim,
            num_classes=self.config['model']['num_classes'],
            hidden_dim=self.config['ddpm']['hidden_dim'],
            num_layers=self.config['ddpm']['num_layers'],
            num_timesteps=self.config['ddpm']['num_timesteps'],
            beta_schedule=self.config['ddpm']['beta_schedule']
        ).to(self.device)
        
        # Setup optimizer
        optimizer = optim.AdamW(ddpm_model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Train DDPM
        print("  Training DDPM model...")
        num_epochs = min(50, self.config['ddpm']['training_steps'] // len(ddpm_loader))
        
        for epoch in range(num_epochs):
            avg_loss = train_feature_ddpm(ddpm_model, ddpm_loader, optimizer, self.device, epoch + 1)
            if (epoch + 1) % 10 == 0:
                print(f"    DDPM Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Generate features for tail classes
        print("  Generating DDPM features for tail classes...")
        ddpm_features = {}
        
        for cls_idx in self.tail_classes:
            num_features = self.config['ddpm']['features_per_class']
            class_label = torch.tensor([cls_idx] * num_features).to(self.device)
            
            # Generate features
            with torch.no_grad():
                generated = ddpm_model.sample(
                    class_ids=class_label,
                    device=self.device
                )
            
            ddpm_features[cls_idx] = generated
            print(f"    Class {cls_idx}: {num_features} DDPM features generated")
        
        # Save DDPM model
        ddpm_path = os.path.join(
            self.config['paths']['checkpoint_dir'],
            f"ddpm_round{len(self.metrics['generation_rounds'])}.pt"
        )
        torch.save({
            'model_state_dict': ddpm_model.state_dict(),
            'config': {
                'feature_dim': actual_feature_dim,
                'num_classes': self.config['model']['num_classes'],
                'hidden_dim': self.config['ddpm']['hidden_dim'],
                'num_layers': self.config['ddpm']['num_layers'],
                'num_timesteps': self.config['ddpm']['num_timesteps'],
                'beta_schedule': self.config['ddpm']['beta_schedule']
            }
        }, ddpm_path)
        
        return ddpm_features
    
    def step6_train_with_synthetic(self, synthetic_features, epochs=5):
        """Step 6: Train model with real data + synthetic features."""
        print("\n" + "="*80)
        print("STEP 6: TRAINING WITH SYNTHETIC DATA")
        print("="*80)
    
        # Store baseline accuracy for comparison
        baseline_acc = self.metrics['epoch_accuracies'][-1] if self.metrics['epoch_accuracies'] else 0.0
        baseline_tail_acc = self._compute_tail_accuracy()
    
        print(f"\nBaseline Performance:")
        print(f"  Overall Accuracy: {baseline_acc:.2f}%")
        print(f"  Tail Classes Accuracy: {baseline_tail_acc:.2f}%")
    
    # Create synthetic dataset from features
        print("\nPreparing augmented training data...")
        from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
    
        synthetic_datasets = []
        total_synthetic = 0
    
        for cls_idx, features in synthetic_features.items():
            if features.shape[0] > 0:
                # Create labels for synthetic samples
                labels = torch.full((features.shape[0],), cls_idx, dtype=torch.long)
            
                # Create dataset from features (features are already extracted, no images needed)
                synthetic_datasets.append(TensorDataset(features, labels))
                total_synthetic += features.shape[0]
                print(f"  Class {cls_idx}: {features.shape[0]} synthetic samples")
    
        if not synthetic_datasets:
            print("⚠ No synthetic features available, skipping augmented training")
            return {'improvement': 0.0, 'baseline': baseline_tail_acc, 'augmented': baseline_tail_acc}
    
        # Combine all synthetic data
        combined_synthetic = ConcatDataset(synthetic_datasets)
        print(f"\nTotal synthetic samples: {total_synthetic}")
        print(f"Total real samples: {len(self.train_loader.dataset)}")
    
        # Create augmented dataloader (real + synthetic)
        # Note: For real data, we need to extract features on-the-fly during training
        augmented_loader = self._create_hybrid_dataloader(combined_synthetic)
    
        print(f"Augmented dataset size: {len(augmented_loader.dataset)}")
    
        # Training setup
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config['training']['lr'] * 0.1,  # Lower LR for fine-tuning
            momentum=self.config['training']['momentum'],
            weight_decay=self.config['training']['weight_decay']
        )
    
        # Use CSLLossFunc to maintain tail-class boosting during fine-tuning
        criterion = CSLLossFunc(
            target_class_index=self.tail_classes,
            num_classes=self.config['model']['num_classes'],
            samples_per_class=self.samples_per_class
        ).to(self.device)
    
        # Training loop
        print(f"\nTraining with augmented data for {epochs} epochs...")
    
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
        
            pbar = tqdm(augmented_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        
            for batch_idx, (data, labels) in enumerate(pbar):
                data, labels = data.to(self.device), labels.to(self.device)
            
                # Forward pass - handle mixed batch of Images (4D) and Features (2D)
                optimizer.zero_grad()
                
                if data.dim() == 4:  # Images (B, C, H, W)
                    # Use model's properly defined interface to get both output and features
                    outputs, feature_data = self.model(data, return_features=True)
                    feature_data = feature_data.detach()
                    
                else:  # Features (B, D)
                    # Ensure correct shape (B, D)
                    if data.dim() > 2:
                        data = data.view(data.size(0), -1)
                    
                    # For features, we skip the backbone and go straight to the classifier
                    # ResNet wrapper uses self.model.fc as the classifier
                    if hasattr(self.model, 'fc'):
                         outputs = self.model.fc(data)
                    elif hasattr(self.model, 'model') and hasattr(self.model.model, 'fc'):
                         outputs = self.model.model.fc(data)
                    else:
                         # Fallback or error if model structure is unexpected
                         # Try calling model directly if it accepts features (unlikely for ResNet)
                         try:
                            outputs = self.model(data)
                         except:
                            raise AttributeError("Could not find classifier (.fc) in model for feature processing")
                    
                    feature_data = data.detach()
            
                loss = criterion(labels, outputs, epoch)
            
                # Backward pass
                loss.backward()
                optimizer.step()
            
                # Update memory bank with new features
                for i, label in enumerate(labels):
                    self.memory_manager.memory_bank.update(label.item(), feature_data[i].detach())
            
                # Statistics
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{epoch_loss/(batch_idx+1):.3f}',
                    'acc': f'{100.*correct/total:.1f}%'
                })
        
            # Validation after each epoch
            val_acc = self._validate_model()
            tail_acc = self._compute_tail_accuracy()
        
            print(f"\n  Epoch {epoch+1} Results:")
            print(f"    Training Accuracy: {100.*correct/total:.2f}%")
            print(f"    Validation Accuracy: {val_acc:.2f}%")
            print(f"    Tail Classes Accuracy: {tail_acc:.2f}%")
        
            # Save checkpoint if improved
            if val_acc > max(self.metrics['epoch_accuracies'], default=0):
                checkpoint_path = os.path.join(
                    self.config['paths']['checkpoint_dir'],
                    f"model_synthetic_round{len(self.metrics['generation_rounds'])}_acc{val_acc:.2f}.pt"
                )
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'accuracy': val_acc,
                    'tail_accuracy': tail_acc
                }, checkpoint_path)
                print(f"    ✓ Checkpoint saved: {checkpoint_path}")
    
        # Final evaluation
        final_acc = self._validate_model()
        final_tail_acc = self._compute_tail_accuracy()
    
        improvement = {
            'overall': final_acc - baseline_acc,
            'tail': final_tail_acc - baseline_tail_acc,
            'baseline_overall': baseline_acc,
            'baseline_tail': baseline_tail_acc,
            'augmented_overall': final_acc,
            'augmented_tail': final_tail_acc
        }
    
        print("\n" + "="*80)
        print("TRAINING RESULTS")
        print("="*80)
        print(f"Overall Accuracy Improvement: {improvement['overall']:+.2f}%")
        print(f"Tail Classes Improvement: {improvement['tail']:+.2f}%")
        print(f"\nFinal Performance:")
        print(f"  Overall: {final_acc:.2f}%")
        print(f"  Tail Classes: {final_tail_acc:.2f}%")
    
        return improvement


    def _create_hybrid_dataloader(self, synthetic_dataset):
        """Create dataloader that combines real images with synthetic features using homogeneous batches."""
        from torch.utils.data import DataLoader, ConcatDataset, Sampler
        import random

        # Custom Sampler to ensure batches only contain EITHER images OR features
        class HomogeneousBatchSampler(Sampler):
            def __init__(self, data_source, batch_size, split_idx):
                self.data_source = data_source
                self.batch_size = batch_size
                self.split_idx = split_idx
                
                self.real_indices = list(range(split_idx))
                self.syn_indices = list(range(split_idx, len(data_source)))

            def __iter__(self):
                # Shuffle indices
                real_indices = torch.randperm(len(self.real_indices)).tolist()
                syn_indices = [x + self.split_idx for x in torch.randperm(len(self.syn_indices)).tolist()]
                
                batches = []
                # Real batches
                for i in range(0, len(real_indices), self.batch_size):
                    batches.append(real_indices[i:i + self.batch_size])
                
                # Synthetic batches
                for i in range(0, len(syn_indices), self.batch_size):
                    batches.append(syn_indices[i:i + self.batch_size])
                
                # Shuffle the order of batches (interleave real and synthetic)
                random.shuffle(batches)
                
                for batch in batches:
                    yield batch

            def __len__(self):
                return (len(self.real_indices) + self.batch_size - 1) // self.batch_size + \
                       (len(self.syn_indices) + self.batch_size - 1) // self.batch_size

        # Combine real dataset with synthetic features dataset
        combined = ConcatDataset([self.train_loader.dataset, synthetic_dataset])
        
        # Split index is the length of the real dataset
        split_idx = len(self.train_loader.dataset)
        
        # Create sampler
        batch_sampler = HomogeneousBatchSampler(combined, self.config['dataset']['batch_size'], split_idx)
    
        loader = DataLoader(
            combined,
            batch_sampler=batch_sampler,
            num_workers=self.config['dataset']['num_workers']
        )
    
        return loader


    def _validate_model(self):
        """Validate model and return accuracy."""
        self.model.eval()
        correct = 0
        total = 0
    
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    
        accuracy = 100. * correct / total
        return accuracy


    def _compute_tail_accuracy(self):
        """Compute accuracy specifically for tail classes."""
        self.model.eval()
        correct = 0
        total = 0
    
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
            
                # Filter for tail classes only
                tail_mask = torch.tensor([l.item() in self.tail_classes for l in labels])
                if tail_mask.sum() == 0:
                    continue
            
                tail_images = images[tail_mask]
                tail_labels = labels[tail_mask]
            
                outputs = self.model(tail_images)
                _, predicted = outputs.max(1)
                total += tail_labels.size(0)
                correct += predicted.eq(tail_labels).sum().item()
    
        if total == 0:
            return 0.0
    
        accuracy = 100. * correct / total
        return accuracy
    
    def run_iterative_improvement(self, max_rounds=3):
        """Run the complete iterative improvement pipeline."""
        print("\n" + "="*80)
        print("RUNNING ITERATIVE IMPROVEMENT PIPELINE")
        print("="*80)
        
        # Step 1: Create imbalanced dataset
        self.step1_create_imbalanced_dataset()
        
        # Step 2: Train memory bank
        initial_acc = self.step2_train_memory_bank(
            epochs=self.config['training']['initial_epochs']
        )
        
        # Iterative improvement rounds
        for round_idx in range(max_rounds):
            print(f"\n{'='*80}")
            print(f"IMPROVEMENT ROUND {round_idx + 1}/{max_rounds}")
            print(f"{'='*80}")
            
            self.metrics['generation_rounds'].append(round_idx)
            
            # Step 3: Generate prompts
            prompts = self.step3_generate_prompts()
            
            # Step 4: Generate images
            images = self.step4_generate_images(prompts)
            
            # Step 5: Extract features (including DDPM)
            features = self.step5_extract_features(images)
            
            # Step 6: Train with synthetic data
            improvement = self.step6_train_with_synthetic(
                features, 
                epochs=self.config['training']['synthetic_epochs']
            )
            
            # Check improvement threshold
            if improvement['tail'] < self.config['generation']['tail_improvement_threshold']:
                print(f"\nImprovement below threshold ({improvement['tail']:.2f}% < {self.config['generation']['tail_improvement_threshold']:.1f}%)")
                print("Stopping iterative improvement.")
                # break  # Optionally continue even if small improvement
                # For now, we'll continue to see if later rounds help more, or uncomment break to be strict
        
        # Generate final report
        self._generate_final_report()
    
    def _get_average_tail_accuracy(self):
        """Calculate average accuracy across tail classes."""
        tail_accs = []
        for cls_idx in self.tail_classes:
            if cls_idx in self.metrics['tail_class_accuracies']:
                accs = self.metrics['tail_class_accuracies'][cls_idx]
                if accs:
                    tail_accs.append(accs[-1])
        
        return np.mean(tail_accs) if tail_accs else 0.0
    
    def _preprocess_image(self, image):
        """Preprocess image for feature extraction.
        
        Uses CIFAR-10 normalization to match the training data distribution.
        Resizes to 32x32 since the model was trained on CIFAR-10 (32x32).
        """
        from PIL import Image
    
        # If string path, load the image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
    
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
    
        return transform(image)
    
    def _save_checkpoint(self, epoch, accuracy, synthetic=False):
        """Save model checkpoint."""
        prefix = "synthetic_" if synthetic else ""
        checkpoint_path = os.path.join(
            self.config['paths']['checkpoint_dir'],
            f"{prefix}model_epoch{epoch}_acc{accuracy:.2f}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'config': self.config
        }, checkpoint_path)
        
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    def _save_memory_bank(self, epoch):
        """Save memory bank state."""
        memory_path = self.memory_manager.save_memory(epoch, prefix="memory_bank")
        print(f"  Memory bank saved: {memory_path}")
    
    def _save_tail_analysis(self, analysis):
        """Save tail class analysis."""
        analysis_path = os.path.join(
            self.config['paths']['results_dir'],
            f"tail_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"  Tail analysis saved: {analysis_path}")
    
    def _generate_final_report(self):
        """Generate comprehensive final report."""
        print("\n" + "="*80)
        print("GENERATING FINAL REPORT")
        print("="*80)
        
        report = {
            'configuration': self.config,
            'metrics': {
                'total_rounds': len(self.metrics['generation_rounds']),
                'synthetic_samples_generated': dict(self.metrics['synthetic_samples_generated']),
                'final_accuracies': {}
            },
            'improvements': {}
        }
        
        # Calculate improvements
        for cls_idx in self.tail_classes:
            if cls_idx in self.metrics['tail_class_accuracies']:
                accs = self.metrics['tail_class_accuracies'][cls_idx]
                if len(accs) >= 2:
                    initial = accs[0]
                    final = accs[-1]
                    improvement = final - initial
                    report['improvements'][self.class_names[cls_idx]] = {
                        'initial': initial,
                        'final': final,
                        'improvement': improvement
                    }
                    report['metrics']['final_accuracies'][self.class_names[cls_idx]] = final
        
        # Save report
        report_path = os.path.join(
            self.config['paths']['results_dir'],
            f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\nFINAL RESULTS SUMMARY:")
        print("-"*50)
        print(f"Total improvement rounds: {report['metrics']['total_rounds']}")
        print(f"Total synthetic samples generated:")
        for cls_idx, count in report['metrics']['synthetic_samples_generated'].items():
            print(f"  {self.class_names[int(cls_idx)]}: {count} samples")
        
        print("\nTail Class Improvements:")
        for cls_name, imp_data in report['improvements'].items():
            print(f"  {cls_name}: {imp_data['initial']:.2f}% → {imp_data['final']:.2f}% "
                  f"(+{imp_data['improvement']:.2f}%)")
        
        print(f"\nFull report saved to: {report_path}")
        
        # Generate visualization
        self._plot_results()
    
    def _plot_results(self):
        """Generate visualization plots."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Training progress
        ax = axes[0, 0]
        epochs = range(1, len(self.metrics['epoch_losses']) + 1)
        ax.plot(epochs, self.metrics['epoch_losses'], label='Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Progress')
        ax.legend()
        ax.grid(True)
        
        # Plot 2: Accuracy progress
        ax = axes[0, 1]
        ax.plot(epochs, self.metrics['epoch_accuracies'], label='Overall', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Validation Accuracy Progress')
        ax.legend()
        ax.grid(True)
        
        # Plot 3: Tail class accuracy evolution
        ax = axes[1, 0]
        for cls_idx in self.tail_classes:
            if cls_idx in self.metrics['tail_class_accuracies']:
                accs = self.metrics['tail_class_accuracies'][cls_idx]
                ax.plot(range(len(accs)), accs, 
                       label=self.class_names[cls_idx], marker='o')
        ax.set_xlabel('Training Iteration')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Tail Class Accuracy Evolution')
        ax.legend()
        ax.grid(True)
        
        # Plot 4: Final class distribution
        ax = axes[1, 1]
        final_accs = []
        class_labels = []
        colors = []
        
        for cls_idx in range(self.config['model']['num_classes']):
            class_labels.append(self.class_names[cls_idx])
            if cls_idx in self.tail_classes:
                colors.append('red')
                if cls_idx in self.metrics['tail_class_accuracies']:
                    accs = self.metrics['tail_class_accuracies'][cls_idx]
                    final_accs.append(accs[-1] if accs else 0)
                else:
                    final_accs.append(0)
            elif cls_idx in self.head_classes:
                colors.append('green')
                if cls_idx in self.metrics['head_class_accuracies']:
                    accs = self.metrics['head_class_accuracies'][cls_idx]
                    final_accs.append(accs[-1] if accs else 0)
                else:
                    final_accs.append(0)
            else:
                colors.append('blue')
                final_accs.append(0)
        
        bars = ax.bar(class_labels, final_accs, color=colors)
        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Final Class-wise Accuracy')
        ax.tick_params(axis='x', rotation=45)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Head Classes'),
            Patch(facecolor='blue', label='Medium Classes'),
            Patch(facecolor='red', label='Tail Classes')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(
            self.config['paths']['results_dir'],
            f"results_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(plot_path, dpi=150)
        plt.show()
        
        print(f"Visualization saved to: {plot_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Memory-Conditioned Diffusion Model Orchestrator'
    )
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (JSON)')
    parser.add_argument('--rounds', type=int, default=3,
                       help='Maximum number of improvement rounds')
    parser.add_argument('--imbalance-ratio', type=int, default=100,
                       help='Imbalance ratio for CIFAR-10-LT')
    parser.add_argument('--initial-epochs', type=int, default=20,
                       help='Number of epochs for initial training')
    parser.add_argument('--synthetic-epochs', type=int, default=10,
                       help='Number of epochs for synthetic training')
    parser.add_argument('--use-ddpm', action='store_true',
                       help='Enable DDPM-based feature generation')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use')
    
    args = parser.parse_args()
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    # Load or create configuration
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Create config from arguments
        config = None  # Will use default
    
    # Initialize orchestrator
    orchestrator = MemoryConditionedOrchestrator(config)
    
    # Override config with command-line arguments if provided
    if config is None:
        orchestrator.config['dataset']['imbalance_ratio'] = args.imbalance_ratio
        orchestrator.config['training']['initial_epochs'] = args.initial_epochs
        orchestrator.config['training']['synthetic_epochs'] = args.synthetic_epochs
        orchestrator.config['ddpm']['enabled'] = args.use_ddpm
    
    # Run the complete pipeline
    try:
        orchestrator.run_iterative_improvement(max_rounds=args.rounds)
        print("\n" + "="*80)
        print("ORCHESTRATOR COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()