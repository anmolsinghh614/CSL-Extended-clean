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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
import os
import random
import sys
import json
import argparse
from pathlib import Path

# Windows defaults stdout to cp1252 when it is redirected to a file, which cannot encode the
# status glyphs printed below; without this, piping a run to a log file crashes on first print.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

# Import all your existing components
from pipeline_config import get_default_config, merge_config
from dataloaders import (get_lt_loaders, get_default_image_size, get_normalization,
                         get_num_classes, uses_imbalance_ratio)
from models import ResNet32, ResNet34, ResNet50
from utils import CSLLossFunc
from utils.memory_manager import MemoryManager
from utils.visual_exemplar_prompt_generator import VisualExemplarPromptGenerator

# Import Option 3 components
from option3_image_generator import Option3ImageGenerator
from phase3_feature_ddpm import FeatureDDPM
from train_feature_ddpm import FeatureDataset, train_feature_ddpm


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
        self.config = self._merge_config(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Seed before anything random happens: weight initialization, batch shuffling, DDPM
        # noise and Stable Diffusion latents all draw from the global generators. Reporting a
        # mean and a standard deviation over several runs only means anything if each run is
        # individually reproducible.
        self._seed_everything(self.config['seed'])
        
        # Resolve the input resolution up front so every consumer (backbone stem, generated
        # image preprocessing, exemplar conversion) reads a concrete number rather than the
        # "use the dataset default" placeholder.
        if self.config['dataset'].get('image_size') is None:
            self.config['dataset']['image_size'] = \
                get_default_image_size(self.config['dataset']['name'])
        
        # Initialize tracking metrics.
        # Every evaluation (initial training and synthetic fine-tuning alike) appends one
        # entry to each series, so the final report reflects the whole run rather than only
        # the initial training phase.
        self.metrics = {
            'epoch_losses': [],
            'epoch_accuracies': [],
            'class_accuracies': defaultdict(list),
            'tail_class_accuracies': defaultdict(list),
            'head_class_accuracies': defaultdict(list),
            'tail_group_accuracies': [],
            'head_group_accuracies': [],
            'stage_boundaries': [],
            'generation_rounds': [],
            'round_summaries': [],
            'synthetic_samples_generated': defaultdict(int)
        }
        
        # Snapshot of performance after initial training, before any synthetic data
        self.baseline_snapshot = None
        
        # Generated images are written under this, so two runs — and two datasets, whose tail
        # class indices can coincide — never write to the same filenames.
        self.run_id = (f"{self.config['dataset']['name']}_"
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Replaced with the dataset's real label names in step 1; placeholders until then so
        # that anything logging a class can do so before the data is loaded.
        self.class_names = [f"class_{i}" for i in range(self.config['model']['num_classes'])]
        
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
    
    @staticmethod
    def _seed_everything(seed):
        """
        Seed every generator the run draws from.

        `dataset.subset_seed` is deliberately *not* tied to this. Which images make up the
        long-tailed subset is a property of the benchmark, so it stays fixed across seeds and
        only the training randomness varies — otherwise a spread across seeds would mix
        run-to-run variance with dataset-to-dataset variance and mean neither.
        """
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _merge_config(self, config):
        """Overlay a user configuration on top of the shared defaults."""
        return merge_config(config)
    
    @staticmethod
    def _get_default_config():
        """Get default configuration for the pipeline (see pipeline_config.py)."""
        return get_default_config()
    
    def _setup_directories(self):
        """Create necessary directories."""
        for path_key, path_value in self.config['paths'].items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
        print("✓ Directories created")
    
    def step1_create_imbalanced_dataset(self):
        """Step 1: Build the long-tailed training set for the configured benchmark."""
        dataset_config = self.config['dataset']
        dataset_name = dataset_config['name']
        
        print("\n" + "="*80)
        print(f"STEP 1: CREATING IMBALANCED {dataset_name.upper()} DATASET")
        print("="*80)
        
        # Routed through dataloaders/registry.py so this pipeline and the baselines it is
        # compared against are trained on the same images with the same preprocessing. CIFAR
        # is subsampled to a chosen imbalance ratio; ImageNet-LT and iNaturalist-2018 instead
        # come long-tailed already, from their published split files.
        loader_args = {
            'dataset': dataset_name,
            'batch_size': dataset_config['batch_size'],
            'num_workers': dataset_config['num_workers'],
            'image_size': dataset_config['image_size'],
        }
        if uses_imbalance_ratio(dataset_name):
            loader_args['imbalance_ratio'] = dataset_config['imbalance_ratio']
            loader_args['data_dir'] = dataset_config['data_dir']
            loader_args['seed'] = dataset_config['subset_seed']
        else:
            loader_args['imbalance_ratio'] = None
            loader_args['data_root'] = dataset_config.get('data_root')
        
        bundle = get_lt_loaders(**loader_args)
        
        self.train_loader = bundle['train_loader']
        self.train_eval_loader = bundle['train_eval_loader']
        self.test_loader = bundle['test_loader']
        self.class_names = bundle['class_names']
        
        # Record the resolution that was actually used, since it may have been left to the
        # dataset's default. The backbone stem and generated-image preprocessing both key off
        # this, and they have to agree with the loader.
        dataset_config['image_size'] = bundle['image_size']
        
        samples_per_class = bundle['samples_per_class']
        self.samples_per_class = samples_per_class
        
        num_classes = bundle['num_classes']
        if num_classes != self.config['model']['num_classes']:
            raise ValueError(
                f"{dataset_name} has {num_classes} classes but the model is configured for "
                f"{self.config['model']['num_classes']}"
            )
        
        tail_percentile = self.config['memory_bank']['tail_threshold_percentile']
        tail_threshold = np.percentile(samples_per_class, tail_percentile)
        head_threshold = np.percentile(samples_per_class, 100 - tail_percentile)
        self.tail_classes = [i for i, count in enumerate(samples_per_class)
                             if count <= tail_threshold]
        self.head_classes = [i for i, count in enumerate(samples_per_class)
                             if count >= head_threshold]
        
        print(f"\n✓ Dataset created successfully")
        print(f"  Total samples: {sum(samples_per_class)}")
        print(f"  Input resolution: {bundle['image_size']}x{bundle['image_size']}")
        print(f"  Imbalance ratio: {max(samples_per_class) / max(1, min(samples_per_class)):.1f}"
              f" ({max(samples_per_class)} to {min(samples_per_class)} per class)")
        
        # Listing every class is unreadable on CIFAR-100, so only do it for small label sets.
        if num_classes <= 20:
            print("  Samples per class:")
            for cls_idx, count in enumerate(samples_per_class):
                group = ("TAIL" if cls_idx in self.tail_classes
                         else "HEAD" if cls_idx in self.head_classes else "MED")
                print(f"    {self.class_names[cls_idx]:10s} [{group}]: {count}")
        else:
            print(f"  Head classes: {len(self.head_classes)}, "
                  f"tail classes: {len(self.tail_classes)}, "
                  f"medium: {num_classes - len(self.head_classes) - len(self.tail_classes)}")
        
        return self.train_loader.dataset, samples_per_class
    
    def step2_train_memory_bank(self, epochs=20):
        """Step 2: Train model with memory bank on imbalanced dataset."""
        print("\n" + "="*80)
        print("STEP 2: TRAINING MEMORY BANK ON IMBALANCED DATA")
        print("="*80)
        
        # The ImageNet backbones need their stem swapped for small inputs: at 32x32 the
        # 7x7-stride-2 conv plus maxpool would shrink the image to 8x8 before the residual
        # blocks. They take image_size and adapt themselves. ResNet32 is natively a CIFAR
        # network and has no such stem.
        image_size = self.config['dataset']['image_size']
        num_classes = self.config['model']['num_classes']
        
        architectures = {'ResNet32': ResNet32, 'ResNet34': ResNet34, 'ResNet50': ResNet50}
        name = self.config['model']['architecture']
        if name not in architectures:
            raise ValueError(
                f"Unknown architecture {name!r}; expected one of {sorted(architectures)}")
        
        self.model = architectures[name](
            num_classes=num_classes, image_size=image_size).to(self.device)
        
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"  Backbone: {name} ({param_count:,} parameters, {image_size}x{image_size} input)")
        
        # Take the feature width from the backbone rather than the config: ResNet50 produces
        # 2048 and ResNet34 512 where the CIFAR ResNet-32 produces 64, and the DDPM and saved
        # report must agree with whichever is actually in use.
        self.config['model']['feature_dim'] = self._get_feature_dim()
        
        # Initialize memory manager with memory bank
        self.memory_manager = MemoryManager(
            model=self.model,
            num_classes=self.config['model']['num_classes'],
            capacity_per_class=self.config['memory_bank']['capacity_per_class'],
            alpha_base=self.config['memory_bank']['alpha_base'],
            tail_threshold_percentile=self.config['memory_bank']['tail_threshold_percentile'],
            device=self.device,
            save_dir=self.config['paths']['memory_dir'],
            tail_refresh_interval=self.config['memory_bank']['tail_refresh_interval']
        )
        
        # Initialize CSL loss with class-balanced weights
        self.criterion = CSLLossFunc(
            target_class_index=self.tail_classes,
            num_classes=self.config['model']['num_classes'],
            samples_per_class=self.samples_per_class
        ).to(self.device)
        
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
        self.metrics['stage_boundaries'].append({
            'index': len(self.metrics['epoch_accuracies']),
            'label': 'initial training'
        })
        
        for epoch in range(epochs):
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            
            # Train
            train_loss, train_acc = self._train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, class_accuracies = self._evaluate(epoch, verbose=True)
            
            # Track metrics
            self._record_evaluation(train_loss, val_acc, class_accuracies)
            
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
        
        # Freeze the pre-synthesis result so later rounds are measured against it
        self.baseline_snapshot = {
            'overall': self.metrics['epoch_accuracies'][-1] if self.metrics['epoch_accuracies'] else 0.0,
            'tail': self.metrics['tail_group_accuracies'][-1] if self.metrics['tail_group_accuracies'] else 0.0,
            'head': self.metrics['head_group_accuracies'][-1] if self.metrics['head_group_accuracies'] else 0.0,
            'per_class': {cls: accs[-1] for cls, accs in self.metrics['class_accuracies'].items() if accs}
        }
        
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
            
            # Update memory bank from the features this pass already produced
            self.memory_manager.update_memory_from_features(labels, features)
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss/total,
                'acc': 100.*correct/total
            })
        
        return train_loss/total, 100.*correct/total
    
    def _evaluate(self, epoch=0, verbose=False):
        """
        Evaluate on the balanced test set in a single pass.
        
        Returns overall accuracy and per-class accuracies together, since the tail metric is
        derived from the same predictions and does not warrant a second pass over the data.
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        num_classes = self.config['model']['num_classes']
        class_correct = torch.zeros(num_classes)
        class_total = torch.zeros(num_classes)
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                # Plain cross-entropy: the CSL criterion carries per-epoch prediction
                # counters used to steer tail weighting, and feeding test predictions into
                # them would corrupt a training signal with validation data.
                loss = F.cross_entropy(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                labels_cpu = labels.cpu()
                hits = predicted.eq(labels).cpu().float()
                class_total += torch.bincount(labels_cpu, minlength=num_classes).float()
                class_correct += torch.bincount(labels_cpu, weights=hits, minlength=num_classes)
        
        val_acc = 100. * correct / total
        class_accuracies = {
            cls: (100. * class_correct[cls] / class_total[cls]).item()
            for cls in range(num_classes) if class_total[cls] > 0
        }
        
        if verbose:
            print(f"\n  Overall Accuracy: {val_acc:.2f}%")
            print(f"  Tail group: {self._group_accuracy(class_accuracies, self.tail_classes):.2f}%"
                  f"  |  Head group: {self._group_accuracy(class_accuracies, self.head_classes):.2f}%")
            print("  Class-wise Accuracy:")
            for cls_idx in range(num_classes):
                acc = class_accuracies.get(cls_idx, 0.0)
                class_type = "TAIL" if cls_idx in self.tail_classes else "HEAD" if cls_idx in self.head_classes else "MED"
                print(f"    {self.class_names[cls_idx]:10s} [{class_type}]: {acc:.2f}%")
        
        return val_loss/total, val_acc, class_accuracies
    
    @staticmethod
    def _group_accuracy(class_accuracies, class_ids):
        """Mean per-class accuracy over a group of classes."""
        values = [class_accuracies.get(cls, 0.0) for cls in class_ids]
        return float(np.mean(values)) if values else 0.0
    
    def _record_evaluation(self, train_loss, val_acc, class_accuracies):
        """Append one evaluation point to every metric series."""
        self.metrics['epoch_losses'].append(train_loss)
        self.metrics['epoch_accuracies'].append(val_acc)
        
        for cls_idx in range(self.config['model']['num_classes']):
            self.metrics['class_accuracies'][cls_idx].append(class_accuracies.get(cls_idx, 0.0))
        
        for cls_idx in self.tail_classes:
            self.metrics['tail_class_accuracies'][cls_idx].append(class_accuracies.get(cls_idx, 0.0))
        
        for cls_idx in self.head_classes:
            self.metrics['head_class_accuracies'][cls_idx].append(class_accuracies.get(cls_idx, 0.0))
        
        self.metrics['tail_group_accuracies'].append(
            self._group_accuracy(class_accuracies, self.tail_classes))
        self.metrics['head_group_accuracies'].append(
            self._group_accuracy(class_accuracies, self.head_classes))
    
    def _replace_last_evaluation(self, val_acc, class_accuracies):
        """
        Overwrite the most recent evaluation point, keeping the recorded training loss.
        
        Used when a stage restores an earlier checkpoint: the series then ends on the model
        the stage returns instead of on the epoch that happened to run last.
        """
        if not self.metrics['epoch_accuracies']:
            return
        
        self.metrics['epoch_accuracies'][-1] = val_acc
        
        for cls_idx in range(self.config['model']['num_classes']):
            self.metrics['class_accuracies'][cls_idx][-1] = class_accuracies.get(cls_idx, 0.0)
        
        for cls_idx in self.tail_classes:
            self.metrics['tail_class_accuracies'][cls_idx][-1] = class_accuracies.get(cls_idx, 0.0)
        
        for cls_idx in self.head_classes:
            self.metrics['head_class_accuracies'][cls_idx][-1] = class_accuracies.get(cls_idx, 0.0)
        
        self.metrics['tail_group_accuracies'][-1] = \
            self._group_accuracy(class_accuracies, self.tail_classes)
        self.metrics['head_group_accuracies'][-1] = \
            self._group_accuracy(class_accuracies, self.head_classes)
    
    def step3_generate_prompts(self):
        """Step 3: Generate prompts using Option 3 (BLIP+CLIP)."""
        print("\n" + "="*80)
        print("STEP 3: GENERATING PROMPTS USING OPTION 3 (BLIP+CLIP)")
        print("="*80)
        
        # Initialize Option 3 generator once and reuse it across rounds; CLIP and BLIP are
        # several hundred megabytes each and reloading them per round is pure overhead.
        if getattr(self, 'prompt_generator', None) is None:
            # Exemplars arrive normalized with the dataset's statistics; the generator needs
            # them to undo that before BLIP and CLIP see the images.
            pixel_mean, pixel_std = get_normalization(self.config['dataset']['name'])
            self.prompt_generator = VisualExemplarPromptGenerator(
                memory_manager=self.memory_manager,
                device=str(self.device),
                pixel_mean=pixel_mean,
                pixel_std=pixel_std,
                use_blip=self.config['generation']['use_blip'],
                use_clip=self.config['generation']['use_clip'],
                cache_dir=os.path.join(self.config['paths']['logs_dir'], 'clip_cache')
            )
        
        num_exemplars = self.config['generation']['num_exemplars_per_tail_class']
        exemplars = self._select_tail_exemplar_images(num_exemplars)
        
        all_prompts = {}
        all_captions = {}
        
        for cls_idx in self.tail_classes:
            print(f"\nGenerating prompts for class {cls_idx} ({self.class_names[cls_idx]})...")
            
            class_exemplars = exemplars.get(cls_idx, [])
            prompts, captions = self.prompt_generator.generate_prompts_from_exemplars(
                class_idx=cls_idx,
                class_name=self.class_names[cls_idx],
                exemplar_images=class_exemplars,
                num_prompts=self.config['generation']['num_prompts_per_tail_class'],
                temperature=self.config['generation']['option3_temperature']
            )
            
            all_prompts[cls_idx] = prompts
            all_captions[cls_idx] = captions
            
            print(f"  Exemplars selected: {len(class_exemplars)}")
            if captions:
                print(f"  Exemplar captions: {captions[:2]}")
            print(f"  Generated {len(prompts)} prompts")
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
                'metadata': {
                    'blip_available': self.prompt_generator.captioning_available,
                    'clip_available': self.prompt_generator.ranking_available,
                    'num_exemplars_per_class': num_exemplars
                },
                'captions': {str(k): v for k, v in all_captions.items()},
                'prompts': {str(k): v for k, v in all_prompts.items()}
            }, f, indent=2)
        
        print(f"\n✓ Prompts saved to {prompts_file}")
        self.current_prompts = all_prompts
        
        return all_prompts
    
    def _select_tail_exemplar_images(self, num_exemplars=5):
        """
        Pick the real training images that best represent each tail class.
        
        The memory bank's EMA prototype defines what the model currently believes the class
        looks like; the images whose features sit closest to it are the class's visual
        exemplars. Prototype and features live in the same backbone feature space, so this
        is a direct comparison and needs no cross-space projection.
        
        Returns:
            Dict[class_id -> list of normalized image tensors]
        """
        print(f"\nSelecting {num_exemplars} visual exemplars per tail class from memory prototypes...")
        
        tail_classes = set(self.tail_classes)
        prototypes = {
            cls: F.normalize(self.memory_manager.memory_bank.get_prototype(cls).flatten(), dim=0)
            for cls in tail_classes
        }
        
        # Keep only the running best-k per class rather than every candidate
        best = {cls: [] for cls in tail_classes}
        
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(self.train_eval_loader, desc="Scoring exemplars", leave=False):
                inputs_device = inputs.to(self.device)
                _, features = self.model(inputs_device, return_features=True)
                features = F.normalize(features, dim=1)
                
                for cls in tail_classes:
                    mask = labels == cls
                    if not bool(mask.any()):
                        continue
                    
                    similarities = features[mask.to(features.device)] @ prototypes[cls]
                    class_images = inputs[mask]
                    
                    for score, image in zip(similarities.cpu().tolist(), class_images):
                        best[cls].append((score, image))
                    
                    best[cls].sort(key=lambda item: item[0], reverse=True)
                    del best[cls][num_exemplars:]
        
        return {cls: [image for _, image in entries] for cls, entries in best.items()}
    
    def step4_generate_images(self, prompts):
        """Step 4: Generate synthetic images from prompts."""
        print("\n" + "="*80)
        print("STEP 4: GENERATING SYNTHETIC IMAGES")
        print("="*80)
        
        if not self.config['generation']['use_stable_diffusion']:
            print("Stable Diffusion generation disabled; relying on feature-space synthesis only.")
            self.current_images = {}
            return {}
        
        # Initialize image generator once and reuse across rounds
        if getattr(self, 'image_generator', None) is None:
            self.image_generator = Option3ImageGenerator(
                model_type="stable_diffusion",
                device=str(self.device),
                output_dir=self.config['paths']['images_dir'],
                low_vram=self.config['generation'].get('sd_low_vram')
            )
        
        round_idx = len(self.metrics['generation_rounds'])
        images_per_prompt = self.config['generation']['images_per_prompt']
        generated_images = {}
        
        for cls_idx, class_prompts in prompts.items():
            cls_idx = int(cls_idx) if isinstance(cls_idx, str) else cls_idx
            print(f"\nGenerating images for class {cls_idx} ({self.class_names[cls_idx]})...")
            
            class_images = []
            for prompt in tqdm(class_prompts, desc="Generating"):
                images = self.image_generator.generate_batch(
                    prompts=[prompt],
                    class_idx=cls_idx,
                    num_images_per_prompt=images_per_prompt,
                    num_inference_steps=self.config['generation']['sd_inference_steps'],
                    image_size=self.config['generation']['sd_image_size'],
                    guidance_scale=self.config['generation']['sd_guidance_scale'],
                    subdir=f"{self.run_id}/round{round_idx}"
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
        
        # The synthetic features must describe the same feature space the classifier will be
        # trained in, so extraction runs with the backbone in eval mode.
        self.model.eval()
        
        # Real features are needed for DDPM training and for confidence filtering, and are
        # reused by step 6, so extract them once here.
        self.real_features = self._extract_real_features()
        
        # Extract features from generated images
        if images:
            print("\nExtracting features from synthetic images...")
            min_confidence = self.config['generation']['min_image_feature_confidence']
            
            for cls_idx, class_images in images.items():
                cls_idx = int(cls_idx) if isinstance(cls_idx, str) else cls_idx
                class_features = self._extract_features_from_image_paths(class_images)
                
                if class_features.numel() == 0:
                    print(f"  Class {cls_idx}: no features extracted")
                    continue
                
                # Diffusion images are photorealistic 512px renders, not CIFAR photos, so
                # some land far from the real class distribution. Training on those hurts the
                # class they are supposed to help, so they are filtered the same way DDPM
                # samples are.
                kept, mean_conf = self._filter_by_prototype_confidence(
                    class_features, self.real_features.get(cls_idx), min_confidence)
                
                if kept.shape[0] == 0:
                    print(f"  Class {cls_idx}: all {class_features.shape[0]} image features "
                          f"below confidence {min_confidence:.2f}, skipping")
                    continue
                
                synthetic_features[cls_idx] = kept
                print(f"  Class {cls_idx}: kept {kept.shape[0]}/{class_features.shape[0]} "
                      f"image features (mean confidence {mean_conf:.3f})")
        
        # Train DDPM for additional feature generation
        if self.config['ddpm']['enabled']:
            print("\nTraining DDPM for feature generation...")
            ddpm_features = self._train_and_generate_ddpm_features(self.real_features)
            
            # Combine features
            for cls_idx, features in ddpm_features.items():
                if cls_idx in synthetic_features:
                    synthetic_features[cls_idx] = torch.cat([
                        synthetic_features[cls_idx].cpu(), features.cpu()
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
    
    def _extract_real_features(self):
        """Extract features for every real training image, grouped by class."""
        print("\nExtracting real features (deterministic transform)...")
        num_classes = self.config['model']['num_classes']
        buckets = {i: [] for i in range(num_classes)}
        
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(self.train_eval_loader, desc="Extracting", leave=False):
                _, features = self.model(inputs.to(self.device), return_features=True)
                features = features.cpu()
                for cls_idx in labels.unique().tolist():
                    buckets[int(cls_idx)].append(features[labels == cls_idx])
        
        feature_dim = self._get_feature_dim()
        return {
            cls_idx: (torch.cat(chunks) if chunks else torch.empty(0, feature_dim))
            for cls_idx, chunks in buckets.items()
        }
    
    def _extract_features_from_image_paths(self, image_paths, batch_size=64):
        """
        Extract backbone features for a list of generated image files.
        
        Images are batched through the model rather than passed one at a time, which is
        where nearly all of this step's runtime used to go.
        """
        feature_dim = self._get_feature_dim()
        collected = []
        
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(image_paths), batch_size):
                chunk = image_paths[start:start + batch_size]
                tensors = []
                for path in chunk:
                    try:
                        tensors.append(self._preprocess_image(path))
                    except Exception as e:
                        print(f"  Warning: could not read {path}: {e}")
                
                if not tensors:
                    continue
                
                batch = torch.stack(tensors).to(self.device)
                _, features = self.model(batch, return_features=True)
                collected.append(features.cpu())
        
        if not collected:
            return torch.empty(0, feature_dim)
        return torch.cat(collected)
    
    @staticmethod
    def _filter_by_prototype_confidence(features, reference_features, min_confidence):
        """
        Drop features that sit far from a class's real feature distribution.
        
        Args:
            features: Candidate synthetic features [n, feature_dim]
            reference_features: Real features for the same class, or None to skip filtering
            min_confidence: Rescaled cosine similarity below which a sample is discarded
        
        Returns:
            (kept_features, mean_confidence_of_kept)
        """
        if reference_features is None or len(reference_features) == 0 or min_confidence <= 0:
            return features, float('nan')
        
        prototype = reference_features.mean(dim=0, keepdim=True)
        confidences = (F.cosine_similarity(features, prototype, dim=1) + 1) / 2
        keep = confidences >= min_confidence
        
        if not bool(keep.any()):
            return features[:0], float('nan')
        
        return features[keep], confidences[keep].mean().item()
    
    def _synthetic_budget(self, cls_idx):
        """
        How many synthetic features to keep for one tail class.
        
        Two bounds apply. The class should be lifted to a typical class count and no further,
        since the goal is to remove the imbalance rather than to invert it. And the amount
        added is capped as a multiple of the class's real samples, because those samples are
        all the generator ever saw — past that point extra samples mostly replicate the
        generator's own errors, and the classifier starts fitting those instead of the class.
        """
        ddpm_config = self.config['ddpm']
        real_count = self.samples_per_class[cls_idx]
        
        target = int(np.percentile(self.samples_per_class,
                                   ddpm_config['balance_target_percentile']))
        budget = max(0, target - real_count)
        budget = min(budget, int(ddpm_config['max_synthetic_ratio'] * real_count))
        
        return min(budget, int(ddpm_config['features_per_class']))
    
    def _get_feature_dim(self):
        """Feature dimension of the current backbone."""
        if hasattr(self.model, 'get_feature_dim'):
            return self.model.get_feature_dim()
        return self.config['model']['feature_dim']
    
    def _get_classifier(self):
        """Return the final linear classifier of the wrapped backbone."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'fc'):
            return self.model.model.fc
        if hasattr(self.model, 'fc'):
            return self.model.fc
        raise AttributeError("Could not find classifier (.fc) in model")
    
    def _train_and_generate_ddpm_features(self, real_features):
        """Train DDPM model and generate additional features."""
        ddpm_config = self.config['ddpm']
        
        # Create training dataset for DDPM
        all_features = []
        all_labels = []
        
        for cls_idx in self.tail_classes:
            features = real_features.get(cls_idx)
            if features is None or len(features) == 0:
                continue
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
        actual_feature_dim = self._get_feature_dim()
        ddpm_model = FeatureDDPM(
            feature_dim=actual_feature_dim,
            num_classes=self.config['model']['num_classes'],
            hidden_dim=ddpm_config['hidden_dim'],
            num_layers=ddpm_config['num_layers'],
            num_timesteps=ddpm_config['num_timesteps'],
            beta_schedule=ddpm_config['beta_schedule']
        ).to(self.device)
        
        # Fit the standardization statistics before training: the diffusion process assumes
        # roughly standard-normal data, while these features are post-ReLU activations.
        ddpm_model.set_feature_stats(all_features.to(self.device), non_negative=True)
        
        # Setup optimizer
        optimizer = optim.AdamW(ddpm_model.parameters(), lr=ddpm_config['lr'], weight_decay=0.01)
        
        # Train DDPM
        print("  Training DDPM model...")
        num_epochs = max(1, min(ddpm_config['max_epochs'],
                                ddpm_config['training_steps'] // max(1, len(ddpm_loader))))
        
        for epoch in range(num_epochs):
            avg_loss = train_feature_ddpm(ddpm_model, ddpm_loader, optimizer, self.device, epoch + 1)
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                print(f"    DDPM Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Class means in raw feature space, used to score how plausible a sample is
        prototypes = {
            cls_idx: real_features[cls_idx].mean(dim=0)
            for cls_idx in self.tail_classes
            if cls_idx in real_features and len(real_features[cls_idx]) > 0
        }
        
        # Generate features for tail classes
        print("  Generating DDPM features for tail classes...")
        ddpm_features = {}
        oversample = max(1.0, float(ddpm_config['oversample_factor']))
        min_confidence = float(ddpm_config['min_confidence'])
        
        ddpm_model.eval()
        for cls_idx in self.tail_classes:
            if cls_idx not in prototypes:
                continue
            
            keep_per_class = self._synthetic_budget(cls_idx)
            if keep_per_class <= 0:
                print(f"    Class {cls_idx}: already at the balance target, no features needed")
                continue
            
            num_to_generate = int(keep_per_class * oversample)
            class_label = torch.full((num_to_generate,), cls_idx, dtype=torch.long, device=self.device)
            
            generated, confidences = ddpm_model.sample_with_confidence(
                class_ids=class_label,
                class_prototypes=prototypes,
                device=self.device,
                num_steps=ddpm_config['sampling_steps'],
                batch_size=ddpm_config['sampling_batch_size']
            )
            
            # Keep the most prototype-consistent samples
            order = torch.argsort(confidences, descending=True)
            keep = order[:keep_per_class]
            if min_confidence > 0:
                keep = keep[confidences[keep] >= min_confidence]
            
            if len(keep) == 0:
                print(f"    Class {cls_idx}: all samples below confidence {min_confidence:.2f}, skipping")
                continue
            
            ddpm_features[cls_idx] = generated[keep].detach().cpu()
            print(f"    Class {cls_idx}: kept {len(keep)}/{num_to_generate} features "
                  f"for {self.samples_per_class[cls_idx]} real samples "
                  f"(mean confidence {confidences[keep].mean().item():.3f})")
            self.metrics['synthetic_samples_generated'][cls_idx] += int(len(keep))
        
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
                'hidden_dim': ddpm_config['hidden_dim'],
                'num_layers': ddpm_config['num_layers'],
                'num_timesteps': ddpm_config['num_timesteps'],
                'beta_schedule': ddpm_config['beta_schedule']
            }
        }, ddpm_path)
        
        return ddpm_features
    
    def step6_train_with_synthetic(self, synthetic_features, epochs=5):
        """Step 6: Train model with real data + synthetic features."""
        print("\n" + "="*80)
        print("STEP 6: TRAINING WITH SYNTHETIC DATA")
        print("="*80)
        
        training_config = self.config['training']
        num_classes = self.config['model']['num_classes']
        
        # Baseline is measured on the current model, so improvements are attributable to
        # this round rather than to whatever the previous round left behind.
        _, baseline_acc, baseline_class_acc = self._evaluate()
        baseline_tail_acc = self._group_accuracy(baseline_class_acc, self.tail_classes)
        baseline_head_acc = self._group_accuracy(baseline_class_acc, self.head_classes)
        
        print(f"\nBaseline Performance:")
        print(f"  Overall Accuracy: {baseline_acc:.2f}%")
        print(f"  Tail Classes Accuracy: {baseline_tail_acc:.2f}%")
        print(f"  Head Classes Accuracy: {baseline_head_acc:.2f}%")
        
        print("\nPreparing augmented training data...")
        synthetic_datasets = []
        synthetic_counts = defaultdict(int)
        
        for cls_idx, features in synthetic_features.items():
            cls_idx = int(cls_idx)
            if features.shape[0] == 0:
                continue
            labels = torch.full((features.shape[0],), cls_idx, dtype=torch.long)
            synthetic_datasets.append(TensorDataset(features.float(), labels))
            synthetic_counts[cls_idx] += features.shape[0]
            print(f"  Class {cls_idx}: {features.shape[0]} synthetic samples")
        
        total_synthetic = sum(synthetic_counts.values())
        if not synthetic_datasets:
            print("⚠ No synthetic features available, skipping augmented training")
            return {
                'overall': 0.0, 'tail': 0.0, 'head': 0.0,
                'baseline_overall': baseline_acc, 'baseline_tail': baseline_tail_acc,
                'baseline_head': baseline_head_acc,
                'augmented_overall': baseline_acc, 'augmented_tail': baseline_tail_acc,
                'augmented_head': baseline_head_acc,
                'synthetic_samples': 0
            }
        
        combined_synthetic = ConcatDataset(synthetic_datasets)
        print(f"\nTotal synthetic samples: {total_synthetic}")
        print(f"Total real samples: {len(self.train_loader.dataset)}")
        
        freeze_backbone = training_config['freeze_backbone_in_synthetic']
        use_precomputed = freeze_backbone and training_config['precompute_real_features']
        
        if use_precomputed:
            augmented_loader = self._create_feature_dataloader(combined_synthetic)
            print("Training mode: classifier retraining on precomputed real + synthetic features")
        else:
            augmented_loader = self._create_hybrid_dataloader(combined_synthetic)
            print("Training mode: hybrid batches of real images and synthetic features")
        
        # Restrict optimization to the classifier when the backbone is frozen. Synthetic
        # features were produced by the current backbone; letting it drift would leave them
        # describing a feature space that no longer exists.
        classifier = self._get_classifier()
        for param in self.model.parameters():
            param.requires_grad = not freeze_backbone
        for param in classifier.parameters():
            param.requires_grad = True
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}"
              f"{' (classifier only)' if freeze_backbone else ''}")
        
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=training_config['lr'] * training_config['synthetic_lr_scale'],
            momentum=training_config['momentum'],
            weight_decay=training_config['weight_decay']
        )
        
        # Reweight CSL using the augmented distribution so tail classes are not corrected
        # twice over (extra samples plus an inverse-frequency loss weight).
        if training_config['reweight_csl_for_synthetic']:
            effective_counts = [
                self.samples_per_class[c] + synthetic_counts.get(c, 0) for c in range(num_classes)
            ]
        else:
            effective_counts = self.samples_per_class
        
        criterion = CSLLossFunc(
            target_class_index=self.tail_classes,
            num_classes=num_classes,
            samples_per_class=effective_counts
        ).to(self.device)
        
        print(f"\nTraining with augmented data for {epochs} epochs...")
        best_tail_acc = baseline_tail_acc
        best_state = None
        best_epoch = None
        epochs_without_gain = 0
        patience = training_config['synthetic_patience']
        
        self.metrics['stage_boundaries'].append({
            'index': len(self.metrics['epoch_accuracies']),
            'label': f"round {len(self.metrics['generation_rounds'])} synthetic"
        })
        
        for epoch in range(epochs):
            # A frozen backbone must stay in eval mode: updating BatchNorm statistics would
            # shift the feature space the synthetic features were extracted from.
            self.model.eval() if freeze_backbone else self.model.train()
            
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(augmented_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
            
            for batch_idx, (data, labels) in enumerate(pbar):
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                if data.dim() == 4:  # Images (B, C, H, W)
                    outputs, feature_data = self.model(data, return_features=True)
                else:  # Features (B, D) bypass the backbone and go straight to the classifier
                    if data.dim() > 2:
                        data = data.view(data.size(0), -1)
                    outputs = classifier(data)
                    feature_data = data
                
                loss = criterion(labels, outputs, epoch)
                loss.backward()
                optimizer.step()
                
                self.memory_manager.update_memory_from_features(labels, feature_data.detach())
                
                epoch_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{epoch_loss/total:.3f}',
                    'acc': f'{100.*correct/total:.1f}%'
                })
            
            # Validation after each epoch
            _, val_acc, class_accuracies = self._evaluate(epoch)
            tail_acc = self._group_accuracy(class_accuracies, self.tail_classes)
            head_acc = self._group_accuracy(class_accuracies, self.head_classes)
            
            # Recorded into the same series as initial training, so the final report and the
            # next round's baseline both see the effect of synthetic data.
            self._record_evaluation(epoch_loss / max(1, total), val_acc, class_accuracies)
            
            print(f"\n  Epoch {epoch+1} Results:")
            print(f"    Training Accuracy: {100.*correct/total:.2f}%")
            print(f"    Validation Accuracy: {val_acc:.2f}% (baseline {baseline_acc:.2f}%)")
            print(f"    Tail Classes Accuracy: {tail_acc:.2f}% (baseline {baseline_tail_acc:.2f}%)")
            print(f"    Head Classes Accuracy: {head_acc:.2f}% (baseline {baseline_head_acc:.2f}%)")
            
            # The point of this stage is tail accuracy, so checkpoint on that
            if tail_acc > best_tail_acc:
                best_tail_acc = tail_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                epochs_without_gain = 0
                
                checkpoint_path = os.path.join(
                    self.config['paths']['checkpoint_dir'],
                    f"model_synthetic_round{len(self.metrics['generation_rounds'])}_tail{tail_acc:.2f}.pt"
                )
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'accuracy': val_acc,
                    'tail_accuracy': tail_acc
                }, checkpoint_path)
                print(f"    ✓ Checkpoint saved: {checkpoint_path}")
            else:
                epochs_without_gain += 1
                if epochs_without_gain >= patience:
                    print(f"    Tail accuracy has not improved for {patience} epochs, "
                          f"stopping the synthetic stage early")
                    break
        
        # Leave the model fully trainable for any later stage
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Return the best-tail model rather than whatever the last epoch produced. Continued
        # training on the augmented set keeps improving training accuracy while tail accuracy
        # falls, so the last epoch is systematically worse than the one selected here.
        if training_config['restore_best_synthetic'] and best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"\nRestored best model from synthetic epoch {best_epoch + 1} "
                  f"(tail {best_tail_acc:.2f}%)")
        
        # Final evaluation
        _, final_acc, final_class_acc = self._evaluate()
        final_tail_acc = self._group_accuracy(final_class_acc, self.tail_classes)
        final_head_acc = self._group_accuracy(final_class_acc, self.head_classes)
        
        # The metrics series has to end on the model this stage actually returns, otherwise
        # the report and plots describe a model that was discarded.
        self._replace_last_evaluation(final_acc, final_class_acc)
        
        improvement = {
            'overall': final_acc - baseline_acc,
            'tail': final_tail_acc - baseline_tail_acc,
            'head': final_head_acc - baseline_head_acc,
            'baseline_overall': baseline_acc,
            'baseline_tail': baseline_tail_acc,
            'baseline_head': baseline_head_acc,
            'augmented_overall': final_acc,
            'augmented_tail': final_tail_acc,
            'augmented_head': final_head_acc,
            'synthetic_samples': total_synthetic
        }
        
        print("\n" + "="*80)
        print("TRAINING RESULTS")
        print("="*80)
        print(f"Overall Accuracy Improvement: {improvement['overall']:+.2f} pp")
        print(f"Tail Classes Improvement: {improvement['tail']:+.2f} pp")
        print(f"Head Classes Change: {improvement['head']:+.2f} pp")
        print(f"\nFinal Performance:")
        print(f"  Overall: {final_acc:.2f}%")
        print(f"  Tail Classes: {final_tail_acc:.2f}%")
        print(f"  Head Classes: {final_head_acc:.2f}%")
        
        return improvement

    def _create_feature_dataloader(self, synthetic_dataset):
        """
        Build a feature-only loader combining precomputed real features with synthetic ones.
        
        Valid only while the backbone is frozen: real features are then constant, so there is
        nothing to gain from re-running the backbone on augmented images every epoch.
        """
        real_features = getattr(self, 'real_features', None)
        if real_features is None:
            real_features = self._extract_real_features()
            self.real_features = real_features
        
        feature_chunks = []
        label_chunks = []
        for cls_idx, features in real_features.items():
            if len(features) == 0:
                continue
            feature_chunks.append(features.float())
            label_chunks.append(torch.full((len(features),), int(cls_idx), dtype=torch.long))
        
        real_dataset = TensorDataset(torch.cat(feature_chunks), torch.cat(label_chunks))
        combined = ConcatDataset([real_dataset, synthetic_dataset])
        
        print(f"Augmented dataset size: {len(combined)} feature vectors")
        
        return DataLoader(
            combined,
            batch_size=self.config['dataset']['batch_size'],
            shuffle=True,
            num_workers=0  # Tensors already live in memory; workers would only add overhead
        )

    def _create_hybrid_dataloader(self, synthetic_dataset):
        """Create dataloader that combines real images with synthetic features using homogeneous batches."""
        from torch.utils.data import Sampler
        import random

        # Custom Sampler to ensure batches only contain EITHER images OR features
        class HomogeneousBatchSampler(Sampler):
            def __init__(self, data_source, batch_size, split_idx):
                self.data_source = data_source
                self.batch_size = batch_size
                self.split_idx = split_idx
                
                self.num_real = split_idx
                self.num_synthetic = len(data_source) - split_idx

            def __iter__(self):
                real_indices = torch.randperm(self.num_real).tolist()
                syn_indices = [x + self.split_idx for x in torch.randperm(self.num_synthetic).tolist()]
                
                batches = []
                for i in range(0, len(real_indices), self.batch_size):
                    batches.append(real_indices[i:i + self.batch_size])
                
                for i in range(0, len(syn_indices), self.batch_size):
                    batches.append(syn_indices[i:i + self.batch_size])
                
                # Interleave real and synthetic batches
                random.shuffle(batches)
                
                for batch in batches:
                    yield batch

            def __len__(self):
                return (self.num_real + self.batch_size - 1) // self.batch_size + \
                       (self.num_synthetic + self.batch_size - 1) // self.batch_size

        # Combine real dataset with synthetic features dataset
        combined = ConcatDataset([self.train_loader.dataset, synthetic_dataset])
        split_idx = len(self.train_loader.dataset)
        
        batch_sampler = HomogeneousBatchSampler(combined, self.config['dataset']['batch_size'], split_idx)
        
        print(f"Augmented dataset size: {len(combined)}")
        
        return DataLoader(
            combined,
            batch_sampler=batch_sampler,
            num_workers=self.config['dataset']['num_workers']
        )
    
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
        threshold = self.config['generation']['tail_improvement_threshold']
        early_stop = self.config['generation']['early_stop_on_low_improvement']
        
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
            
            improvement['round'] = round_idx
            self.metrics['round_summaries'].append(improvement)
            
            # Check improvement threshold
            if improvement['tail'] < threshold:
                print(f"\nTail improvement below threshold "
                      f"({improvement['tail']:+.2f} pp < {threshold:.1f} pp)")
                if early_stop:
                    print("Stopping iterative improvement.")
                    break
                print("Continuing anyway (early_stop_on_low_improvement is disabled).")
        
        # Generate final report
        self._generate_final_report()
    
    def _get_average_tail_accuracy(self):
        """Calculate average accuracy across tail classes."""
        if self.metrics['tail_group_accuracies']:
            return self.metrics['tail_group_accuracies'][-1]
        return 0.0
    
    def _preprocess_image(self, image):
        """
        Preprocess a generated image so the backbone sees it exactly as training data.
        
        Diffusion output is much larger than the training resolution, so it is resized and
        normalized with the same statistics as the dataset; feeding it in any other scale
        would put its features in a different part of the space from the real ones.
        """
        from PIL import Image
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        image_size = self.config['dataset']['image_size']
        mean, std = get_normalization(self.config['dataset']['name'])
        
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        return transform(image)
    
    def _save_checkpoint(self, epoch, accuracy, synthetic=False):
        """
        Save the model checkpoint, keeping only the most recent of each kind.

        Validation accuracy improves many times over a long run, and each ResNet-50 checkpoint
        is ~100MB, so retaining every improvement filled the disk on the large benchmarks. Only
        the latest is kept, which is all the pipeline ever reloads. A write failure warns rather
        than aborting: losing a checkpoint must not throw away hours of completed training.
        """
        prefix = "synthetic_" if synthetic else ""
        checkpoint_path = os.path.join(
            self.config['paths']['checkpoint_dir'],
            f"{prefix}model_epoch{epoch}_acc{accuracy:.2f}.pt"
        )
        
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'accuracy': accuracy,
                'config': self.config
            }, checkpoint_path)
        except OSError as error:
            print(f"  Warning: could not save checkpoint ({error}); continuing training")
            return
        
        # Delete the previously kept checkpoint only now that the new one is safely written, so
        # an interrupted save never leaves zero checkpoints on disk.
        if not hasattr(self, '_last_checkpoint_paths'):
            self._last_checkpoint_paths = {}
        previous = self._last_checkpoint_paths.get(prefix)
        if previous and previous != checkpoint_path and os.path.exists(previous):
            try:
                os.remove(previous)
            except OSError:
                pass
        self._last_checkpoint_paths[prefix] = checkpoint_path
        
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    @staticmethod
    def _remove_checkpoint_group(pt_path):
        """Delete a memory-bank .pt along with the summary/stats JSON written beside it."""
        for path in (pt_path, pt_path.replace('.pt', '_summary.json'),
                     pt_path.replace('.pt', '_stats.json')):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
    
    def _save_memory_bank(self, epoch):
        """
        Save the memory bank, keeping only the most recent.

        Each save is num_classes x capacity x feature_dim floats — about 1.6GB on
        iNaturalist-2018 — and the 5-epoch cadence over a long run accumulates enough of these
        to fill the disk. The intermediate banks are diagnostic and never reloaded mid-run, so
        only the latest is retained.
        """
        try:
            memory_path = self.memory_manager.save_memory(epoch, prefix="memory_bank")
        except OSError as error:
            print(f"  Warning: could not save memory bank ({error}); continuing training")
            return
        
        previous = getattr(self, '_last_memory_path', None)
        if previous and previous != memory_path:
            self._remove_checkpoint_group(previous)
        self._last_memory_path = memory_path
        
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
        
        baseline = self.baseline_snapshot or {}
        baseline_per_class = baseline.get('per_class', {})
        
        final_overall = self.metrics['epoch_accuracies'][-1] if self.metrics['epoch_accuracies'] else 0.0
        final_tail = self.metrics['tail_group_accuracies'][-1] if self.metrics['tail_group_accuracies'] else 0.0
        final_head = self.metrics['head_group_accuracies'][-1] if self.metrics['head_group_accuracies'] else 0.0
        
        report = {
            'configuration': self.config,
            'metrics': {
                'total_rounds': len(self.metrics['generation_rounds']),
                'synthetic_samples_generated': {
                    str(k): int(v) for k, v in self.metrics['synthetic_samples_generated'].items()
                },
                'final_accuracies': {},
                'samples_per_class': {
                    self.class_names[i]: int(c) for i, c in enumerate(self.samples_per_class)
                }
            },
            # Everything is measured against the state of the model after initial training and
            # before any synthetic data, which is the comparison the method is claiming.
            'summary': {
                'before_synthetic': {
                    'overall': baseline.get('overall', 0.0),
                    'tail_group': baseline.get('tail', 0.0),
                    'head_group': baseline.get('head', 0.0)
                },
                'after_synthetic': {
                    'overall': final_overall,
                    'tail_group': final_tail,
                    'head_group': final_head
                },
                'delta': {
                    'overall': final_overall - baseline.get('overall', 0.0),
                    'tail_group': final_tail - baseline.get('tail', 0.0),
                    'head_group': final_head - baseline.get('head', 0.0)
                }
            },
            'round_summaries': self.metrics['round_summaries'],
            'improvements': {}
        }
        
        # Per-class improvements over the pre-synthesis baseline
        for cls_idx in range(self.config['model']['num_classes']):
            accs = self.metrics['class_accuracies'].get(cls_idx, [])
            if not accs:
                continue
            
            initial = baseline_per_class.get(cls_idx, accs[0])
            final = accs[-1]
            group = 'tail' if cls_idx in self.tail_classes else 'head' if cls_idx in self.head_classes else 'medium'
            
            report['improvements'][self.class_names[cls_idx]] = {
                'group': group,
                'initial': initial,
                'final': final,
                'improvement': final - initial
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
        print("-"*66)
        print(f"Total improvement rounds: {report['metrics']['total_rounds']}")
        if self.metrics['synthetic_samples_generated']:
            print("Synthetic samples generated:")
            for cls_idx, count in sorted(self.metrics['synthetic_samples_generated'].items()):
                print(f"  {self.class_names[int(cls_idx)]}: {count} samples")
        
        summary = report['summary']
        print(f"\n{'Group':<12}{'Before':>12}{'After':>12}{'Change':>12}")
        for label, key in [('Overall', 'overall'), ('Tail', 'tail_group'), ('Head', 'head_group')]:
            print(f"{label:<12}{summary['before_synthetic'][key]:>11.2f}%"
                  f"{summary['after_synthetic'][key]:>11.2f}%"
                  f"{summary['delta'][key]:>+11.2f}")
        
        print("\nPer-class change (before → after synthetic training):")
        for cls_name, data in report['improvements'].items():
            print(f"  {cls_name:<12} [{data['group']:>6}]: "
                  f"{data['initial']:6.2f}% → {data['final']:6.2f}% ({data['improvement']:+.2f} pp)")
        
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
        
        # Plot 2: Overall vs group accuracy, with stage boundaries marked
        ax = axes[0, 1]
        ax.plot(epochs, self.metrics['epoch_accuracies'], label='Overall', linewidth=2)
        if self.metrics['tail_group_accuracies']:
            ax.plot(epochs, self.metrics['tail_group_accuracies'], label='Tail group', linewidth=2)
        if self.metrics['head_group_accuracies']:
            ax.plot(epochs, self.metrics['head_group_accuracies'], label='Head group', linewidth=2)
        self._mark_stages(ax)
        ax.set_xlabel('Evaluation')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Validation Accuracy Progress')
        ax.legend()
        ax.grid(True)
        
        # Plot 3: Tail class accuracy evolution
        ax = axes[1, 0]
        for cls_idx in self.tail_classes:
            accs = self.metrics['class_accuracies'].get(cls_idx, [])
            if accs:
                ax.plot(range(1, len(accs) + 1), accs, label=self.class_names[cls_idx])
        self._mark_stages(ax)
        ax.set_xlabel('Evaluation')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Tail Class Accuracy Evolution')
        ax.legend()
        ax.grid(True)
        
        # Plot 4: Per-class accuracy before vs after synthetic training
        ax = axes[1, 1]
        num_classes = self.config['model']['num_classes']
        baseline_per_class = (self.baseline_snapshot or {}).get('per_class', {})
        
        class_labels = [self.class_names[i] for i in range(num_classes)]
        before = []
        after = []
        colors = []
        
        for cls_idx in range(num_classes):
            accs = self.metrics['class_accuracies'].get(cls_idx, [])
            final = accs[-1] if accs else 0.0
            after.append(final)
            before.append(baseline_per_class.get(cls_idx, accs[0] if accs else 0.0))
            colors.append('red' if cls_idx in self.tail_classes
                          else 'green' if cls_idx in self.head_classes else 'blue')
        
        positions = np.arange(num_classes)
        ax.bar(positions - 0.2, before, width=0.4, color='lightgray', label='Before synthetic')
        ax.bar(positions + 0.2, after, width=0.4, color=colors, label='After synthetic')
        ax.set_xticks(positions)
        ax.set_xticklabels(class_labels, rotation=45, ha='right')
        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Class-wise Accuracy: Before vs After Synthetic Training')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgray', label='Before synthetic'),
            Patch(facecolor='green', label='Head (after)'),
            Patch(facecolor='blue', label='Medium (after)'),
            Patch(facecolor='red', label='Tail (after)')
        ]
        ax.legend(handles=legend_elements, fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(
            self.config['paths']['results_dir'],
            f"results_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        
        print(f"Visualization saved to: {plot_path}")
    
    def _mark_stages(self, ax):
        """Draw vertical markers where the pipeline switched stages."""
        for boundary in self.metrics['stage_boundaries']:
            if boundary['index'] <= 0:
                continue
            ax.axvline(boundary['index'] + 0.5, color='gray', linestyle='--', alpha=0.6)
            ax.annotate(boundary['label'], xy=(boundary['index'] + 0.5, ax.get_ylim()[0]),
                        rotation=90, fontsize=7, va='bottom', ha='right', color='gray')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Memory-Conditioned Diffusion Model Orchestrator'
    )
    
    # Defaults are None so that "not passed" is distinguishable from "passed the default
    # value". Anything explicitly passed overrides the config file; anything omitted leaves
    # the file (or the built-in default) alone.
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (JSON)')
    parser.add_argument('--rounds', type=int, default=3,
                       help='Maximum number of improvement rounds')
    parser.add_argument('--dataset',
                       choices=['cifar10', 'cifar100', 'imagenet_lt', 'inaturalist'],
                       default=None, help='Long-tailed benchmark to run on')
    parser.add_argument('--imbalance-ratio', type=int, default=None,
                       help='Imbalance ratio (n_max / n_min) of the long-tailed training set')
    parser.add_argument('--initial-epochs', type=int, default=None,
                       help='Number of epochs for initial training')
    parser.add_argument('--synthetic-epochs', type=int, default=None,
                       help='Number of epochs for synthetic training')
    parser.add_argument('--use-ddpm', dest='use_ddpm', action='store_true', default=None,
                       help='Enable DDPM-based feature generation')
    parser.add_argument('--no-ddpm', dest='use_ddpm', action='store_false',
                       help='Disable DDPM-based feature generation')
    parser.add_argument('--no-stable-diffusion', dest='use_sd', action='store_false', default=None,
                       help='Skip Stable Diffusion image synthesis (feature-space DDPM only)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for weight init and shuffling (default: config value)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use')
    
    args = parser.parse_args()
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    # Load configuration file if given; missing keys fall back to the defaults
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Selecting a dataset also fixes the class count, so apply it before the model is built
    # rather than through the override loop below.
    if args.dataset:
        num_classes = get_num_classes(args.dataset)
        # Resolution is cleared along with the name: a config saved for one dataset records a
        # concrete image_size, which would otherwise be carried over to the other one.
        config = merge_config({
            'dataset': {'name': args.dataset, 'num_classes': num_classes,
                        'image_size': None},
            'model': {'num_classes': num_classes}
        }, base=merge_config(config))
        print(f"Config override: dataset.name = {args.dataset} ({num_classes} classes)")
    
    # The seed has to be in place before construction, since the orchestrator seeds the global
    # generators there — applying it through the override loop below would be too late.
    if args.seed is not None:
        config = merge_config({'seed': args.seed}, base=merge_config(config))
        print(f"Config override: seed = {args.seed}")
    
    # Initialize orchestrator
    orchestrator = MemoryConditionedOrchestrator(config)
    
    # Apply explicit command-line overrides
    overrides = [
        ('dataset', 'imbalance_ratio', args.imbalance_ratio),
        ('training', 'initial_epochs', args.initial_epochs),
        ('training', 'synthetic_epochs', args.synthetic_epochs),
        ('ddpm', 'enabled', args.use_ddpm),
        ('generation', 'use_stable_diffusion', args.use_sd),
    ]
    for section, key, value in overrides:
        if value is not None:
            orchestrator.config[section][key] = value
            print(f"Config override: {section}.{key} = {value}")
    
    # Run the complete pipeline. The exit status has to reflect what happened: a long run
    # launched from a script and left unattended must not report success after failing.
    try:
        orchestrator.run_iterative_improvement(max_rounds=args.rounds)
        print("\n" + "="*80)
        print("ORCHESTRATOR COMPLETED SUCCESSFULLY!")
        print("="*80)
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 130
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())