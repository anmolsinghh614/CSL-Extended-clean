"""
Single source of truth for pipeline configuration.

Both `orchestrator.py` and `run.py` build their configs from here so that a tuned
hyperparameter only ever has to be changed in one place. This module deliberately has no
heavy imports so it can be read by tooling without pulling in torch.
"""

import copy


def get_default_config():
    """Default configuration for the full pipeline."""
    return {
        # Dataset configuration
        'dataset': {
            # 'cifar10' or 'cifar100'; both go through dataloaders/cifar_lt_loader.py so the
            # long-tail profile matches the CSL benchmark this work is compared against.
            'name': 'cifar10',
            'imbalance_ratio': 100,      # n_max / n_min, the standard CIFAR-LT setting
            'num_classes': 10,
            'batch_size': 128,
            'num_workers': 4,
            'data_dir': './data',
            # None takes the resolution the CSL baseline used for the chosen dataset: 32 for
            # CIFAR-10, 224 for CIFAR-100. Set it explicitly only to deviate from that.
            'image_size': None,
            'subset_seed': 42            # Fixes which images each class keeps
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
            'tail_refresh_interval': 512  # Samples between tail/head reclassifications
        },

        # Training configuration
        'training': {
            'initial_epochs': 200,        # ResNet-34 needs ~200 epochs on CIFAR-10
            'synthetic_epochs': 25,       # Enough to integrate synthetic features
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'scheduler_milestones': [160, 180],  # Decay at 80% and 90% of total
            'scheduler_gamma': 0.1,
            'synthetic_lr_scale': 0.1,    # Fine-tuning LR relative to the base LR
            # Synthetic features are vectors in the feature space of the frozen backbone.
            # If the backbone keeps training they stop describing anything the classifier
            # sees, so the synthetic stage retrains the classifier only.
            'freeze_backbone_in_synthetic': True,
            # With the backbone frozen, real features are constant, so extracting them once
            # is equivalent to (and far cheaper than) re-running the backbone each epoch.
            'precompute_real_features': True,
            # Reweight the CSL loss using augmented class counts during the synthetic stage,
            # so tail classes are not corrected twice (extra samples *and* a larger loss
            # weight), which costs head accuracy.
            'reweight_csl_for_synthetic': True,
            # Tail accuracy peaks early in the synthetic stage and then declines as the
            # classifier fits the generator's approximation of the tail instead of the real
            # tail. Keep the best-tail epoch and stop once it stops improving, so the stage
            # returns the model it found rather than the last one it happened to produce.
            'restore_best_synthetic': True,
            'synthetic_patience': 5
        },

        # Synthetic generation configuration
        'generation': {
            'num_prompts_per_tail_class': 50,
            'images_per_prompt': 4,
            'generation_rounds': 3,
            'tail_improvement_threshold': 2.0,  # Percentage points; below this, stop early
            'early_stop_on_low_improvement': True,
            'option3_temperature': 0.8,
            'use_blip': True,
            'use_clip': True,
            'num_exemplars_per_tail_class': 5,
            # Stable Diffusion is the expensive half of the synthesis. Disabling it leaves
            # the feature-space DDPM path, which is fast enough to iterate on.
            'use_stable_diffusion': True,
            'sd_inference_steps': 30,
            'sd_image_size': 512,
            'sd_guidance_scale': 7.5,
            # Diffusion renders that land far from the real class distribution are discarded
            # rather than trained on. Same scale as the DDPM confidence.
            'min_image_feature_confidence': 0.5
        },

        # DDPM configuration
        'ddpm': {
            'enabled': True,
            'num_timesteps': 1000,
            'beta_schedule': 'cosine',
            'hidden_dim': 1024,           # 2x feature_dim for representational capacity
            'num_layers': 4,
            'training_steps': 10000,      # More training = better feature quality
            'max_epochs': 50,
            'lr': 1e-4,
            'features_per_class': 300,    # Upper bound on synthetic features per tail class
            'sampling_steps': 200,        # Strided reverse diffusion; None uses all steps
            # Generate extra samples and keep only the ones closest to the class prototype,
            # so off-manifold samples are not trained on.
            'oversample_factor': 1.5,
            'min_confidence': 0.5,
            'sampling_batch_size': 256,
            # Augmentation aims to bring each tail class up to a typical class count, not
            # past it: this percentile of the real class counts is the target size.
            'balance_target_percentile': 50.0,
            # ...but never add more than this multiple of a class's real samples. The
            # generator only ever saw those real samples, so the amount of genuinely new
            # information it can supply is bounded by them. Without this cap a 50-sample
            # class ends up ~90% synthetic and the classifier fits the generator's errors.
            'max_synthetic_ratio': 3.0
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


def merge_config(overrides, base=None):
    """
    Overlay a partial configuration on top of the defaults, section by section.

    A config file therefore only needs to list the keys it wants to change; a file written
    before a new option existed keeps working and picks up that option's default.
    """
    merged = copy.deepcopy(base) if base is not None else get_default_config()
    if not overrides:
        return merged

    for section, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(section), dict):
            merged[section].update(value)
        else:
            merged[section] = value

    return merged


def get_test_config():
    """
    Fast smoke-test configuration.

    Sized to complete on CPU: Stable Diffusion is skipped (minutes per image without a GPU)
    while the feature-space DDPM path, which is the part that drives tail accuracy, stays on.
    """
    return merge_config({
        'dataset': {'imbalance_ratio': 10, 'batch_size': 64, 'num_workers': 2},
        'memory_bank': {'capacity_per_class': 64, 'tail_refresh_interval': 256},
        'training': {
            'initial_epochs': 3,
            'synthetic_epochs': 3,
            'scheduler_milestones': [2],
        },
        'generation': {
            'num_prompts_per_tail_class': 5,
            'images_per_prompt': 2,
            'generation_rounds': 1,
            'tail_improvement_threshold': 0.5,
            'use_stable_diffusion': False,
            'sd_inference_steps': 20,
        },
        'ddpm': {
            'enabled': True,
            'num_timesteps': 200,
            'hidden_dim': 256,
            'num_layers': 3,
            'training_steps': 600,
            'max_epochs': 15,
            'features_per_class': 200,
            'sampling_steps': 50,
            'min_confidence': 0.4,
        },
        'paths': {
            'checkpoint_dir': './test_checkpoints',
            'memory_dir': './test_memory',
            'prompts_dir': './test_prompts',
            'images_dir': './test_images',
            'features_dir': './test_features',
            'logs_dir': './test_logs',
            'results_dir': './test_results'
        }
    })
