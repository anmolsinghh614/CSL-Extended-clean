#!/usr/bin/env python3
"""
Quick Start Runner for Memory-Conditioned Diffusion Model Orchestrator
This script provides simple commands to run the complete pipeline
"""

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path

from pipeline_config import (get_default_config, get_test_config, merge_config,
                            memory_bank_capacity_for)
from dataloaders import get_num_classes, uses_imbalance_ratio
from dataset_configs import get_dataset_config

DATASET_CHOICES = ['cifar10', 'cifar100', 'imagenet_lt', 'inaturalist']

# Windows defaults stdout to cp1252 when it is redirected to a file, which cannot encode the
# status glyphs used below; without this, piping a run to a log file crashes on the first print.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

def get_python_executable():
    """Get the correct python executable, preferring local venv."""
    # Base path relative to this script
    base_dir = Path(__file__).parent.resolve()
    
    # Windows keeps the interpreter in Scripts/, POSIX in bin/
    relative_paths = [
        Path('.venv') / 'Scripts' / 'python.exe',
        Path('.venv') / 'bin' / 'python',
        Path('venv') / 'Scripts' / 'python.exe',
        Path('venv') / 'bin' / 'python',
    ]
    
    for root in (base_dir, base_dir.parent):
        for relative in relative_paths:
            candidate = root / relative
            if candidate.exists():
                return str(candidate)
    
    return sys.executable

PYTHON_EXE = get_python_executable()
print(f"Using Python executable: {PYTHON_EXE}")

# Verify python executable works
try:
    subprocess.run([PYTHON_EXE, '--version'], check=True, capture_output=True)
except subprocess.CalledProcessError:
    print(f"⚠️ Warning: The detected Python executable '{PYTHON_EXE}' does not seem to work.")
    print("Falling back to sys.executable")
    PYTHON_EXE = sys.executable
except Exception as e:
    print(f"⚠️ Warning: Error verifying Python executable: {e}")

def _write_config(config, config_path):
    """Persist a config next to the run so results are reproducible."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config written to: {config_path}")
    return config_path


def _dataset_config(dataset, arch=None, imbalance=None, overrides=None):
    """
    Build a config for one benchmark, with that benchmark's published training protocol.

    The class count, backbone, resolution, batch size and optimizer schedule all follow from
    the dataset, and are read from dataset_configs.py so there is one place where "how is this
    benchmark trained" is recorded. The CIFAR pair uses ResNet-32 at 32x32 for 200 epochs;
    ImageNet-LT and iNaturalist use ResNet-50 at 224 for 120 and 160. `arch` overrides the
    backbone only for deliberate deviations.

    Deriving all of it here means a run cannot be started with a model or a schedule that
    disagrees with its data.
    """
    num_classes = get_num_classes(dataset)
    protocol = get_dataset_config(dataset)

    dataset_section = {
        'name': dataset,
        'num_classes': num_classes,
        'batch_size': protocol['batch_size'],
    }
    # An imbalance ratio only means something where this code builds the imbalance. ImageNet-LT
    # and iNaturalist-2018 come long-tailed from their published splits, and the loader rejects
    # a ratio rather than pretending to honour it.
    if uses_imbalance_ratio(dataset):
        if imbalance is not None:
            dataset_section['imbalance_ratio'] = imbalance
    else:
        dataset_section['imbalance_ratio'] = None
        if imbalance is not None:
            print(f"Note: {protocol['name']} has a fixed long-tailed split, so "
                  f"--imbalance {imbalance} does not apply and is ignored.")

    backbone = arch or protocol['backbone']
    feature_dim = 2048 if backbone in ('ResNet50', 'ResNet101') else 64

    config = merge_config({
        'dataset': dataset_section,
        'model': {
            'num_classes': num_classes,
            'architecture': backbone,
        },
        # 8142 classes of 2048-dim features at the CIFAR default would reserve ~17GB of
        # reservoir, so the capacity is sized to the dataset rather than fixed.
        'memory_bank': {
            'capacity_per_class': memory_bank_capacity_for(num_classes, feature_dim),
        },
        'training': {
            'initial_epochs': protocol['epochs'],
            'lr': protocol['lr'],
            'momentum': protocol['momentum'],
            'weight_decay': protocol['weight_decay'],
            'scheduler_milestones': protocol['lr_decay_epochs'],
            'scheduler_gamma': protocol['lr_decay_factor'],
        },
    })

    return merge_config(overrides, base=config)


def run_quick_test():
    """Run a quick smoke test that exercises every stage of the pipeline."""
    print("🚀 Running Quick Test Mode...")
    print("Small-scale run to verify the pipeline works end to end.")
    print("Stable Diffusion is skipped; the feature-space DDPM path still runs.")
    
    config_path = _write_config(get_test_config(), 'test_config.json')
    return subprocess.run(
        [PYTHON_EXE, 'orchestrator.py', '--config', config_path, '--rounds', '1']
    ).returncode


def run_full_training():
    """Run full training with optimal settings on the chosen benchmark."""
    parser = argparse.ArgumentParser(description='Full training run')
    parser.add_argument('--dataset', choices=DATASET_CHOICES, default='cifar10',
                       help='Long-tailed benchmark to run on (default: cifar10)')
    parser.add_argument('--arch', choices=['ResNet32', 'ResNet34', 'ResNet50'], default=None,
                       help="Backbone; defaults to the benchmark's own (ResNet32 for CIFAR, "
                            "ResNet50 for ImageNet-LT and iNaturalist)")
    parser.add_argument('--imbalance', type=int, default=None,
                       help='Imbalance ratio for the CIFAR benchmarks (default: 100)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (default: 0)')
    args = parser.parse_args(sys.argv[2:])  # Skip 'run.py full'
    
    print("🔥 Running Full Training Mode...")
    print("This will run the complete pipeline with optimal settings.")
    
    config = _dataset_config(args.dataset, args.arch, args.imbalance)
    print(f"  Dataset: {args.dataset} ({config['model']['num_classes']} classes)")
    print(f"  Backbone: {config['model']['architecture']}")
    print(f"  Imbalance ratio: {config['dataset']['imbalance_ratio']}")
    
    config_path = _write_config(
        config, f'full_config_{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    rounds = str(config['generation']['generation_rounds'])
    return subprocess.run([
        PYTHON_EXE, 'orchestrator.py',
        '--config', config_path,
        '--rounds', rounds,
        '--gpu', str(args.gpu)
    ]).returncode


def run_custom():
    """
    Run with custom settings.

    Every tunable defaults to None and is only written into the config when it is passed, so
    an unspecified option keeps its benchmark value from pipeline_config.py. Repeating those
    values as argparse defaults is how a "custom" run silently stops being comparable to a
    "full" run: a stale default here would quietly override the protocol.
    """
    parser = argparse.ArgumentParser(description='Custom run configuration')
    
    parser.add_argument('--dataset', choices=DATASET_CHOICES, default='cifar10',
                       help='Long-tailed benchmark to run on (default: cifar10)')
    parser.add_argument('--arch', choices=['ResNet32', 'ResNet34', 'ResNet50'], default=None,
                       help='Backbone; defaults to the benchmark ResNet32 for both CIFAR '
                            'variants')
    parser.add_argument('--imbalance', type=int, default=None,
                       help='Imbalance ratio (default: the config value, 100)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Initial training epochs (default: the config value, 200)')
    parser.add_argument('--synthetic-epochs', type=int, default=None,
                       help='Synthetic training epochs (default: the config value, 25)')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Number of improvement rounds (default: the config value, 3)')
    parser.add_argument('--prompts', type=int, default=None,
                       help='Prompts per tail class (default: the config value, 50)')
    parser.add_argument('--images', type=int, default=None,
                       help='Images per prompt (default: the config value, 4)')
    parser.add_argument('--no-ddpm', dest='ddpm', action='store_false', default=None,
                       help='Disable DDPM feature generation (default: enabled)')
    parser.add_argument('--no-stable-diffusion', dest='stable_diffusion',
                       action='store_false', default=None,
                       help='Skip Stable Diffusion image synthesis (much faster)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    
    args = parser.parse_args(sys.argv[2:])  # Skip 'run.py custom'
    
    def section(**values):
        return {key: value for key, value in values.items() if value is not None}
    
    # Build a real config so custom runs get the same defaults as the full run, rather than
    # the orchestrator's bare-argument path.
    config = _dataset_config(args.dataset, args.arch, args.imbalance, overrides={
        'training': section(initial_epochs=args.epochs,
                            synthetic_epochs=args.synthetic_epochs),
        'generation': section(num_prompts_per_tail_class=args.prompts,
                              images_per_prompt=args.images,
                              generation_rounds=args.rounds,
                              use_stable_diffusion=args.stable_diffusion),
        'ddpm': section(enabled=args.ddpm),
    })
    
    training = config['training']
    generation = config['generation']
    
    print(f"🎯 Running with custom settings:")
    print(f"  Dataset: {args.dataset} ({config['model']['num_classes']} classes)")
    print(f"  Backbone: {config['model']['architecture']}")
    print(f"  Imbalance ratio: {config['dataset']['imbalance_ratio']}")
    print(f"  Initial epochs: {training['initial_epochs']}")
    print(f"  Synthetic epochs: {training['synthetic_epochs']}")
    print(f"  Improvement rounds: {generation['generation_rounds']}")
    print(f"  Prompts per class: {generation['num_prompts_per_tail_class']}")
    print(f"  Images per prompt: {generation['images_per_prompt']}")
    print(f"  DDPM enabled: {config['ddpm']['enabled']}")
    print(f"  Stable Diffusion enabled: {generation['use_stable_diffusion']}")
    print(f"  GPU: {args.gpu}")
    
    config_path = _write_config(
        config,
        f'custom_config_{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    return subprocess.run([
        PYTHON_EXE, 'orchestrator.py',
        '--config', config_path,
        '--rounds', str(generation['generation_rounds']),
        '--gpu', str(args.gpu)
    ]).returncode


def show_status():
    """Show current training status and results."""
    print("📊 Checking Training Status...")
    
    # Check for existing checkpoints
    checkpoint_dirs = ['./checkpoints', './test_checkpoints']
    memory_dirs = ['./memory_checkpoints', './test_memory']
    results_dirs = ['./results', './test_results']
    
    for dir_name, dirs in [('Checkpoints', checkpoint_dirs), 
                           ('Memory Banks', memory_dirs),
                           ('Results', results_dirs)]:
        print(f"\n{dir_name}:")
        found = False
        for d in dirs:
            if os.path.exists(d):
                files = list(Path(d).glob('*'))
                if files:
                    print(f"  {d}: {len(files)} files")
                    # Show latest files
                    latest = sorted(files, key=lambda x: x.stat().st_mtime)[-3:]
                    for f in latest:
                        print(f"    - {f.name}")
                    found = True
        
        if not found:
            print("  No files found")
    
    # Check for latest results
    for results_dir in results_dirs:
        report_path = Path(results_dir)
        if report_path.exists():
            reports = list(report_path.glob('final_report_*.json'))
            if reports:
                latest_report = max(reports, key=lambda x: x.stat().st_mtime)
                print(f"\n📈 Latest Report: {latest_report}")
                
                with open(latest_report, 'r') as f:
                    report = json.load(f)
                
                if 'summary' in report:
                    summary = report['summary']
                    print(f"\n{'Group':<12}{'Before':>12}{'After':>12}{'Change':>12}")
                    for label, key in [('Overall', 'overall'), ('Tail', 'tail_group'),
                                       ('Head', 'head_group')]:
                        print(f"{label:<12}{summary['before_synthetic'][key]:>11.2f}%"
                              f"{summary['after_synthetic'][key]:>11.2f}%"
                              f"{summary['delta'][key]:>+11.2f}")
                
                if 'improvements' in report:
                    print("\nPer-class change:")
                    for cls_name, data in report['improvements'].items():
                        print(f"  {cls_name:<12}: {data['initial']:.2f}% → {data['final']:.2f}% "
                              f"({data['improvement']:+.2f} pp)")


def clean_outputs():
    """Clean all output directories."""
    print("🧹 Cleaning output directories...")
    
    dirs_to_clean = [
        './checkpoints', './test_checkpoints',
        './memory_checkpoints', './test_memory',
        './prompts', './test_prompts',
        './synthetic_images', './test_images',
        './synthetic_features', './test_features',
        './logs', './test_logs',
        './results', './test_results'
    ]
    
    for d in dirs_to_clean:
        if os.path.exists(d):
            import shutil
            shutil.rmtree(d)
            print(f"  Removed: {d}")
    
    # Clean config files
    for config_file in Path('.').glob('*config*.json'):
        os.remove(config_file)
        print(f"  Removed: {config_file}")
    
    print("✅ Cleanup complete!")


def main():
    """Main runner function."""
    print("\n" + "="*60)
    print("MEMORY-CONDITIONED DIFFUSION MODEL - RUNNER")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python run.py test      - Run quick test (few epochs, minimal data)")
        print("  python run.py full      - Run full training (complete pipeline)")
        print("  python run.py custom    - Run with custom settings")
        print("  python run.py benchmark - Run CSL baselines on multiple datasets")
        print("  python run.py status    - Show current training status")
        print("  python run.py clean     - Clean all output directories")
        print("\nExamples:")
        print("  python run.py full --dataset cifar10  --imbalance 100")
        print("  python run.py full --dataset cifar100 --imbalance 100")
        print("  python run.py custom --dataset cifar100 --prompts 10 --images 2")
        print("  python run.py benchmark --datasets cifar10 cifar100")
        print("\n'custom' keeps every benchmark default it is not given, so it stays")
        print("comparable to 'full' unless you deliberately override something.")
        print("\nThe dataset sets the class count; both run at the benchmark protocol:")
        print("  cifar10  -> 10 classes,  32x32, ResNet32 (0.46M params)")
        print("  cifar100 -> 100 classes, 32x32, ResNet32 (0.46M params)")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    # Every training command returns the child's exit status so that an unattended run which
    # failed is reported as a failure rather than as a success.
    if command == 'test':
        return run_quick_test()
    elif command == 'full':
        return run_full_training()
    elif command == 'custom':
        return run_custom()
    elif command == 'benchmark':
        # Delegate to benchmark_runner.py with remaining args
        benchmark_args = sys.argv[2:]  # pass everything after 'benchmark'
        cmd = [PYTHON_EXE, 'benchmark_runner.py'] + benchmark_args
        print(f"Running: {' '.join(cmd)}")
        return subprocess.run(cmd).returncode
    elif command == 'status':
        show_status()
    elif command == 'clean':
        clean_outputs()
    else:
        print(f"❌ Unknown command: {command}")
        print("Valid commands: test, full, custom, benchmark, status, clean")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())