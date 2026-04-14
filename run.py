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

def get_python_executable():
    """Get the correct python executable, preferring local venv."""
    # Base path relative to this script
    base_dir = Path(__file__).parent.resolve()
    
    # Check for local venv in current directory
    local_venv = base_dir / '.venv' / 'Scripts' / 'python.exe'
    if local_venv.exists():
        return str(local_venv)
    
    # Check for venv in parent directory
    parent_venv = base_dir.parent / '.venv' / 'Scripts' / 'python.exe'
    if parent_venv.exists():
        return str(parent_venv)
        
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

def run_quick_test():
    """Run a quick test with minimal configuration."""
    print("🚀 Running Quick Test Mode...")
    print("This will run a small-scale test to verify the pipeline works.")
    
    config = {
        'dataset': {
            'name': 'CIFAR10',
            'imbalance_ratio': 10,  # Smaller imbalance for testing
            'num_classes': 10,
            'batch_size': 32,        # Smaller batch for testing
            'num_workers': 2
        },
        'model': {
            'architecture': 'ResNet32',
            'feature_dim': 512,  # ResNet34 features
            'num_classes': 10
        },
        'memory_bank': {
            'capacity_per_class': 64,  # Smaller capacity for testing
            'alpha_base': 0.1,
            'tail_threshold_percentile': 30.0,
            'update_interval': 50
        },
        'training': {
            'initial_epochs': 2,        # Very few epochs for testing
            'synthetic_epochs': 1,      # Minimal synthetic training
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'scheduler_milestones': [5, 8],
            'scheduler_gamma': 0.1
        },
        'generation': {
            'num_prompts_per_tail_class': 5,     # Few prompts for testing
            'images_per_prompt': 2,               # Few images per prompt
            'generation_rounds': 1,               # Single round for testing
            'tail_improvement_threshold': 0.01,
            'option3_temperature': 0.8,
            'use_blip': True,
            'use_clip': True
        },
        'ddpm': {
            'enabled': False,  # Disable for quick test
            'num_timesteps': 100,
            'beta_schedule': 'cosine',
            'hidden_dim': 128,
            'num_layers': 2,
            'training_steps': 100,
            'features_per_class': 10
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
    }
    
    # Save config
    config_path = 'test_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run orchestrator
    subprocess.run([PYTHON_EXE, 'orchestrator.py', '--config', config_path, '--rounds', '1'])


def run_full_training():
    """Run full training with optimal settings."""
    print("🔥 Running Full Training Mode...")
    print("This will run the complete pipeline with optimal settings.")
    
    config = {
        'dataset': {
            'name': 'CIFAR10',
            'imbalance_ratio': 10,       # Moderate imbalance — enough tail samples for good prototypes
            'num_classes': 10,
            'batch_size': 128,
            'num_workers': 4
        },
        'model': {
            'architecture': 'ResNet32',
            'feature_dim': 512,  # ResNet34 features
            'num_classes': 10
        },
        'memory_bank': {
            'capacity_per_class': 256,
            'alpha_base': 0.1,
            'tail_threshold_percentile': 30.0,
            'update_interval': 100
        },
        'training': {
            'initial_epochs': 200,        # ResNet-34 needs ~200 epochs on CIFAR-10
            'synthetic_epochs': 25,       # Enough to integrate synthetic features
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'scheduler_milestones': [160, 180],  # Decay at 80% and 90% of total
            'scheduler_gamma': 0.1
        },
        'generation': {
            'num_prompts_per_tail_class': 50,
            'images_per_prompt': 4,
            'generation_rounds': 3,
            'tail_improvement_threshold': 2.0,
            'option3_temperature': 0.8,
            'use_blip': True,
            'use_clip': True
        },
        'ddpm': {
            'enabled': True,
            'num_timesteps': 1000,
            'beta_schedule': 'cosine',
            'hidden_dim': 1024,           # 2x feature_dim for better capacity
            'num_layers': 4,
            'training_steps': 10000,      # More training = better feature quality
            'features_per_class': 300     # More synthetic features per tail class
        },
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
    
    # Save config
    config_path = f'full_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run orchestrator
    subprocess.run([PYTHON_EXE, 'orchestrator.py', '--config', config_path, '--rounds', '3'])


def run_custom():
    """Run with custom settings."""
    parser = argparse.ArgumentParser(description='Custom run configuration')
    
    parser.add_argument('--imbalance', type=int, default=100,
                       help='Imbalance ratio (default: 100)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Initial training epochs (default: 20)')
    parser.add_argument('--synthetic-epochs', type=int, default=10,
                       help='Synthetic training epochs (default: 10)')
    parser.add_argument('--rounds', type=int, default=3,
                       help='Number of improvement rounds (default: 3)')
    parser.add_argument('--prompts', type=int, default=50,
                       help='Prompts per tail class (default: 50)')
    parser.add_argument('--images', type=int, default=4,
                       help='Images per prompt (default: 4)')
    parser.add_argument('--ddpm', action='store_true',
                       help='Enable DDPM feature generation')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    
    args = parser.parse_args(sys.argv[2:])  # Skip 'run.py custom'
    
    print(f"🎯 Running with custom settings:")
    print(f"  Imbalance ratio: {args.imbalance}")
    print(f"  Initial epochs: {args.epochs}")
    print(f"  Synthetic epochs: {args.synthetic_epochs}")
    print(f"  Improvement rounds: {args.rounds}")
    print(f"  Prompts per class: {args.prompts}")
    print(f"  Images per prompt: {args.images}")
    print(f"  DDPM enabled: {args.ddpm}")
    print(f"  GPU: {args.gpu}")
    
    
    cmd = [
        PYTHON_EXE, 'orchestrator.py',
        '--imbalance-ratio', str(args.imbalance),
        '--initial-epochs', str(args.epochs),
        '--synthetic-epochs', str(args.synthetic_epochs),
        '--rounds', str(args.rounds),
        '--gpu', str(args.gpu)
    ]
    
    if args.ddpm:
        cmd.append('--use-ddpm')
    
    subprocess.run(cmd)


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
                
                if 'improvements' in report:
                    print("\nTail Class Improvements:")
                    for cls_name, data in report['improvements'].items():
                        print(f"  {cls_name}: {data['initial']:.2f}% → {data['final']:.2f}% "
                              f"(+{data['improvement']:.2f}%)")


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
        print("  python run.py test      - Run quick test (2 epochs, minimal data)")
        print("  python run.py full      - Run full training (complete pipeline)")
        print("  python run.py custom    - Run with custom settings")
        print("  python run.py benchmark - Run benchmarks on multiple datasets")
        print("  python run.py status    - Show current training status")
        print("  python run.py clean     - Clean all output directories")
        print("\nExample:")
        print("  python run.py test")
        print("  python run.py benchmark --datasets cifar10 cifar100")
        print("  python run.py custom --epochs 30 --ddpm --gpu 1")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'test':
        run_quick_test()
    elif command == 'full':
        run_full_training()
    elif command == 'custom':
        run_custom()
    elif command == 'benchmark':
        # Delegate to benchmark_runner.py with remaining args
        benchmark_args = sys.argv[2:]  # pass everything after 'benchmark'
        cmd = [PYTHON_EXE, 'benchmark_runner.py'] + benchmark_args
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
    elif command == 'status':
        show_status()
    elif command == 'clean':
        clean_outputs()
    else:
        print(f"❌ Unknown command: {command}")
        print("Valid commands: test, full, custom, benchmark, status, clean")
        sys.exit(1)


if __name__ == "__main__":
    main()