"""
Benchmark Runner
=================
Runs the CSL + Memory Bank pipeline on multiple benchmark datasets
and collects statistics matching the CSL paper format.

Usage:
    py benchmark_runner.py --datasets cifar10 cifar100 tiny_imagenet
    py benchmark_runner.py --datasets cifar10 --imbalance-ratio 50
    py benchmark_runner.py --all
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

from models import ResNet18, ResNet32, ResNet50
from utils import CSLLossFunc
from dataset_configs import DATASET_CONFIGS, get_dataset_config, get_available_datasets
from dataloaders import get_cifar100_loaders, get_tiny_imagenet_loaders


# ─── Model Factory ───────────────────────────────────────────────────────────

def create_model(backbone, num_classes, image_size, device):
    """Create a model matching the CSL paper's backbone choice."""
    model_map = {
        'ResNet18': ResNet18,
        'ResNet32': ResNet32,
        'ResNet50': ResNet50,
    }
    if backbone not in model_map:
        raise ValueError(f"Unknown backbone: {backbone}. Available: {list(model_map.keys())}")

    model = model_map[backbone](
        num_classes=num_classes,
        pretrained=False,
        image_size=image_size
    )
    return model.to(device)


# ─── CIFAR-10-LT Loader (inline like orchestrator) ───────────────────────────

def get_cifar10_loaders(batch_size=128, data_dir='./datasets/CIFAR10',
                         imbalance_ratio=100, num_workers=4):
    """CIFAR-10-LT loader with exponential decay imbalance."""
    num_classes = 10

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    val_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    # Create imbalanced version
    targets = train_set.targets if hasattr(train_set, 'targets') else [train_set[i][1] for i in range(len(train_set))]
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    img_max = max(len(v) for v in class_indices.values())
    img_num_per_cls = []
    for cls_idx in range(num_classes):
        num = img_max * (imbalance_ratio ** (-cls_idx / (num_classes - 1)))
        img_num_per_cls.append(int(num))

    selected_indices = []
    for cls_idx in range(num_classes):
        indices = class_indices[cls_idx]
        np.random.seed(42)
        np.random.shuffle(indices)
        take = min(img_num_per_cls[cls_idx], len(indices))
        selected_indices.extend(indices[:take])
        img_num_per_cls[cls_idx] = take

    train_set = Subset(train_set, selected_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, img_num_per_cls


# ─── Training Logic ──────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, epoch, device):
    """Train one epoch, return avg loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"  Train Epoch {epoch+1}", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(labels, outputs, epoch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=total_loss/total, acc=100.*correct/total)

    return total_loss / total, 100. * correct / total


def validate(model, loader, num_classes, device):
    """Validate, return overall accuracy and per-class accuracies."""
    model.eval()
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    # Overall accuracy
    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    overall_acc = 100. * total_correct / total_samples if total_samples > 0 else 0

    # Per-class accuracy
    class_accs = {}
    for cls in range(num_classes):
        if class_total[cls] > 0:
            class_accs[cls] = 100. * class_correct[cls] / class_total[cls]
        else:
            class_accs[cls] = 0.0

    return overall_acc, class_accs


def compute_group_accuracies(class_accs, samples_per_class, num_classes):
    """Compute head, medium, tail group accuracies (Many/Medium/Few shot)."""
    if samples_per_class is None:
        return {'overall': np.mean(list(class_accs.values()))}

    samples = np.array(samples_per_class[:num_classes])
    tail_thresh = np.percentile(samples, 30)
    head_thresh = np.percentile(samples, 70)

    head_accs, med_accs, tail_accs = [], [], []
    for cls_idx in range(num_classes):
        acc = class_accs.get(cls_idx, 0.0)
        count = samples[cls_idx] if cls_idx < len(samples) else 0
        if count >= head_thresh:
            head_accs.append(acc)
        elif count <= tail_thresh:
            tail_accs.append(acc)
        else:
            med_accs.append(acc)

    return {
        'overall': np.mean(list(class_accs.values())),
        'head': np.mean(head_accs) if head_accs else 0.0,
        'medium': np.mean(med_accs) if med_accs else 0.0,
        'tail': np.mean(tail_accs) if tail_accs else 0.0,
    }


# ─── Single Dataset Benchmark ────────────────────────────────────────────────

def run_benchmark(dataset_name, imbalance_ratio=None, device='cuda',
                  num_workers=4, results_dir='./benchmark_results'):
    """Run a full benchmark on one dataset with CSL paper settings."""
    cfg = get_dataset_config(dataset_name)
    ir = imbalance_ratio or cfg['imbalance_ratios'][0]

    print(f"\n{'='*80}")
    print(f"  BENCHMARK: {cfg['name']}  |  Imbalance Ratio: {ir}")
    print(f"  Backbone: {cfg['backbone']}  |  Epochs: {cfg['epochs']}  |  LR: {cfg['lr']}")
    print(f"{'='*80}")

    # ── Load data ──
    if dataset_name == 'cifar10':
        train_loader, val_loader, samples_per_class = get_cifar10_loaders(
            batch_size=cfg['batch_size'], imbalance_ratio=ir, num_workers=num_workers
        )
    elif dataset_name == 'cifar100':
        train_loader, val_loader, samples_per_class = get_cifar100_loaders(
            batch_size=cfg['batch_size'], imbalance_factor=ir, num_workers=num_workers
        )
    elif dataset_name == 'tiny_imagenet':
        train_loader, val_loader, samples_per_class = get_tiny_imagenet_loaders(
            batch_size=cfg['batch_size'], imbalance_factor=ir, num_workers=num_workers
        )
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not auto-downloadable.")

    print(f"  Training samples: {sum(samples_per_class) if samples_per_class else '?'}")
    print(f"  Max/Min per class: {max(samples_per_class)}/{min(samples_per_class)}")

    # ── Identify tail classes ──
    samples_arr = np.array(samples_per_class)
    tail_thresh = np.percentile(samples_arr, 30)
    tail_classes = [i for i, c in enumerate(samples_arr) if c <= tail_thresh]

    # ── Create model ──
    model = create_model(cfg['backbone'], cfg['num_classes'], cfg['image_size'], device)

    # ── Loss with class-balanced weights ──
    criterion = CSLLossFunc(
        target_class_index=tail_classes,
        num_classes=cfg['num_classes'],
        samples_per_class=samples_per_class
    ).to(device)

    # ── Optimizer (SGD + momentum, matching paper) ──
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay']
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg['lr_decay_epochs'],
        gamma=cfg['lr_decay_factor']
    )

    # ── Training loop ──
    best_acc = 0.0
    best_class_accs = {}

    for epoch in range(cfg['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, device)
        overall_acc, class_accs = validate(model, val_loader, cfg['num_classes'], device)
        scheduler.step()

        if overall_acc > best_acc:
            best_acc = overall_acc
            best_class_accs = class_accs.copy()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            group_accs = compute_group_accuracies(class_accs, samples_per_class, cfg['num_classes'])
            print(f"  Epoch {epoch+1:3d}/{cfg['epochs']}  |  "
                  f"Loss: {train_loss:.4f}  |  "
                  f"Val Acc: {overall_acc:.2f}%  |  "
                  f"Head: {group_accs.get('head', 0):.1f}%  "
                  f"Med: {group_accs.get('medium', 0):.1f}%  "
                  f"Tail: {group_accs.get('tail', 0):.1f}%")

    # ── Final results ──
    final_groups = compute_group_accuracies(best_class_accs, samples_per_class, cfg['num_classes'])

    result = {
        'dataset': cfg['name'],
        'imbalance_ratio': ir,
        'backbone': cfg['backbone'],
        'epochs': cfg['epochs'],
        'best_accuracy': round(best_acc, 2),
        'head_accuracy': round(final_groups.get('head', 0), 2),
        'medium_accuracy': round(final_groups.get('medium', 0), 2),
        'tail_accuracy': round(final_groups.get('tail', 0), 2),
        'timestamp': datetime.now().isoformat(),
    }

    print(f"\n  ✓ BEST: {best_acc:.2f}%  |  "
          f"Head: {final_groups.get('head', 0):.2f}%  "
          f"Med: {final_groups.get('medium', 0):.2f}%  "
          f"Tail: {final_groups.get('tail', 0):.2f}%")

    # Save result
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"{dataset_name}_ir{ir}_result.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Result saved to {result_file}")

    return result


# ─── Multi-Dataset Runner ────────────────────────────────────────────────────

def run_all_benchmarks(dataset_names, imbalance_ratio=None, device='cuda',
                       num_workers=4, results_dir='./benchmark_results'):
    """Run benchmarks on multiple datasets and print summary table."""
    all_results = []

    for ds in dataset_names:
        cfg = get_dataset_config(ds)
        if not cfg.get('auto_download', False):
            print(f"\n⚠ Skipping {cfg['name']} — requires manual image download")
            continue

        ratios = [imbalance_ratio] if imbalance_ratio else cfg['imbalance_ratios']
        for ir in ratios:
            if ir is None:
                continue
            try:
                result = run_benchmark(ds, ir, device, num_workers, results_dir)
                all_results.append(result)
            except Exception as e:
                print(f"\n✗ Error on {cfg['name']} (IR={ir}): {e}")
                import traceback
                traceback.print_exc()

    # Print summary table
    if all_results:
        print_summary_table(all_results)
        # Save combined results
        combined_path = os.path.join(results_dir, 'benchmark_summary.json')
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results saved to {combined_path}")

    return all_results


def print_summary_table(results):
    """Print results in a table matching CSL paper format."""
    print(f"\n{'='*80}")
    print("  BENCHMARK SUMMARY — CSL + Memory Bank + Synthetic Data")
    print(f"{'='*80}")
    print(f"  {'Dataset':<20} {'IR':<6} {'Overall':>8} {'Head':>8} {'Med':>8} {'Tail':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for r in results:
        print(f"  {r['dataset']:<20} {r['imbalance_ratio']:<6} "
              f"{r['best_accuracy']:>7.2f}% "
              f"{r['head_accuracy']:>7.2f}% "
              f"{r['medium_accuracy']:>7.2f}% "
              f"{r['tail_accuracy']:>7.2f}%")

    print(f"{'='*80}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Benchmark Runner for CSL Pipeline')
    parser.add_argument('--datasets', nargs='+', default=None,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Datasets to benchmark')
    parser.add_argument('--all', action='store_true',
                        help='Run all auto-downloadable datasets')
    parser.add_argument('--imbalance-ratio', type=int, default=None,
                        help='Override imbalance ratio (default: use paper settings)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--results-dir', type=str, default='./benchmark_results',
                        help='Directory to save results')

    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if args.all:
        dataset_names = get_available_datasets()
    elif args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = get_available_datasets()
        print(f"No datasets specified, running all available: {dataset_names}")

    run_all_benchmarks(
        dataset_names,
        imbalance_ratio=args.imbalance_ratio,
        device=device,
        num_workers=args.num_workers,
        results_dir=args.results_dir
    )


if __name__ == '__main__':
    main()
