#!/usr/bin/env python3
"""
Multi-seed benchmark table runner.

Reproduces the reporting format of LDAL's Table V — mean top-1 accuracy, standard deviation
and a 95% confidence interval over several independent runs with different random seeds — for
this pipeline, so the two can be placed side by side.

    python run_table.py --dataset cifar10  --imbalance 100
    python run_table.py --dataset cifar100 --imbalance 100 --prompts 10 --images 2
    python run_table.py --dataset cifar10  --seeds 3          # fewer runs while iterating
    python run_table.py --check                               # what is runnable right now

Each seed is a separate `orchestrator.py` process, so one crashed run does not take the rest
of the sweep with it, and its results are read back from the report it writes to disk.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Only torch-free modules at import time. `get_num_classes` is imported lazily inside
# build_config, because reaching it pulls in dataloaders/__init__ and from there the whole
# torch stack — several seconds of load for subcommands like --check that just read dicts and
# look for files on disk.
from dataset_configs import DATASET_CONFIGS
from pipeline_config import merge_config

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).parent.resolve()

# LDAL's Table V, for the side-by-side. Imbalance factor 100 for both CIFAR variants.
LDAL_TABLE_V = {
    'cifar10': {'mean': 79.77, 'std': 0.73, 'ci': (79.13, 80.41)},
    'cifar100': {'mean': 48.88, 'std': 0.86, 'ci': (48.12, 49.63)},
    'imagenet_lt': {'mean': 49.67, 'std': 0.66, 'ci': (49.09, 50.24)},
    'inaturalist': {'mean': 66.53, 'std': 0.83, 'ci': (65.80, 67.25)},
}

# The paper reports the *best* single run in its comparison tables and the multi-seed spread
# separately, so both are worth having in view.
LDAL_BEST = {
    'cifar10': {100: 80.19, 50: 84.41},
    'cifar100': {200: 45.04, 100: 49.79, 50: 52.72},
}

CIFAR_DATASETS = ('cifar10', 'cifar100')


def python_executable():
    """Prefer a project virtualenv over whatever interpreter launched this script."""
    candidates = [
        Path('.venv') / 'Scripts' / 'python.exe',
        Path('.venv') / 'bin' / 'python',
        Path('venv') / 'Scripts' / 'python.exe',
        Path('venv') / 'bin' / 'python',
    ]
    for root in (REPO_ROOT, REPO_ROOT.parent):
        for relative in candidates:
            if (root / relative).exists():
                return str(root / relative)
    return sys.executable


PYTHON_EXE = python_executable()


# ─── Dataset availability ────────────────────────────────────────────────────

def describe_availability(dataset):
    """
    Report whether `dataset` can be run, and if not, exactly what is missing.

    CIFAR-10 and CIFAR-100 download themselves through torchvision. ImageNet-LT and
    iNaturalist-2018 cannot: they need image archives that are hundreds of gigabytes and, for
    ImageNet, an account on image-net.org. Both also need the split files that define which
    images belong to the long-tailed subset, and only the validation splits are in this repo.
    """
    cfg = DATASET_CONFIGS[dataset]

    if dataset in CIFAR_DATASETS:
        return True, [f"torchvision downloads {cfg['name']} automatically on first use "
                      f"(~170 MB) into the configured data_dir"]

    problems = []
    for key in ('train_txt', 'val_txt', 'test_txt'):
        path = cfg.get(key)
        if path and not (REPO_ROOT / path).exists():
            problems.append(f"missing split file: {path}")

    image_root = os.environ.get(f"{dataset.upper()}_ROOT")
    if not image_root:
        problems.append(f"image directory not configured (set the {dataset.upper()}_ROOT "
                        f"environment variable to the extracted image tree)")
    elif not Path(image_root).is_dir():
        problems.append(f"{dataset.upper()}_ROOT points at {image_root}, which does not exist")

    return (not problems), problems


DOWNLOAD_NOTES = {
    'imagenet_lt': """\
ImageNet-LT is a subset of ILSVRC-2012, which cannot be downloaded without an account.
  1. Register at https://image-net.org and download ILSVRC2012_img_train.tar (~138 GB)
     and ILSVRC2012_img_val.tar (~6.3 GB).
  2. Extract them, then point IMAGENET_LT_ROOT at the directory holding the class folders.
  3. Fetch the split files from the OLTR release (Liu et al., 2019) and place them at
     dataloaders/ImageNet_LT/ImageNet_LT_{train,val,test}.txt
     https://github.com/zhmiao/OpenLongTailRecognition-OLTR
Training ResNet-50 for 120 epochs on this is a multi-day job on a single GPU.""",

    'inaturalist': """\
iNaturalist-2018 images are public but very large.
  1. Download train_val2018.tar.gz (~120 GB) from
     https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz
  2. Extract it, then point INATURALIST_ROOT at the resulting directory.
  3. Fetch the split files from the OLTR release and place them at
     dataloaders/Inaturalist18/iNaturalist18_{train,val,test}.txt
Training ResNet-50 for 160 epochs on 8142 classes is a multi-day job on a single GPU.""",
}


def print_availability():
    """Print a readiness report for every dataset in the table."""
    print("\nDataset readiness")
    print("=" * 78)
    for dataset in ('cifar10', 'cifar100', 'imagenet_lt', 'inaturalist'):
        cfg = DATASET_CONFIGS[dataset]
        ready, notes = describe_availability(dataset)
        status = "READY" if ready else "NOT AVAILABLE"
        print(f"\n{cfg['name']:<20} {status}")
        print(f"  {cfg['num_classes']} classes, {cfg['image_size']}x{cfg['image_size']}, "
              f"{cfg['backbone']}, {cfg['epochs']} epochs")
        for note in notes:
            print(f"  - {note}")
        if not ready and dataset in DOWNLOAD_NOTES:
            for line in DOWNLOAD_NOTES[dataset].splitlines():
                print(f"  {line}")
    print("\n" + "=" * 78)


# ─── Statistics ──────────────────────────────────────────────────────────────

def summarize(values):
    """
    Mean, sample standard deviation and 95% confidence interval of a list of accuracies.

    The interval uses 1.96 * s / sqrt(n), which is what reproduces the intervals printed in
    LDAL's Table V from its own means and standard deviations. For five runs a t-interval
    (2.776 for 4 degrees of freedom) is the more defensible choice, so it is reported
    alongside rather than instead.
    """
    count = len(values)
    if count == 0:
        return None

    mean = sum(values) / count
    if count == 1:
        return {'n': 1, 'mean': mean, 'std': 0.0, 'sem': 0.0,
                'ci95_normal': (mean, mean), 'ci95_t': (mean, mean), 'values': values}

    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    std = math.sqrt(variance)
    sem = std / math.sqrt(count)

    # Two-sided 97.5th percentile of Student's t, for the small sample sizes used here.
    t_critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                  7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(count, 1.96)

    return {
        'n': count,
        'mean': mean,
        'std': std,
        'sem': sem,
        'ci95_normal': (mean - 1.96 * sem, mean + 1.96 * sem),
        'ci95_t': (mean - t_critical * sem, mean + t_critical * sem),
        'values': values,
    }


# ─── Running one seed ────────────────────────────────────────────────────────

def build_config(dataset, imbalance, seed, args, results_dir):
    """Config for one seed, with the benchmark protocol left untouched unless overridden."""
    from dataloaders.cifar_lt_loader import get_num_classes
    num_classes = get_num_classes(dataset)

    generation = {}
    if args.prompts is not None:
        generation['num_prompts_per_tail_class'] = args.prompts
    if args.images is not None:
        generation['images_per_prompt'] = args.images
    if args.rounds is not None:
        generation['generation_rounds'] = args.rounds
    if args.no_stable_diffusion:
        generation['use_stable_diffusion'] = False
    if args.sd_low_vram is not None:
        generation['sd_low_vram'] = args.sd_low_vram

    return merge_config({
        'seed': seed,
        'dataset': {'name': dataset, 'num_classes': num_classes,
                    'imbalance_ratio': imbalance, 'image_size': None},
        'model': {'num_classes': num_classes},
        'generation': generation,
        # Each seed writes into its own results directory, so reports cannot be confused for
        # one another when they are read back.
        'paths': {'results_dir': results_dir},
    })


def latest_report(results_dir):
    """Most recent final_report_*.json in a results directory, or None."""
    reports = sorted(Path(results_dir).glob('final_report_*.json'),
                     key=lambda path: path.stat().st_mtime)
    return reports[-1] if reports else None


def run_seed(dataset, imbalance, seed, args, sweep_dir):
    """Run one seed to completion and return its parsed summary, or None if it failed."""
    results_dir = sweep_dir / f"seed{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(dataset, imbalance, seed, args, str(results_dir))

    config_path = sweep_dir / f"config_seed{seed}.json"
    with open(config_path, 'w') as handle:
        json.dump(config, handle, indent=2)

    log_path = sweep_dir / f"seed{seed}.log"
    rounds = str(config['generation']['generation_rounds'])

    print(f"\n{'─' * 78}")
    print(f"  {DATASET_CONFIGS[dataset]['name']}  imbalance {imbalance}  seed {seed}")
    print(f"  log: {log_path}")
    print(f"{'─' * 78}", flush=True)

    started = time.time()
    with open(log_path, 'w', encoding='utf-8') as log:
        completed = subprocess.run(
            [PYTHON_EXE, 'orchestrator.py',
             '--config', str(config_path),
             '--rounds', rounds,
             '--seed', str(seed),
             '--gpu', str(args.gpu)],
            cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT
        )
    elapsed = time.time() - started

    if completed.returncode != 0:
        print(f"  FAILED (exit {completed.returncode}) after {timedelta(seconds=int(elapsed))}")
        print(f"  see {log_path}")
        return None

    report_path = latest_report(results_dir)
    if report_path is None:
        print(f"  completed but wrote no report; see {log_path}")
        return None

    with open(report_path) as handle:
        report = json.load(handle)

    summary = report['summary']
    result = {
        'seed': seed,
        'elapsed_seconds': round(elapsed, 1),
        'report': str(report_path),
        'rounds_run': report['metrics']['total_rounds'],
        'before': summary['before_synthetic'],
        'after': summary['after_synthetic'],
    }

    print(f"  done in {timedelta(seconds=int(elapsed))} — "
          f"overall {result['before']['overall']:.2f} -> {result['after']['overall']:.2f}, "
          f"tail {result['before']['tail_group']:.2f} -> {result['after']['tail_group']:.2f}")
    return result


# ─── Reporting ───────────────────────────────────────────────────────────────

METRICS = [
    ('overall', 'Overall'),
    ('tail_group', 'Tail'),
    ('head_group', 'Head'),
]


def aggregate(results):
    """Per-metric statistics for the before- and after-synthesis states."""
    stats = {}
    for key, _ in METRICS:
        for stage in ('before', 'after'):
            stats[f"{stage}_{key}"] = summarize([r[stage][key] for r in results])
    return stats


def print_table(dataset, imbalance, results, stats):
    cfg = DATASET_CONFIGS[dataset]
    print("\n" + "=" * 78)
    print(f"  {cfg['name']}  |  imbalance {imbalance}  |  {len(results)} seeds"
          f"  |  {cfg['backbone']}, {cfg['epochs']} epochs")
    print("=" * 78)

    print(f"\n  {'Metric':<22}{'Mean':>9}{'Std':>8}{'95% CI':>20}")
    print(f"  {'-' * 22}{'-' * 9}{'-' * 8}{'-' * 20}")
    for key, label in METRICS:
        for stage, stage_label in (('before', 'before synth'), ('after', 'after synth')):
            entry = stats[f"{stage}_{key}"]
            if entry is None:
                continue
            low, high = entry['ci95_normal']
            print(f"  {label + ' (' + stage_label + ')':<22}"
                  f"{entry['mean']:>8.2f}%{entry['std']:>8.2f}"
                  f"{f'[{low:.2f}, {high:.2f}]':>20}")

    print(f"\n  Per-seed overall accuracy after synthesis:")
    for result in results:
        print(f"    seed {result['seed']}: {result['after']['overall']:.2f}%  "
              f"({result['rounds_run']} rounds, "
              f"{timedelta(seconds=int(result['elapsed_seconds']))})")

    reference = LDAL_TABLE_V.get(dataset)
    after = stats['after_overall']
    if reference and after:
        low, high = reference['ci']
        gap = after['mean'] - reference['mean']
        print(f"\n  {'Comparison (overall top-1)':<30}{'Mean':>9}{'Std':>8}{'95% CI':>20}")
        print(f"  {'-' * 30}{'-' * 9}{'-' * 8}{'-' * 20}")
        print(f"  {'LDAL (paper, Table V)':<30}{reference['mean']:>8.2f}%"
              f"{reference['std']:>8.2f}{f'[{low:.2f}, {high:.2f}]':>20}")
        alow, ahigh = after['ci95_normal']
        print(f"  {'Ours (after synthesis)':<30}{after['mean']:>8.2f}%"
              f"{after['std']:>8.2f}{f'[{alow:.2f}, {ahigh:.2f}]':>20}")
        verdict = "ahead of" if gap > 0 else "behind"
        print(f"\n  Gap: {gap:+.2f} pp ({verdict} LDAL)")
        if reference['ci'][0] <= ahigh and alow <= reference['ci'][1]:
            print("  The two confidence intervals overlap, so this difference is not "
                  "separable at 95%.")

    best = LDAL_BEST.get(dataset, {}).get(imbalance)
    if best is not None and after:
        best_of_ours = max(r['after']['overall'] for r in results)
        print(f"\n  Best single run: ours {best_of_ours:.2f}% vs LDAL {best:.2f}% "
              f"({best_of_ours - best:+.2f} pp)")

    print("\n" + "=" * 78)


def write_outputs(dataset, imbalance, results, stats, sweep_dir):
    payload = {
        'dataset': dataset,
        'dataset_name': DATASET_CONFIGS[dataset]['name'],
        'imbalance_ratio': imbalance,
        'seeds': [r['seed'] for r in results],
        'protocol': {
            key: DATASET_CONFIGS[dataset][key]
            for key in ('backbone', 'epochs', 'lr', 'lr_decay_epochs', 'lr_decay_factor',
                        'weight_decay', 'momentum', 'batch_size', 'image_size')
        },
        'runs': results,
        'statistics': stats,
        'ldal_reference': LDAL_TABLE_V.get(dataset),
        'generated': datetime.now().isoformat(),
    }

    json_path = sweep_dir / 'table.json'
    with open(json_path, 'w') as handle:
        json.dump(payload, handle, indent=2)

    after = stats['after_overall']
    reference = LDAL_TABLE_V.get(dataset)
    lines = [
        f"### {DATASET_CONFIGS[dataset]['name']} (imbalance {imbalance}, "
        f"{len(results)} seeds)",
        "",
        "| Method | Mean Acc. (%) | Std. Deviation (±) | 95% Confidence |",
        "|--------|---------------|--------------------|----------------|",
    ]
    if reference:
        lines.append(f"| LDAL (paper) | {reference['mean']:.2f} | {reference['std']:.2f} | "
                     f"[{reference['ci'][0]:.2f}, {reference['ci'][1]:.2f}] |")
    for stage, label in (('before', 'Ours, before synthesis'),
                         ('after', 'Ours, after synthesis')):
        entry = stats[f'{stage}_overall']
        if entry:
            low, high = entry['ci95_normal']
            lines.append(f"| {label} | {entry['mean']:.2f} | {entry['std']:.2f} | "
                         f"[{low:.2f}, {high:.2f}] |")

    markdown_path = sweep_dir / 'table.md'
    with open(markdown_path, 'w', encoding='utf-8') as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"\nWrote {json_path}")
    print(f"Wrote {markdown_path}")
    if after:
        print(f"Per-seed logs and reports are under {sweep_dir}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run a benchmark over several seeds and report mean, std and 95% CI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument('--dataset', choices=list(DATASET_CONFIGS.keys()), default='cifar10',
                        help='Benchmark to run (default: cifar10)')
    parser.add_argument('--imbalance', type=int, default=100,
                        help='Imbalance ratio (default: 100, the standard setting)')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds, run as 0..N-1 (default: 5, matching the paper)')
    parser.add_argument('--seed-list', type=int, nargs='+', default=None,
                        help='Explicit seeds, overriding --seeds')
    parser.add_argument('--prompts', type=int, default=None,
                        help='Prompts per tail class (default: benchmark value, 50)')
    parser.add_argument('--images', type=int, default=None,
                        help='Images per prompt (default: benchmark value, 4)')
    parser.add_argument('--rounds', type=int, default=None,
                        help='Improvement rounds (default: benchmark value, 3)')
    parser.add_argument('--no-stable-diffusion', action='store_true',
                        help='Skip Stable Diffusion; feature-space DDPM only (much faster)')
    parser.add_argument('--sd-low-vram', dest='sd_low_vram', action='store_true', default=None,
                        help='Force weight streaming for Stable Diffusion (slow, low VRAM)')
    parser.add_argument('--no-sd-low-vram', dest='sd_low_vram', action='store_false',
                        help='Force the diffusion pipeline to stay resident on the GPU (fast)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--out', type=str, default='./table_results',
                        help='Where to write logs, reports and the table')
    parser.add_argument('--check', action='store_true',
                        help='Report which datasets are runnable and exit')

    args = parser.parse_args()

    if args.check:
        print_availability()
        return 0

    dataset = args.dataset
    ready, problems = describe_availability(dataset)
    if not ready:
        print(f"\nCannot run {DATASET_CONFIGS[dataset]['name']}:")
        for problem in problems:
            print(f"  - {problem}")
        note = DOWNLOAD_NOTES.get(dataset)
        if note:
            print("\n" + note)
        print("\nRun `python run_table.py --check` for the full readiness report.")
        return 1

    if dataset not in CIFAR_DATASETS:
        print(f"\n{DATASET_CONFIGS[dataset]['name']} has its data in place, but the pipeline "
              f"itself only builds CIFAR loaders today (orchestrator.step1). Extending it to "
              f"this dataset is a separate piece of work.")
        return 1

    seeds = args.seed_list if args.seed_list else list(range(args.seeds))

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_dir = Path(args.out) / f"{dataset}_ir{args.imbalance}_{stamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    cfg = DATASET_CONFIGS[dataset]
    print("=" * 78)
    print(f"  {cfg['name']}  |  imbalance {args.imbalance}  |  seeds {seeds}")
    print(f"  {cfg['backbone']}, {cfg['image_size']}x{cfg['image_size']}, "
          f"{cfg['epochs']} epochs, lr {cfg['lr']} decayed by {cfg['lr_decay_factor']} at "
          f"{cfg['lr_decay_epochs']}, weight decay {cfg['weight_decay']}")
    print(f"  Output: {sweep_dir}")
    print("=" * 78)

    results = []
    for seed in seeds:
        result = run_seed(dataset, args.imbalance, seed, args, sweep_dir)
        if result is not None:
            results.append(result)

    if not results:
        print("\nEvery seed failed; nothing to aggregate. Check the logs listed above.")
        return 1

    if len(results) < len(seeds):
        print(f"\nWarning: {len(seeds) - len(results)} of {len(seeds)} seeds failed. The "
              f"statistics below cover only the {len(results)} that completed.")

    stats = aggregate(results)
    print_table(dataset, args.imbalance, results, stats)
    write_outputs(dataset, args.imbalance, results, stats, sweep_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
