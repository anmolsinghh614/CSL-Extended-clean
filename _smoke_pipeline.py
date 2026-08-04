"""
End-to-end pipeline check on a small synthetic dataset.

Exercises every stage the real run goes through (memory bank training, exemplar selection,
prompt generation, feature-space synthesis, classifier retraining, reporting) without waiting
on the CIFAR-10 download or a GPU, and asserts the invariants that are easy to break: the
backbone stays frozen while the classifier is retrained, the synthetic budget stays in
proportion to the real samples, early stopping and best-model restore take effect, and every
metric series stays the same length so the report describes the model that was returned.

Whether tail accuracy improves is *not* asserted here. A few hundred 16x16 noise images are
too little for a ResNet to learn stable features from, so the run-to-run spread in the
initial training swamps the effect of augmentation. `_smoke_mechanism.py` tests that claim
directly in feature space, where the generator actually operates, against a real-only control.
"""

import shutil
import types

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from orchestrator import MemoryConditionedOrchestrator
from pipeline_config import merge_config

NUM_CLASSES = 6
IMAGE_SIZE = 16
IMBALANCE_RATIO = 20
MAX_PER_CLASS = 300
TEST_PER_CLASS = 60


# Every image carries a marker block for *every* class with a random intensity; the label is the
# block that happens to be brightest, and the true class only gets a modest intensity bonus. A
# single always-present bright block would be found (or missed) wholesale by the conv net, pinning
# accuracy at 100% or at chance; making the classes compete gives a graded, tunable difficulty and
# leaves the tail genuinely harder than the head.
BLOCK = 4
MARKER_BONUS = 1.6
MARKER_SPREAD = 1.0
NOISE_SCALE = 0.5


def _block_slice(class_id):
    blocks_per_row = IMAGE_SIZE // BLOCK
    row = (class_id // blocks_per_row) * BLOCK
    col = (class_id % blocks_per_row) * BLOCK
    return slice(row, row + BLOCK), slice(col, col + BLOCK)


def make_class_images(class_id, count, generator):
    """Background noise plus one competing marker block per class, brightest on the true class."""
    images = torch.randn(count, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator) * NOISE_SCALE

    intensities = torch.randn(count, NUM_CLASSES, generator=generator) * MARKER_SPREAD
    intensities[:, class_id] += MARKER_BONUS

    for other in range(NUM_CLASSES):
        rows, cols = _block_slice(other)
        images[:, :, rows, cols] += intensities[:, other].view(count, 1, 1, 1)

    return images


def build_synthetic_data():
    generator = torch.Generator().manual_seed(0)

    train_images, train_labels = [], []
    counts = []
    for class_id in range(NUM_CLASSES):
        count = max(8, int(MAX_PER_CLASS * (IMBALANCE_RATIO ** (-class_id / (NUM_CLASSES - 1)))))
        counts.append(count)
        train_images.append(make_class_images(class_id, count, generator))
        train_labels.append(torch.full((count,), class_id, dtype=torch.long))

    test_images, test_labels = [], []
    for class_id in range(NUM_CLASSES):
        test_images.append(make_class_images(class_id, TEST_PER_CLASS, generator))
        test_labels.append(torch.full((TEST_PER_CLASS,), class_id, dtype=torch.long))

    train = TensorDataset(torch.cat(train_images), torch.cat(train_labels))
    test = TensorDataset(torch.cat(test_images), torch.cat(test_labels))
    return train, test, counts


def patched_step1(self):
    """Stand-in for step 1 that installs the synthetic dataset."""
    train, test, counts = build_synthetic_data()

    batch_size = self.config['dataset']['batch_size']
    self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    self.train_eval_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0)
    self.test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)

    self.samples_per_class = counts
    tail_percentile = self.config['memory_bank']['tail_threshold_percentile']
    tail_threshold = np.percentile(counts, tail_percentile)
    head_threshold = np.percentile(counts, 100 - tail_percentile)
    self.tail_classes = [i for i, c in enumerate(counts) if c <= tail_threshold]
    self.head_classes = [i for i, c in enumerate(counts) if c >= head_threshold]

    print(f"Synthetic dataset: {counts}")
    print(f"  Tail classes: {self.tail_classes}  Head classes: {self.head_classes}")
    return train, counts


def main():
    # Model init and shuffling both matter a lot at this scale; without a seed the initial
    # training lands anywhere between collapse and convergence and the assertions below
    # become coin flips.
    torch.manual_seed(0)
    np.random.seed(0)

    config = merge_config({
        'dataset': {'num_classes': NUM_CLASSES, 'batch_size': 32, 'num_workers': 0,
                    'imbalance_ratio': IMBALANCE_RATIO},
        'model': {'num_classes': NUM_CLASSES},
        'memory_bank': {'capacity_per_class': 64, 'tail_refresh_interval': 128,
                        'tail_threshold_percentile': 34.0},
        'training': {'initial_epochs': 12, 'synthetic_epochs': 8, 'lr': 0.01,
                     'scheduler_milestones': [9], 'synthetic_patience': 3},
        'generation': {'num_prompts_per_tail_class': 6, 'generation_rounds': 1,
                       'use_stable_diffusion': False, 'use_blip': False, 'use_clip': False,
                       'tail_improvement_threshold': 0.5, 'num_exemplars_per_tail_class': 3},
        'ddpm': {'enabled': True, 'num_timesteps': 200, 'hidden_dim': 256, 'num_layers': 3,
                 'training_steps': 900, 'max_epochs': 30, 'features_per_class': 250,
                 'sampling_steps': 50, 'min_confidence': 0.4},
        'paths': {k: f'./_smoke_out/{k}' for k in
                  ['checkpoint_dir', 'memory_dir', 'prompts_dir', 'images_dir',
                   'features_dir', 'logs_dir', 'results_dir']}
    })

    shutil.rmtree('./_smoke_out', ignore_errors=True)

    orchestrator = MemoryConditionedOrchestrator(config)
    orchestrator.step1_create_imbalanced_dataset = types.MethodType(patched_step1, orchestrator)

    # Capture backbone and classifier weights around the synthetic stage to confirm that the
    # freeze actually holds and that the classifier is the thing being retrained.
    weights = {}
    original_step6 = orchestrator.step6_train_with_synthetic

    def instrumented_step6(features, epochs):
        classifier = orchestrator._get_classifier()
        stem = orchestrator.model.model.conv1
        weights['fc_before'] = classifier.weight.detach().clone()
        weights['stem_before'] = stem.weight.detach().clone()
        result = original_step6(features, epochs)
        weights['fc_after'] = classifier.weight.detach().clone()
        weights['stem_after'] = stem.weight.detach().clone()
        return result

    orchestrator.step6_train_with_synthetic = instrumented_step6
    orchestrator.run_iterative_improvement(max_rounds=1)

    fc_delta = (weights['fc_after'] - weights['fc_before']).abs().max().item()
    stem_delta = (weights['stem_after'] - weights['stem_before']).abs().max().item()
    assert fc_delta > 1e-6, "classifier weights did not change during synthetic training"
    assert stem_delta == 0.0, f"frozen backbone changed by {stem_delta}"
    print(f"\nClassifier weight delta: {fc_delta:.5f}; frozen backbone delta: {stem_delta:.1e}")

    # ---- Assertions on what the run produced ----
    metrics = orchestrator.metrics
    baseline = orchestrator.baseline_snapshot

    assert baseline is not None, "baseline snapshot was never captured"
    # The synthetic stage can stop early, so the count is a range rather than a fixed number.
    initial_epochs = config['training']['initial_epochs']
    max_evals = initial_epochs + config['training']['synthetic_epochs']
    num_evals = len(metrics['epoch_accuracies'])
    assert initial_epochs < num_evals <= max_evals, \
        f"expected between {initial_epochs + 1} and {max_evals} evaluations, got {num_evals}"
    assert len(metrics['tail_group_accuracies']) == num_evals
    assert len(metrics['head_group_accuracies']) == num_evals
    assert len(metrics['epoch_losses']) == num_evals
    assert len(metrics['round_summaries']) == 1
    assert len(metrics['stage_boundaries']) == 2

    summary = metrics['round_summaries'][0]
    assert summary['synthetic_samples'] > 0, "no synthetic features were produced"

    for cls in range(NUM_CLASSES):
        assert len(metrics['class_accuracies'][cls]) == num_evals

    # Synthetic samples must stay in proportion to the real ones. Flooding a 15-sample class
    # with hundreds of generated features makes the classifier fit the generator, not the class.
    counts = orchestrator.samples_per_class
    generated = metrics['synthetic_samples_generated']
    ratio_cap = config['ddpm']['max_synthetic_ratio']
    for cls, count in generated.items():
        assert count <= ratio_cap * counts[cls] + 1, \
            f"class {cls}: {count} synthetic vs {counts[cls]} real exceeds the {ratio_cap}x cap"
        assert counts[cls] + count <= max(counts), \
            f"class {cls} was augmented past the largest real class"
    assert set(generated) <= set(orchestrator.tail_classes), \
        f"non-tail classes were augmented: {sorted(set(generated) - set(orchestrator.tail_classes))}"

    # The reported final numbers must match the model that is actually left in place, which is
    # the restored best-tail checkpoint rather than the last epoch trained.
    _, live_acc, live_class_acc = orchestrator._evaluate()
    assert abs(live_acc - metrics['epoch_accuracies'][-1]) < 1e-6, \
        (f"report ends at {metrics['epoch_accuracies'][-1]:.2f}% but the model in memory "
         f"scores {live_acc:.2f}%")
    live_tail = orchestrator._group_accuracy(live_class_acc, orchestrator.tail_classes)
    assert abs(live_tail - metrics['tail_group_accuracies'][-1]) < 1e-6

    tail_before = baseline['tail']
    tail_after = metrics['tail_group_accuracies'][-1]
    head_before = baseline['head']
    head_after = metrics['head_group_accuracies'][-1]

    print("\n" + "=" * 70)
    print("SMOKE TEST ASSERTIONS")
    print("=" * 70)
    print(f"Synthetic samples used: {summary['synthetic_samples']}")
    print(f"Tail group: {tail_before:.2f}% -> {tail_after:.2f}% ({tail_after - tail_before:+.2f} pp)")
    print(f"Head group: {head_before:.2f}% -> {head_after:.2f}% ({head_after - head_before:+.2f} pp)")
    print(f"Overall:    {baseline['overall']:.2f}% -> "
          f"{metrics['epoch_accuracies'][-1]:.2f}% "
          f"({metrics['epoch_accuracies'][-1] - baseline['overall']:+.2f} pp)")

    # Report and plot must exist
    import os
    results = os.listdir('./_smoke_out/results_dir')
    assert any(f.startswith('final_report_') for f in results), results
    assert any(f.startswith('results_visualization_') for f in results), results
    assert any(f.startswith('tail_analysis_') for f in results), results

    memory_files = os.listdir('./_smoke_out/memory_dir')
    assert any(f.endswith('.pt') for f in memory_files), memory_files
    assert any(f.endswith('_summary.json') for f in memory_files), memory_files

    print("\nArtifacts written:", sorted(results))
    print("Memory checkpoints:", sorted(memory_files))
    print("\nPipeline smoke test completed.")

    if tail_after <= tail_before:
        print(f"\nNOTE: tail accuracy did not improve in this synthetic setting "
              f"({tail_before:.2f}% -> {tail_after:.2f}%).")


if __name__ == '__main__':
    main()
