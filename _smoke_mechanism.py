"""
Focused check of the mechanism the pipeline relies on.

Claim under test: given a classifier fitted on an imbalanced feature distribution, retraining
it on real features augmented with DDPM-generated tail features raises tail accuracy.

This works at the feature level so it isolates the claim from backbone training, and it
includes a real-only control so any gain cannot be explained by the extra epochs alone.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from phase3_feature_ddpm import FeatureDDPM
from utils.csl_loss import CSLLossFunc

FEATURE_DIM = 32
NUM_CLASSES = 8
TRAIN_COUNTS = [400, 250, 150, 90, 55, 33, 20, 12]
TEST_PER_CLASS = 200
TAIL_CLASSES = [5, 6, 7]
HEAD_CLASSES = [0, 1, 2]
# Enough noise that the classes genuinely overlap. With well-separated classes the baseline
# already sits near 100% on the tail and there is no headroom for the test to detect.
NOISE = 3.5


def build_features(generator):
    """Post-ReLU-like class-conditional features with overlapping classes."""
    means = torch.rand(NUM_CLASSES, FEATURE_DIM, generator=generator) * 3.0

    def sample(class_id, count):
        noise = torch.randn(count, FEATURE_DIM, generator=generator) * NOISE
        return torch.relu(means[class_id].unsqueeze(0) + noise)

    train_x, train_y = [], []
    for class_id, count in enumerate(TRAIN_COUNTS):
        train_x.append(sample(class_id, count))
        train_y.append(torch.full((count,), class_id, dtype=torch.long))

    test_x, test_y = [], []
    for class_id in range(NUM_CLASSES):
        test_x.append(sample(class_id, TEST_PER_CLASS))
        test_y.append(torch.full((TEST_PER_CLASS,), class_id, dtype=torch.long))

    return (torch.cat(train_x), torch.cat(train_y),
            torch.cat(test_x), torch.cat(test_y))


def group_accuracy(classifier, test_x, test_y, class_ids):
    classifier.eval()
    with torch.no_grad():
        predictions = classifier(test_x).argmax(dim=1)

    accuracies = []
    for class_id in class_ids:
        mask = test_y == class_id
        accuracies.append((predictions[mask] == class_id).float().mean().item() * 100)
    return sum(accuracies) / len(accuracies)


def train_classifier(classifier, features, labels, counts, epochs, lr, batch_size=64):
    criterion = CSLLossFunc(target_class_index=TAIL_CLASSES, num_classes=NUM_CLASSES,
                            samples_per_class=counts)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(epochs):
        classifier.train()
        order = torch.randperm(len(features))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(labels[idx], classifier(features[idx]), epoch)
            loss.backward()
            optimizer.step()

    return classifier


def main():
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    train_x, train_y, test_x, test_y = build_features(generator)

    # ---- Stage 1: fit the classifier on the imbalanced distribution ----
    baseline = nn.Linear(FEATURE_DIM, NUM_CLASSES)
    train_classifier(baseline, train_x, train_y, TRAIN_COUNTS, epochs=30, lr=0.02)

    baseline_tail = group_accuracy(baseline, test_x, test_y, TAIL_CLASSES)
    baseline_head = group_accuracy(baseline, test_x, test_y, HEAD_CLASSES)
    baseline_all = group_accuracy(baseline, test_x, test_y, list(range(NUM_CLASSES)))
    print(f"After imbalanced training: overall {baseline_all:.2f}%  "
          f"tail {baseline_tail:.2f}%  head {baseline_head:.2f}%")

    # ---- Stage 2: fit the feature DDPM on tail-class features ----
    tail_mask = torch.isin(train_y, torch.tensor(TAIL_CLASSES))
    tail_x, tail_y = train_x[tail_mask], train_y[tail_mask]

    ddpm = FeatureDDPM(feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES, hidden_dim=256,
                       num_layers=4, num_timesteps=400, beta_schedule='cosine')
    ddpm.set_feature_stats(tail_x, non_negative=True)

    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=1e-3, weight_decay=0.01)
    for step in range(1500):
        idx = torch.randint(0, len(tail_x), (64,))
        loss = ddpm(tail_x[idx], tail_y[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
        optimizer.step()
    print(f"DDPM final training loss: {loss.item():.4f}")

    # ---- Stage 3: generate and confidence-filter synthetic tail features ----
    ddpm.eval()
    prototypes = {c: train_x[train_y == c].mean(dim=0) for c in TAIL_CLASSES}

    synthetic_x, synthetic_y = [], []
    per_class_target = 300
    for class_id in TAIL_CLASSES:
        request = int(per_class_target * 1.5)
        generated, confidences = ddpm.sample_with_confidence(
            torch.full((request,), class_id, dtype=torch.long), prototypes,
            device='cpu', num_steps=100, batch_size=256)

        keep = torch.argsort(confidences, descending=True)[:per_class_target]
        keep = keep[confidences[keep] >= 0.5]
        synthetic_x.append(generated[keep])
        synthetic_y.append(torch.full((len(keep),), class_id, dtype=torch.long))

        real_class = train_x[train_y == class_id]
        print(f"  class {class_id}: kept {len(keep)}/{request}, "
              f"real mean {real_class.mean():.2f} vs generated mean {generated[keep].mean():.2f}, "
              f"mean confidence {confidences[keep].mean():.3f}")

    synthetic_x = torch.cat(synthetic_x)
    synthetic_y = torch.cat(synthetic_y)

    # ---- Stage 4: retrain the classifier, with a real-only control ----
    augmented_x = torch.cat([train_x, synthetic_x])
    augmented_y = torch.cat([train_y, synthetic_y])
    augmented_counts = [
        TRAIN_COUNTS[c] + int((synthetic_y == c).sum()) for c in range(NUM_CLASSES)
    ]

    finetune_epochs = 30
    finetune_lr = 0.002

    control = train_classifier(copy.deepcopy(baseline), train_x, train_y, TRAIN_COUNTS,
                               epochs=finetune_epochs, lr=finetune_lr)
    augmented = train_classifier(copy.deepcopy(baseline), augmented_x, augmented_y,
                                 augmented_counts, epochs=finetune_epochs, lr=finetune_lr)

    control_tail = group_accuracy(control, test_x, test_y, TAIL_CLASSES)
    control_head = group_accuracy(control, test_x, test_y, HEAD_CLASSES)
    control_all = group_accuracy(control, test_x, test_y, list(range(NUM_CLASSES)))

    aug_tail = group_accuracy(augmented, test_x, test_y, TAIL_CLASSES)
    aug_head = group_accuracy(augmented, test_x, test_y, HEAD_CLASSES)
    aug_all = group_accuracy(augmented, test_x, test_y, list(range(NUM_CLASSES)))

    print("\n" + "=" * 74)
    print(f"{'Variant':<28}{'Overall':>12}{'Tail':>12}{'Head':>12}")
    print("-" * 74)
    print(f"{'Imbalanced baseline':<28}{baseline_all:>11.2f}%{baseline_tail:>11.2f}%{baseline_head:>11.2f}%")
    print(f"{'+ real-only finetune':<28}{control_all:>11.2f}%{control_tail:>11.2f}%{control_head:>11.2f}%")
    print(f"{'+ synthetic finetune':<28}{aug_all:>11.2f}%{aug_tail:>11.2f}%{aug_head:>11.2f}%")
    print("=" * 74)
    print(f"Tail gain over baseline:     {aug_tail - baseline_tail:+.2f} pp")
    print(f"Tail gain over real-only:    {aug_tail - control_tail:+.2f} pp")
    print(f"Overall change vs baseline:  {aug_all - baseline_all:+.2f} pp")

    assert aug_tail > baseline_tail, \
        f"synthetic augmentation did not improve tail accuracy ({baseline_tail:.2f} -> {aug_tail:.2f})"
    assert aug_tail > control_tail, \
        f"gain is explained by extra epochs, not synthetic data ({control_tail:.2f} -> {aug_tail:.2f})"

    print("\nMechanism check passed.")


if __name__ == '__main__':
    main()
