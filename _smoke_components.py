"""Temporary component checks for the pipeline fixes."""

import torch

from utils.memory_bank import MemoryBank
from utils.memory_manager import MemoryManager
from utils.csl_loss import CSLLossFunc
from phase3_feature_ddpm import FeatureDDPM
from models import ResNet32, ResNet34
from pipeline_config import get_default_config, get_test_config, merge_config


def check_memory_bank():
    bank = MemoryBank(num_classes=4, feature_dim=8, capacity_per_class=5,
                      tail_threshold_percentile=30.0, tail_refresh_interval=4)
    labels = torch.tensor([0, 0, 1, 2, 3, 3, 3, 3])
    features = torch.rand(8, 8)
    bank.update_batch(labels, features)
    assert bank.total_samples == 8
    assert bank.class_frequencies.tolist() == [2, 1, 1, 4]
    assert len(bank.reservoir_buffers[3]) == 4
    assert bank.tail_classes, "tail classes should be populated after refresh"

    # Prototypes must be unit norm (features are L2-normalized before EMA)
    norms = torch.norm(bank.ema_prototypes, dim=1)
    assert torch.all(norms[:4] > 0.5), norms

    # History is capped
    small = MemoryBank(num_classes=2, feature_dim=4, history_limit=3)
    for _ in range(20):
        small.update(0, torch.rand(4))
    assert len(small.update_history[0]) == 3

    # Round trip
    path = './_tmp_memory_bank.pt'
    bank.save(path)
    restored = MemoryBank(num_classes=4, feature_dim=8, capacity_per_class=5)
    restored.load(path)
    assert restored.total_samples == 8
    assert torch.allclose(restored.ema_prototypes, bank.ema_prototypes)
    assert len(restored.reservoir_buffers[3]) == 4
    assert restored.reservoir_buffers[3][0].dtype == torch.float32
    import os
    os.remove(path)
    print("memory bank OK")


def check_memory_manager():
    model = ResNet32(num_classes=4)
    manager = MemoryManager(model=model, num_classes=4, capacity_per_class=4,
                            device='cpu', save_dir='./_tmp_memory')
    labels = torch.tensor([0, 1, 1, 3])
    features = torch.rand(4, model.get_feature_dim())
    manager.update_memory_from_features(labels, features)
    assert manager.update_stats['total_updates'] == 4
    assert manager.update_stats['updates_per_class'][1] == 2

    path = manager.save_memory(step=1, prefix='memory_bank')
    assert path.endswith('.pt')
    found = manager.load_latest_memory(prefix='memory_bank')
    assert found is not None

    import shutil
    shutil.rmtree('./_tmp_memory')
    print("memory manager OK")


def check_csl_loss():
    loss_fn = CSLLossFunc(target_class_index=[2, 3], num_classes=4,
                          samples_per_class=[100, 80, 20, 10])
    logits = torch.randn(16, 4, requires_grad=True)
    labels = torch.randint(0, 4, (16,))
    value = loss_fn(labels, logits, epoch=0)
    value.backward()
    assert torch.isfinite(value), value
    assert logits.grad is not None

    # Second epoch exercises the prev/current comparison branch
    value2 = loss_fn(labels, logits.detach().requires_grad_(True), epoch=1)
    assert torch.isfinite(value2)
    print(f"csl loss OK (loss={value.item():.4f})")


def check_ddpm_standardization():
    torch.manual_seed(0)
    feature_dim = 16
    # Post-ReLU-like features: non-negative, large scale, some dead dimensions
    real = torch.relu(torch.randn(400, feature_dim) * 6 + 8)
    real[:, -3:] = 0.0

    model = FeatureDDPM(feature_dim=feature_dim, num_classes=3, hidden_dim=64,
                        num_layers=3, num_timesteps=50)
    model.set_feature_stats(real, non_negative=True)

    standardized = model.standardize(real)
    assert standardized.abs().mean() < 3.0, standardized.abs().mean()
    assert torch.isfinite(standardized).all()

    # Dead dimensions must not explode
    assert torch.isfinite(model.standardize(real)).all()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Give each class a distinct offset so class conditioning has something to learn
    labels = torch.randint(0, 3, (400,))
    real = real + labels.unsqueeze(1).float() * 4.0
    model.set_feature_stats(real, non_negative=True)

    losses = []
    for step in range(400):
        idx = torch.randperm(400)[:64]
        loss = model(real[idx], labels[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (losses[0], losses[-1])

    model.eval()
    prototypes = {c: real[labels == c].mean(dim=0) for c in range(3)}

    for num_steps in (25, None):
        generated, confidences = model.sample_with_confidence(
            torch.zeros(64, dtype=torch.long), prototypes, device='cpu',
            num_steps=num_steps, batch_size=32)

        assert generated.shape == (64, feature_dim)
        assert torch.isfinite(generated).all()
        assert generated.min() >= 0.0, "non-negative clamp not applied"
        assert confidences.shape == (64,)
        assert 0.0 <= confidences.min() <= confidences.max() <= 1.0

        real_class0 = real[labels == 0]
        real_mean = real_class0.mean().item()
        gen_mean = generated.mean().item()
        # Generated features must live on roughly the same scale as the real ones. Before
        # standardization and DDIM striding this ratio came out around 90x.
        ratio = gen_mean / max(real_mean, 1e-6)
        assert 0.3 < ratio < 3.0, f"generated scale off by {ratio:.1f}x (steps={num_steps})"

        label = num_steps if num_steps else 'full'
        print(f"  steps={label}: real mean {real_mean:.2f} vs generated mean {gen_mean:.2f} "
              f"(ratio {ratio:.2f}), mean confidence {confidences.mean():.3f}")

    print(f"ddpm OK (loss {losses[0]:.3f} -> {losses[-1]:.3f})")


def check_config():
    default = get_default_config()
    assert default['generation']['tail_improvement_threshold'] == 2.0

    # The long-tailed CIFAR protocol the published accuracies are measured under. A silent
    # drift here makes every number in the report incomparable, so it is pinned.
    training = default['training']
    assert default['model']['architecture'] == 'ResNet32'
    assert training['initial_epochs'] == 200
    assert training['lr'] == 0.1
    assert training['momentum'] == 0.9
    assert training['weight_decay'] == 2e-4
    assert training['scheduler_milestones'] == [160, 180]
    assert training['scheduler_gamma'] == 0.01

    merged = merge_config({'training': {'initial_epochs': 7}})
    assert merged['training']['initial_epochs'] == 7
    assert merged['training']['synthetic_epochs'] == default['training']['synthetic_epochs']
    test = get_test_config()
    assert test['generation']['use_stable_diffusion'] is False
    assert test['paths']['results_dir'] == './test_results'
    print("config OK")


def check_backbone():
    """
    The CIFAR ResNet-32 is the backbone the published CIFAR-LT accuracies are measured with,
    so its shape is a benchmark property rather than an implementation detail. 6n+2 = 32
    weighted layers at 16/32/64 channels comes to 0.46M parameters; a torchvision ResNet with
    a swapped stem would be ~21M and would not be the same experiment.
    """
    model = ResNet32(num_classes=10)
    params = sum(p.numel() for p in model.parameters())
    assert 0.45e6 < params < 0.48e6, f"ResNet32 has {params:,} parameters, expected ~0.46M"

    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(convs) == 31, f"expected 31 convolutions (1 stem + 15 blocks x 2), got {len(convs)}"
    assert model.get_feature_dim() == 64

    logits, features = model(torch.randn(2, 3, 32, 32), return_features=True)
    assert logits.shape == (2, 10), logits.shape
    assert features.shape == (2, 64), features.shape
    # Features are read after the block's final ReLU, which the DDPM relies on when it fits
    # standardization statistics with non_negative=True.
    assert features.min() >= 0.0

    # 100-class head, and a resolution other than 32 (adaptive pooling should absorb it).
    wide = ResNet32(num_classes=100)
    assert wide(torch.randn(2, 3, 64, 64)).shape == (2, 100)

    try:
        ResNet32(num_classes=10, pretrained=True)
    except ValueError:
        pass
    else:
        raise AssertionError("ResNet32 should reject pretrained=True; no such weights exist")

    print(f"backbone OK ({params:,} parameters, {model.get_feature_dim()}-dim features)")


def check_classifier_lookup():
    from orchestrator import MemoryConditionedOrchestrator
    orch = MemoryConditionedOrchestrator.__new__(MemoryConditionedOrchestrator)
    orch.model = ResNet32(num_classes=10)
    orch.config = get_default_config()
    classifier = orch._get_classifier()
    assert classifier is orch.model.fc
    assert orch._get_feature_dim() == 64
    print("classifier lookup OK")

    # The wrapped torchvision backbones nest their classifier one level deeper, so the lookup
    # has to handle both shapes.
    orch.model = ResNet34(num_classes=10)
    assert orch._get_classifier() is orch.model.model.fc
    assert orch._get_feature_dim() == 512
    print("classifier lookup (wrapped backbone) OK")


if __name__ == '__main__':
    check_config()
    check_memory_bank()
    check_memory_manager()
    check_csl_loss()
    check_backbone()
    check_classifier_lookup()
    check_ddpm_standardization()
    print("\nAll component checks passed.")
