"""
Focused test for the large-benchmark loader.

Builds a miniature ImageNet-LT tree from the real split files — the first few hundred entries,
with placeholder images — and checks the bundle the orchestrator consumes. This verifies the
wiring without the 145GB download; the only thing it cannot verify is the image content.

    py _smoke_lt_loader.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# Windows defaults stdout to cp1252, which cannot encode the status glyphs printed below.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

from dataloaders import lt_image_loader

PER_CLASS = 1


def take_per_class(paths, labels, per_class):
    """
    First `per_class` entries of every class.

    Sampling has to span all 1000 classes rather than taking a prefix of the file: the split is
    grouped by class, so a prefix covers only a handful of them and the class-name lookup
    correctly refuses to map a partial label set.
    """
    seen = {}
    kept_paths, kept_labels = [], []
    for path, label in zip(paths, labels):
        if seen.get(label, 0) >= per_class:
            continue
        seen[label] = seen.get(label, 0) + 1
        kept_paths.append(path)
        kept_labels.append(label)
    return kept_paths, kept_labels


def build_fake_tree(root, paths):
    """Write a small placeholder JPEG at each relative path."""
    for relative in paths:
        target = Path(root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        noise = (np.random.rand(40, 40, 3) * 255).astype('uint8')
        Image.fromarray(noise).save(target, format='JPEG')


def main():
    real_read_split = lt_image_loader.read_split

    train_paths, train_labels = real_read_split('imagenet_lt', 'train')
    eval_paths, eval_labels = real_read_split('imagenet_lt', 'test')

    subset = {
        'train': take_per_class(train_paths, train_labels, PER_CLASS),
        'test': take_per_class(eval_paths, eval_labels, PER_CLASS),
    }
    total = len(subset['train'][0]) + len(subset['test'][0])

    root = tempfile.mkdtemp(prefix='lt_smoke_')
    try:
        build_fake_tree(root, subset['train'][0] + subset['test'][0])
        print(f"✓ fake tree with {total} images at {root}")

        lt_image_loader.read_split = lambda dataset, split: subset[split]
        bundle = lt_image_loader.get_lt_image_loaders(
            'imagenet_lt', batch_size=16, num_workers=0, data_root=root
        )

        expected_keys = {'train_loader', 'train_eval_loader', 'test_loader', 'class_names',
                         'samples_per_class', 'num_classes', 'image_size'}
        missing = expected_keys - set(bundle)
        assert not missing, f"bundle is missing {missing}"
        print(f"✓ bundle keys match what the orchestrator reads")

        assert bundle['num_classes'] == 1000, bundle['num_classes']
        assert bundle['image_size'] == 224, bundle['image_size']
        assert len(bundle['samples_per_class']) == 1000
        assert sum(bundle['samples_per_class']) == len(subset['train'][0])
        assert min(bundle['samples_per_class']) == PER_CLASS, "every class should be covered"
        print(f"✓ {bundle['num_classes']} classes at {bundle['image_size']}px, "
              f"{sum(bundle['samples_per_class'])} train samples")

        names = bundle['class_names']
        assert len(names) == 1000, len(names)
        assert names[0] == 'tench', names[0]
        assert names[999] == 'toilet tissue', names[999]
        print(f"✓ class names resolved (0='{names[0]}', 999='{names[999]}')")

        images, labels = next(iter(bundle['train_loader']))
        assert images.shape == (16, 3, 224, 224), images.shape
        assert labels.shape == (16,), labels.shape
        print(f"✓ train batch {tuple(images.shape)}, labels {tuple(labels.shape)}")

        eval_images, _ = next(iter(bundle['test_loader']))
        assert eval_images.shape == (16, 3, 224, 224), eval_images.shape
        print(f"✓ eval batch {tuple(eval_images.shape)}")

        # A ResNet-50 must accept these and report 2048-dim features, which is what the memory
        # bank and the feature DDPM are sized from.
        from models import ResNet50
        model = ResNet50(num_classes=1000, pretrained=False, image_size=224)
        logits, features = model(eval_images[:2], return_features=True)
        assert logits.shape == (2, 1000), logits.shape
        assert features.shape == (2, 2048), features.shape
        print(f"✓ ResNet50 forward: logits {tuple(logits.shape)}, "
              f"features {tuple(features.shape)}")

        print("\nAll checks passed.")
        return 0
    finally:
        lt_image_loader.read_split = real_read_split
        shutil.rmtree(root, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
