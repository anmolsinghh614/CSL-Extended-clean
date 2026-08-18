"""
Focused test for the large-benchmark loaders.

Builds miniature ImageNet-LT and iNaturalist-2018 trees from the real split files — one image
per class, as placeholders — and checks the bundle the orchestrator consumes. This verifies the
wiring without the 265GB of downloads; the only thing it cannot verify is image content.

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


EXPECTED = {
    'imagenet_lt': {
        'num_classes': 1000,
        'eval_split': 'test',
        'names': {0: 'tench', 999: 'toilet tissue'},
    },
    'inaturalist': {
        'num_classes': 8142,
        'eval_split': 'val',
        'names': {0: 'animal species 0', 7477: 'plant species 7477'},
    },
}


def check_dataset(dataset):
    """Build a miniature tree for one dataset and verify the bundle."""
    expected = EXPECTED[dataset]
    num_classes = expected['num_classes']
    eval_split = expected['eval_split']

    print(f"\n--- {dataset} ---")
    real_read_split = lt_image_loader.read_split

    # The split each benchmark is scored on is a property of the benchmark, and mixing them up
    # silently produces numbers that look reasonable but compare to nothing.
    assert lt_image_loader.DATASET_SPECS[dataset]['eval_split'] == eval_split
    print(f"✓ evaluates on the '{eval_split}' split")

    train_paths, train_labels = real_read_split(dataset, 'train')
    eval_paths, eval_labels = real_read_split(dataset, eval_split)

    subset = {
        'train': take_per_class(train_paths, train_labels, PER_CLASS),
        eval_split: take_per_class(eval_paths, eval_labels, PER_CLASS),
    }
    total = len(subset['train'][0]) + len(subset[eval_split][0])

    root = tempfile.mkdtemp(prefix=f'lt_smoke_{dataset}_')
    try:
        build_fake_tree(root, subset['train'][0] + subset[eval_split][0])
        print(f"✓ fake tree with {total} images")

        lt_image_loader.read_split = lambda _dataset, split: subset[split]
        bundle = lt_image_loader.get_lt_image_loaders(
            dataset, batch_size=16, num_workers=0, data_root=root
        )

        expected_keys = {'train_loader', 'train_eval_loader', 'test_loader', 'class_names',
                         'samples_per_class', 'num_classes', 'image_size'}
        missing = expected_keys - set(bundle)
        assert not missing, f"bundle is missing {missing}"

        assert bundle['num_classes'] == num_classes, bundle['num_classes']
        assert bundle['image_size'] == 224, bundle['image_size']
        assert len(bundle['samples_per_class']) == num_classes
        assert sum(bundle['samples_per_class']) == len(subset['train'][0])
        assert min(bundle['samples_per_class']) == PER_CLASS, "every class should be covered"
        print(f"✓ bundle: {num_classes} classes at 224px, "
              f"{sum(bundle['samples_per_class'])} train samples")

        names = bundle['class_names']
        assert len(names) == num_classes, len(names)
        assert len(set(names)) == num_classes, "class names must be unique to key the report"
        for index, expected_name in expected['names'].items():
            assert names[index] == expected_name, f"{index}: {names[index]!r}"
        shown = ', '.join(f"{i}='{names[i]}'" for i in expected['names'])
        print(f"✓ class names unique and resolved ({shown})")

        images, labels = next(iter(bundle['train_loader']))
        assert images.shape == (16, 3, 224, 224), images.shape
        assert labels.shape == (16,), labels.shape
        eval_images, _ = next(iter(bundle['test_loader']))
        assert eval_images.shape == (16, 3, 224, 224), eval_images.shape
        print(f"✓ batches {tuple(images.shape)} from train and eval")

        return eval_images
    finally:
        lt_image_loader.read_split = real_read_split
        shutil.rmtree(root, ignore_errors=True)


def check_memory_bank_sizing():
    """The reservoir has to be sized per dataset or iNaturalist reserves ~17GB."""
    from pipeline_config import memory_bank_capacity_for

    cifar = memory_bank_capacity_for(10, 64)
    imagenet = memory_bank_capacity_for(1000, 2048)
    inat = memory_bank_capacity_for(8142, 2048)

    assert cifar == 256, cifar
    assert imagenet <= 256 and imagenet >= 16, imagenet
    assert inat >= 16, inat
    assert inat < imagenet, "more classes must not mean a larger per-class reservoir"

    for classes, dim, capacity in ((10, 64, cifar), (1000, 2048, imagenet),
                                   (8142, 2048, inat)):
        gigabytes = classes * capacity * dim * 4 / (1024 ** 3)
        assert gigabytes <= 2.0, f"{classes} classes would reserve {gigabytes:.1f}GB"
        print(f"✓ {classes:5d} classes x {dim:4d} dims -> capacity {capacity:3d} "
              f"({gigabytes:.2f}GB reserved)")


def main():
    print("--- memory bank sizing ---")
    check_memory_bank_sizing()

    eval_images = None
    for dataset in ('imagenet_lt', 'inaturalist'):
        eval_images = check_dataset(dataset)

    # A ResNet-50 must accept these and report 2048-dim features, which is what the memory
    # bank and the feature DDPM are sized from.
    print("\n--- backbone ---")
    from models import ResNet50
    model = ResNet50(num_classes=8142, pretrained=False, image_size=224)
    logits, features = model(eval_images[:2], return_features=True)
    assert logits.shape == (2, 8142), logits.shape
    assert features.shape == (2, 2048), features.shape
    print(f"✓ ResNet50 forward: logits {tuple(logits.shape)}, "
          f"features {tuple(features.shape)}")

    print("\nAll checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
