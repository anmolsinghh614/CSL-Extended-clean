"""
Long-tailed loaders for the large image benchmarks (ImageNet-LT, iNaturalist-2018).

These differ from the CIFAR benchmarks in one important way: the imbalance is not something
this code creates. CIFAR-10/100-LT are produced by subsampling a balanced dataset with an
exponential profile, so an imbalance ratio is a parameter. Here the long tail is *defined* by
the published split files under `dataloaders/ImageNet_LT/` and `dataloaders/Inaturalist18/`,
which every method in the literature trains on so that numbers are comparable. The class
counts are read back out of the split rather than imposed on it.

The returned bundle deliberately has the same shape as `get_cifar_lt_loaders`, so the
orchestrator consumes either without branching.
"""

import os
from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGENET_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
INATURALIST_STATS = ((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))

# `eval_split` differs between the two, and getting it wrong produces a plausible-looking
# number that cannot be compared with anything published:
#   ImageNet-LT — `test` is the balanced 50-per-class set built from the ILSVRC-2012 validation
#     images (50,000 total), which is what the literature reports on. Its `val` file is a
#     20-per-class sample drawn from the training directory, for model selection.
#   iNaturalist-2018 — `val` is the official 3-per-class validation set (24,426 total) that the
#     published accuracies use. Its `test` file is unlabelled and cannot be evaluated at all.
DATASET_SPECS = {
    'imagenet_lt': {
        'display': 'ImageNet-LT',
        'num_classes': 1000,
        'stats': IMAGENET_STATS,
        'image_size': 224,
        'split_dir': 'ImageNet_LT',
        'split_prefix': 'ImageNet_LT',
        'env_var': 'IMAGENET_LT_ROOT',
        'eval_split': 'test',
    },
    'inaturalist': {
        'display': 'iNaturalist-2018',
        'num_classes': 8142,
        'stats': INATURALIST_STATS,
        'image_size': 224,
        'split_dir': 'Inaturalist18',
        'split_prefix': 'iNaturalist18',
        'env_var': 'INATURALIST_ROOT',
        'eval_split': 'val',
    },
}

_ALIASES = {
    'imagenetlt': 'imagenet_lt',
    'imagenet': 'imagenet_lt',
    'inaturalist': 'inaturalist',
    'inaturalist18': 'inaturalist',
    'inat': 'inaturalist',
    'inat18': 'inaturalist',
}

REPO_ROOT = Path(__file__).resolve().parent.parent


def normalize_name(name):
    """Canonical key for a dataset name, or None if it is not one of these benchmarks."""
    key = str(name).lower().replace('-', '').replace('_', '').replace(' ', '')
    if key in DATASET_SPECS:
        return key
    return _ALIASES.get(key)


def is_lt_image_dataset(name):
    return normalize_name(name) is not None


def get_normalization(dataset):
    return DATASET_SPECS[normalize_name(dataset)]['stats']


def get_num_classes(dataset):
    return DATASET_SPECS[normalize_name(dataset)]['num_classes']


def get_default_image_size(dataset):
    return DATASET_SPECS[normalize_name(dataset)]['image_size']


def split_path(dataset, split):
    """Location of a published split file, e.g. ImageNet_LT_train.txt."""
    spec = DATASET_SPECS[normalize_name(dataset)]
    return REPO_ROOT / 'dataloaders' / spec['split_dir'] / f"{spec['split_prefix']}_{split}.txt"


def resolve_data_root(dataset, data_root=None):
    """
    Where the extracted images live.

    An explicit argument wins; otherwise the dataset's environment variable is used, which
    keeps a machine-specific absolute path out of the committed config.
    """
    spec = DATASET_SPECS[normalize_name(dataset)]
    root = data_root or os.environ.get(spec['env_var'])
    if not root:
        raise FileNotFoundError(
            f"{spec['display']} images are not configured. Set the {spec['env_var']} "
            f"environment variable to the directory holding the extracted image tree, or "
            f"pass dataset.data_root in the config."
        )
    if not Path(root).is_dir():
        raise FileNotFoundError(f"{spec['env_var']} points at {root}, which is not a directory")
    return str(root)


def check_data_root(dataset, data_root, paths, sample=32):
    """
    Confirm the root actually contains the images the split file names.

    The split paths are relative and carry their own `train/` or `val/` prefix, so the root has
    to be the directory those subtrees sit in. Pointing one level too deep or too shallow is the
    usual mistake, and without this it surfaces as a failure on the first batch — after the
    model, memory bank and diffusion pipeline have all been constructed.
    """
    missing = [p for p in paths[:sample] if not Path(data_root, p).is_file()]
    if not missing:
        return

    spec = DATASET_SPECS[normalize_name(dataset)]
    prefixes = sorted({Path(p).parts[0] for p in paths[:sample]})
    raise FileNotFoundError(
        f"{len(missing)} of the first {min(sample, len(paths))} {spec['display']} images were "
        f"not found under {data_root} (e.g. {missing[0]}). The split files use paths relative "
        f"to a directory containing {', '.join(prefixes)}/, so {spec['env_var']} should point "
        f"at that parent rather than at one of the subdirectories."
    )


# ─── Dataset ─────────────────────────────────────────────────────────────────

def read_split(dataset, split):
    """
    Parse a split file into parallel lists of relative paths and integer labels.

    A file without labels is rejected rather than silently loaded, since a label-free
    evaluation set would report meaningless accuracy.
    """
    path = split_path(dataset, split)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    paths, labels = [], []
    with open(path, 'r') as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{line_number} has {len(parts)} field(s), expected 'path label'"
                )
            paths.append(parts[0])
            labels.append(int(parts[1]))

    if not paths:
        raise ValueError(f"{path} contains no entries")
    return paths, labels


class LTImageDataset(Dataset):
    """Images listed by a split file, loaded from a root directory."""

    def __init__(self, data_root, paths, labels, transform=None):
        self.data_root = data_root
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        full_path = os.path.join(self.data_root, self.paths[index])
        # A silent substitution here would quietly change the class distribution the whole
        # method is reasoning about, so a missing or unreadable file is an error.
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as error:
            raise RuntimeError(f"Could not read {full_path}: {error}") from error

        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


def build_transforms(dataset, image_size=None):
    """
    Training and evaluation transforms for the large benchmarks.

    Random-resized-crop, horizontal flip and colour jitter for training; resize-then-centre-crop
    for evaluation. This is the recipe the long-tailed ImageNet literature uses, and it matches
    what the previous loaders in this repo applied.
    """
    name = normalize_name(dataset)
    spec = DATASET_SPECS[name]
    if image_size is None:
        image_size = spec['image_size']
    normalize = transforms.Normalize(*spec['stats'])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        normalize,
    ])
    transform_eval = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return transform_train, transform_eval


# ─── Class names ─────────────────────────────────────────────────────────────

def _imagenet_class_names(paths, labels, num_classes):
    """
    Readable names for the 1000 ImageNet classes, keyed to the split file's label indices.

    The split paths carry the WordNet id as their parent directory (`train/n01440764/...`), so
    the label-to-synset mapping can be recovered from the data itself. Whether index i then
    corresponds to torchvision's i-th category depends on the labels following the usual
    sorted-wnid convention, which is checked rather than assumed: getting this wrong would feed
    the wrong class name into every generated prompt, and nothing downstream would notice.
    """
    wnid_by_label = {}
    for path, label in zip(paths, labels):
        wnid = Path(path).parent.name
        existing = wnid_by_label.setdefault(label, wnid)
        if existing != wnid:
            print(f"  Warning: label {label} appears under both {existing} and {wnid}; "
                  f"falling back to numeric class names")
            return [f"class_{i}" for i in range(num_classes)]

    if len(wnid_by_label) != num_classes:
        print(f"  Warning: split covers {len(wnid_by_label)} of {num_classes} classes; "
              f"using WordNet ids where known")
        return [wnid_by_label.get(i, f"class_{i}") for i in range(num_classes)]

    ordered_wnids = [wnid_by_label[i] for i in range(num_classes)]
    if ordered_wnids != sorted(ordered_wnids):
        print("  Warning: labels are not in sorted-WordNet-id order, so torchvision's "
              "category list does not apply; using WordNet ids as class names")
        return ordered_wnids

    try:
        from torchvision.models import ResNet50_Weights
        categories = list(ResNet50_Weights.IMAGENET1K_V1.meta['categories'])
    except Exception:
        return ordered_wnids

    if len(categories) != num_classes:
        return ordered_wnids

    # Keep the synset alongside the word so a mislabelled prompt is traceable later.
    return [categories[i] for i in range(num_classes)]


# iNaturalist paths are train_val2018/<supercategory>/<category_id>/<hash>.jpg, so the
# biological kingdom or class is recoverable even though the species name is not — that lives
# in the competition's taxonomy JSON, which is not part of the split release. A coarse but real
# noun gives Stable Diffusion something to work with, where a bare index would give it nothing.
INATURALIST_SUPERCATEGORY_NOUNS = {
    'Actinopterygii': 'fish',
    'Amphibia': 'amphibian',
    'Animalia': 'animal',
    'Arachnida': 'arachnid',
    'Aves': 'bird',
    'Bacteria': 'bacterium',
    'Chromista': 'alga',
    'Fungi': 'fungus',
    'Insecta': 'insect',
    'Mammalia': 'mammal',
    'Mollusca': 'mollusc',
    'Plantae': 'plant',
    'Protozoa': 'protozoan',
    'Reptilia': 'reptile',
}


def _inaturalist_class_names(paths, labels, num_classes):
    """
    Coarse but real names for the 8142 iNaturalist species.

    The species index is kept in the name because the pipeline uses class names as dictionary
    keys when reporting per-class results: several thousand classes share a supercategory, and
    collapsing them to the bare noun would silently drop all but one of each group from the
    report.
    """
    supercategory_by_label = {}
    for path, label in zip(paths, labels):
        parts = Path(path).parts
        if len(parts) >= 3:
            supercategory_by_label.setdefault(label, parts[1])

    names = []
    for i in range(num_classes):
        supercategory = supercategory_by_label.get(i)
        noun = INATURALIST_SUPERCATEGORY_NOUNS.get(supercategory)
        names.append(f"{noun} species {i}" if noun else f"class_{i}")
    return names


def build_class_names(dataset, paths, labels, num_classes):
    """Human-readable class names, used to condition prompt generation."""
    name = normalize_name(dataset)
    if name == 'imagenet_lt':
        return _imagenet_class_names(paths, labels, num_classes)
    if name == 'inaturalist':
        return _inaturalist_class_names(paths, labels, num_classes)
    return [f"class_{i}" for i in range(num_classes)]


# ─── Entry point ─────────────────────────────────────────────────────────────

def get_lt_image_loaders(dataset, batch_size=256, num_workers=8, data_root=None,
                         image_size=None, imbalance_ratio=None, eval_split=None,
                         **_ignored):
    """
    Build long-tailed loaders for ImageNet-LT or iNaturalist-2018.

    Returns the same bundle as `get_cifar_lt_loaders`: train, train_eval and test loaders,
    class names, per-class sample counts, class count and the resolution used.

    `eval_split` defaults to whichever split each benchmark's published accuracies are measured
    on, which is not the same file name for both — see DATASET_SPECS above.

    `imbalance_ratio` is accepted only so callers can pass a uniform argument list; the
    imbalance is fixed by the published split and cannot be varied, so a value is rejected
    rather than ignored.
    """
    name = normalize_name(dataset)
    if name is None:
        raise ValueError(f"{dataset} is not one of {list(DATASET_SPECS)}")

    spec = DATASET_SPECS[name]
    if imbalance_ratio is not None:
        raise ValueError(
            f"{spec['display']} has a fixed, naturally long-tailed split, so an imbalance "
            f"ratio cannot be applied. Pass imbalance_ratio=None."
        )

    root = resolve_data_root(name, data_root)
    if image_size is None:
        image_size = spec['image_size']
    if eval_split is None:
        eval_split = spec['eval_split']
    num_classes = spec['num_classes']

    train_paths, train_labels = read_split(name, 'train')
    eval_paths, eval_labels = read_split(name, eval_split)

    out_of_range = [l for l in (min(train_labels), max(train_labels))
                    if l < 0 or l >= num_classes]
    if out_of_range:
        raise ValueError(f"{spec['display']} split has labels outside [0, {num_classes})")

    check_data_root(name, root, train_paths)
    check_data_root(name, root, eval_paths)

    transform_train, transform_eval = build_transforms(name, image_size)

    train_set = LTImageDataset(root, train_paths, train_labels, transform_train)
    # Same images, deterministic transform. Feature extraction and exemplar selection read
    # from this, where augmentation noise would corrupt the comparison.
    train_eval_set = LTImageDataset(root, train_paths, train_labels, transform_eval)
    test_set = LTImageDataset(root, eval_paths, eval_labels, transform_eval)

    counts = Counter(train_labels)
    samples_per_class = [counts.get(i, 0) for i in range(num_classes)]

    loader_kwargs = {'num_workers': num_workers, 'pin_memory': True}
    if num_workers > 0:
        # Re-spawning workers each epoch is a noticeable share of an epoch at this scale.
        loader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              # A trailing batch of one would break BatchNorm in train mode.
                              drop_last=True, **loader_kwargs)
    train_eval_loader = DataLoader(train_eval_set, batch_size=batch_size, shuffle=False,
                                   **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **loader_kwargs)

    print(f"  {spec['display']}: {len(train_paths)} train / {len(eval_paths)} {eval_split} "
          f"images over {num_classes} classes")
    print(f"  Imbalance ratio: {max(samples_per_class)} to {min(samples_per_class)} per class")

    return {
        'train_loader': train_loader,
        'train_eval_loader': train_eval_loader,
        'test_loader': test_loader,
        'class_names': build_class_names(name, train_paths, train_labels, num_classes),
        'samples_per_class': samples_per_class,
        'num_classes': num_classes,
        'image_size': image_size,
    }
