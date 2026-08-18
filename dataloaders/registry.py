"""
Dataset dispatch.

The CIFAR benchmarks and the large image benchmarks are prepared in fundamentally different
ways — one is subsampled from a balanced dataset with a chosen imbalance ratio, the other is
defined by a published split file — but the orchestrator only needs the same bundle out of
either. Routing by name here keeps that branch out of the pipeline itself.
"""

from . import cifar_lt_loader
from . import lt_image_loader


def _module_for(dataset):
    """The loader module responsible for a dataset name."""
    if lt_image_loader.is_lt_image_dataset(dataset):
        return lt_image_loader
    try:
        cifar_lt_loader._normalize_name(dataset)
    except ValueError:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Supported: {', '.join(available_datasets())}"
        ) from None
    return cifar_lt_loader


def available_datasets():
    return ['cifar10', 'cifar100'] + list(lt_image_loader.DATASET_SPECS)


def uses_imbalance_ratio(dataset):
    """
    Whether an imbalance ratio is a meaningful parameter for this dataset.

    False for ImageNet-LT and iNaturalist-2018, whose imbalance is fixed by their published
    splits — passing a ratio there is a configuration error rather than a preference.
    """
    return _module_for(dataset) is cifar_lt_loader


def get_normalization(dataset):
    """Per-channel mean and std used to normalize this dataset."""
    return _module_for(dataset).get_normalization(dataset)


def get_num_classes(dataset):
    """Number of classes in this dataset."""
    return _module_for(dataset).get_num_classes(dataset)


def get_default_image_size(dataset):
    """Input resolution the published benchmark for this dataset is reported at."""
    return _module_for(dataset).get_default_image_size(dataset)


def get_lt_loaders(dataset, **kwargs):
    """
    Build the long-tailed loaders for any supported dataset.

    Returns a bundle with train_loader, train_eval_loader, test_loader, class_names,
    samples_per_class, num_classes and image_size.
    """
    module = _module_for(dataset)
    if module is cifar_lt_loader:
        return cifar_lt_loader.get_cifar_lt_loaders(dataset=dataset, **kwargs)

    # The CIFAR loader's data_dir is a download destination; the large benchmarks instead need
    # a root for an image tree that was extracted by hand.
    kwargs.pop('data_dir', None)
    kwargs.pop('seed', None)
    return lt_image_loader.get_lt_image_loaders(dataset=dataset, **kwargs)
