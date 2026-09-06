"""
Loading MedMNIST datasets in a uniform way.

The tricky part: some datasets are RGB, some are grayscale, and the ResNet
we use expects 3 channels at a reasonable size. Every dataset must come out
of here looking identical to the model, or the comparison is meaningless.
"""
import numpy as np
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .config import CACHE_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, SEED


def _to_rgb(img):
    """Grayscale -> 3 identical channels. RGB -> unchanged.

    Defined as a real function (not a lambda) so it can be pickled by
    DataLoader workers. A lambda here fails on Windows.
    """
    return img.convert("RGB")


def build_transform():
    return transforms.Compose([
        transforms.Lambda(_to_rgb),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # ImageNet statistics: the pretrained ResNet expects inputs
        # normalised this way.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def n_classes(name):
    """How many classes this dataset has."""
    return len(INFO[name]["label"])


def get_dataset(name, split, cap=None):
    """
    name  : 'pneumoniamnist' etc.
    split : 'train' | 'val' | 'test'
    cap   : if the split is bigger than this, take a random subset.
            Fixed seed, so the same subset every time.
    """
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    ds = DataClass(split=split, transform=build_transform(),
                   download=True, root=str(CACHE_DIR))
    if cap is not None and len(ds) > cap:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(ds), size=cap, replace=False)
        ds = Subset(ds, idx.tolist())
    return ds


def get_loader(name, split, cap=None, shuffle=False):
    ds = get_dataset(name, split, cap)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True,
                      drop_last=False)
