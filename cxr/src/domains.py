"""
Turning a discovered spec into images + binary labels.

Every domain ends up identical to the network: 224x224, 3 channels,
label 0 = Normal, 1 = Abnormal. That shared label space is what makes
zero-shot transfer possible.
"""
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .config import (IMG_SIZE, BATCH_SIZE, NUM_WORKERS, SEED,
                     VAL_FRACTION, TEST_FRACTION, get_domain)

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def _to_rgb(img):
    return img.convert("RGB")


def build_transform(train=False):
    t = [transforms.Lambda(_to_rgb), transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        t.append(transforms.RandomHorizontalFlip(0.5))
    t += [transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(t)


class ListDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.transform(Image.open(p)), int(y)


def _scan(dirs):
    out = []
    for d in dirs:
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                out.append(p)
    return out


def _partition(paths, part):
    """
    Deterministic disjoint slice. Used when several 'domains' would otherwise
    share the same pool of Normal images -- that would be leakage.
    Hashing the filename keeps the assignment stable across runs.
    """
    if not part:
        return paths
    i, n = part
    return [p for p in paths
            if int(hashlib.md5(p.name.encode()).hexdigest(), 16) % n == i]


def _index_images(dirs):
    idx = {}
    for d in dirs:
        for p in d.rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                idx.setdefault(p.name, p)
    return idx


def list_items(name):
    """[(path, 0|1), ...] for one domain."""
    d = get_domain(name)
    mode = d["mode"]

    if mode == "globs":
        norm = _partition(_scan(d["normal_dirs"]), d.get("partition"))
        abn = _partition(_scan(d["abnormal_dirs"]), d.get("partition")) \
            if d.get("partition_abnormal") else _scan(d["abnormal_dirs"])
        items = [(p, 0) for p in norm] + [(p, 1) for p in abn]

    elif mode == "filename":
        # label is the last character of the filename stem: _0 / _1
        items = []
        for p in _scan(d["dirs"]):
            key = p.stem[-1]
            if key in d["suffix_map"]:
                items.append((p, d["suffix_map"][key]))

    elif mode == "csv":
        df = pd.read_csv(d["csv"])
        for col, vals in (d.get("filter") or {}).items():
            if col in df.columns:
                df = df[df[col].isin(vals)]
        idx = _index_images(d["image_dirs"])
        normal = set(d["normal_values"])
        items = []
        for _, r in df.iterrows():
            p = idx.get(str(r[d["image_col"]]))
            if p is None:
                continue
            items.append((p, 0 if str(r[d["label_col"]]).strip() in normal else 1))

    elif mode == "indiana":
        proj = pd.read_csv(d["projections"])
        rep = pd.read_csv(d["reports"])
        m = proj.merge(rep, on="uid", how="inner")
        if "projection" in m.columns:
            m = m[m["projection"].astype(str).str.lower().str.startswith("front")]
        lab_col = "Problems" if "Problems" in m.columns else "MeSH"
        idx = _index_images(d["image_dirs"])
        items = []
        for _, r in m.iterrows():
            p = idx.get(str(r.get("filename", "")))
            if p is None:
                continue
            txt = str(r[lab_col]).strip().lower()
            items.append((p, 0 if txt in ("normal", "no indexing") else 1))
    else:
        raise ValueError(f"unknown mode {mode}")

    if not items:
        raise RuntimeError(f"domain '{name}' produced zero images")
    return items


def split_items(items, seed=SEED):
    """Stratified train/val/test -- each split keeps the class ratio."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in (0, 1):
        lst = [i for i in items if i[1] == c]
        idx = rng.permutation(len(lst))
        n = len(lst)
        n_te, n_va = int(round(n * TEST_FRACTION)), int(round(n * VAL_FRACTION))
        te += [lst[i] for i in idx[:n_te]]
        va += [lst[i] for i in idx[n_te:n_te + n_va]]
        tr += [lst[i] for i in idx[n_te + n_va:]]
    return tr, va, te


def get_split(name, split, cap=None, seed=SEED):
    tr, va, te = split_items(list_items(name), seed)
    items = {"train": tr, "val": va, "test": te}[split]
    if cap and len(items) > cap:
        idx = np.random.default_rng(seed).choice(len(items), cap, replace=False)
        items = [items[i] for i in idx]
    return items


def get_loader(name, split, cap=None, train_aug=False, shuffle=False):
    ds = ListDataset(get_split(name, split, cap), build_transform(train_aug))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)


def domain_summary(name):
    items = list_items(name)
    tr, va, te = split_items(items)
    y = np.array([i[1] for i in items])
    d = get_domain(name)
    return dict(domain=name, n_total=len(items), n_train=len(tr),
                n_val=len(va), n_test=len(te),
                n_normal=int((y == 0).sum()), n_abnormal=int((y == 1).sum()),
                abnormal_rate=float(y.mean()),
                kaggle_dir=d.get("kaggle_dir", ""),
                **{f"meta_{k}": v for k, v in d.get("meta", {}).items()})
