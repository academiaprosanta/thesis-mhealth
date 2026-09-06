"""Push images through a frozen backbone and collect the resulting vectors."""
import numpy as np
import torch
from tqdm.auto import tqdm


@torch.no_grad()
def extract_features(backbone, loader, device="cuda", desc=""):
    """
    Returns
    -------
    X : (N, 512) float32   -- one row per image
    y : (N,)     int64     -- its label

    torch.no_grad() switches off gradient tracking. We are not training here,
    so this saves memory and roughly doubles the speed.
    """
    backbone = backbone.to(device).eval()
    feats, labels = [], []
    for xb, yb in tqdm(loader, desc=desc, leave=False):
        out = backbone(xb.to(device, non_blocking=True))
        feats.append(out.cpu().numpy())
        labels.append(np.asarray(yb).ravel())
    return (np.concatenate(feats).astype(np.float32),
            np.concatenate(labels).astype(np.int64))
