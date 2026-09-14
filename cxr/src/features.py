"""One forward pass gives both the 512-d features and the zero-shot prediction."""
import numpy as np
import torch
from tqdm.auto import tqdm


@torch.no_grad()
def extract(backbone, head, loader, device="cuda", desc=""):
    """
    Returns
    -------
    X : (N, 512) features from the frozen backbone
    P : (N, 2)   softmax probabilities from the source's OWN classifier
    y : (N,)     true labels (used only for scoring, never for adaptation)

    Nothing is updated. torch.no_grad() disables gradient tracking; .eval()
    freezes BatchNorm statistics so target images cannot shift them.
    """
    backbone = backbone.to(device).eval()
    head = head.to(device).eval()
    Xs, Ps, ys = [], [], []
    for xb, yb in tqdm(loader, desc=desc, leave=False):
        f = backbone(xb.to(device, non_blocking=True))
        p = torch.softmax(head(f), dim=1)
        Xs.append(f.cpu().numpy())
        Ps.append(p.cpu().numpy())
        ys.append(np.asarray(yb).ravel())
    return (np.concatenate(Xs).astype(np.float32),
            np.concatenate(Ps).astype(np.float32),
            np.concatenate(ys).astype(np.int64))
