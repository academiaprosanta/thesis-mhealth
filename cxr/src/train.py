"""Fine-tune one ResNet-18 on one domain's Normal-vs-Abnormal task."""
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .config import CKPT_DIR, EPOCHS, LR, WEIGHT_DECAY, MAX_TRAIN, MAX_EVAL, SEED
from .domains import get_loader, get_split
from .models import build_model, split_model
from .features import extract
from .metrics import compute_metrics


def train_source_model(name, epochs=EPOCHS, device="cuda", force=False):
    """Returns (model, in_domain_metrics). Cached, so reruns skip finished work."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ckpt = CKPT_DIR / f"{name}.pt"
    model = build_model(pretrained=True)

    if ckpt.exists() and not force:
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"[{name}] loaded checkpoint")
    else:
        loader = get_loader(name, "train", cap=MAX_TRAIN, train_aug=True, shuffle=True)

        # class weighting: chest X-ray datasets are often 3:1 imbalanced
        ytr = np.array([y for _, y in get_split(name, "train", cap=MAX_TRAIN)])
        cnt = np.bincount(ytr, minlength=2).astype(float)
        w = torch.tensor((cnt.sum() / (2 * np.maximum(cnt, 1))), dtype=torch.float32)
        print(f"[{name}] train n={len(ytr)}  abnormal={ytr.mean():.3f}  weights={w.tolist()}")

        model = model.to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        lossf = nn.CrossEntropyLoss(weight=w.to(device))

        for ep in range(epochs):
            tot = 0.0
            for xb, yb in tqdm(loader, desc=f"[{name}] ep{ep+1}/{epochs}", leave=False):
                xb = xb.to(device, non_blocking=True)
                yb = yb.long().to(device, non_blocking=True)
                opt.zero_grad()
                loss = lossf(model(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            sched.step()
            print(f"[{name}] epoch {ep+1}: loss {tot/len(loader.dataset):.4f}")

        torch.save(model.state_dict(), ckpt)

    bb, head = split_model(model)
    loader = get_loader(name, "test", cap=MAX_EVAL)
    _, P, y = extract(bb, head, loader, device, desc=f"{name}/self-test")
    m = compute_metrics(y, P[:, 1])
    print(f"[{name}] in-domain AUC {m['auc']:.4f}")
    return model, m
