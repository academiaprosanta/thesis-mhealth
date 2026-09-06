"""Fine-tune one ResNet-18 on one dataset. This produces a 'source model'."""
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .config import CKPT_DIR, EPOCHS, LR, WEIGHT_DECAY, MAX_TRAIN, MAX_EVAL, SEED
from .data import get_loader, n_classes
from .models import build_model
from .metrics import compute_metrics


def train_source_model(name, epochs=EPOCHS, device="cuda", force=False):
    """
    Trains on `name` and saves the weights.
    Returns (model, in_domain_metrics).

    If a checkpoint already exists it is loaded instead of retrained --
    so re-running the script after a crash does not redo finished work.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ckpt = CKPT_DIR / f"{name}.pt"
    n_out = n_classes(name)
    model = build_model(n_out, pretrained=True)

    if ckpt.exists() and not force:
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"[{name}] loaded existing checkpoint")
    else:
        train_loader = get_loader(name, "train", cap=MAX_TRAIN, shuffle=True)
        model = model.to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        lossf = nn.CrossEntropyLoss()

        for ep in range(epochs):
            running = 0.0
            for xb, yb in tqdm(train_loader, desc=f"[{name}] epoch {ep+1}/{epochs}", leave=False):
                xb = xb.to(device, non_blocking=True)
                # MedMNIST labels arrive shaped (B, 1). CrossEntropyLoss wants (B,).
                yb = yb.squeeze(-1).long().to(device, non_blocking=True)
                opt.zero_grad()
                loss = lossf(model(xb), yb)
                loss.backward()
                opt.step()
                running += loss.item() * len(xb)
            sched.step()
            print(f"[{name}] epoch {ep+1}: loss {running/len(train_loader.dataset):.4f}")

        torch.save(model.state_dict(), ckpt)

    # in-domain performance: how good is this source model on its OWN data?
    m = evaluate_in_domain(model, name, device)
    print(f"[{name}] in-domain AUC {m['auc']:.4f}")
    return model, m


@torch.no_grad()
def evaluate_in_domain(model, name, device="cuda"):
    model = model.to(device).eval()
    loader = get_loader(name, "test", cap=MAX_EVAL)
    probs, ys = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        ys.append(np.asarray(yb).ravel())
    return compute_metrics(np.concatenate(ys), np.concatenate(probs))
