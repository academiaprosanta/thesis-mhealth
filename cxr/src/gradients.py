"""
Gradient subspaces for GSA (proposal eq. 7).

The proposal defines
    G_t := grad_Delta L_t(Delta) at Delta = 0,   shape (C_out, d),  d = 9 * C_in
    V_t := top-r right singular subspace of G_t
    rho_hat(U, t) = (1/r) || P_U P_V_t ||_F^2

In your setting there is no low-rank adapter, so the natural analogue at a
chosen conv layer is the gradient of the loss with respect to that layer's
weight, unrolled from (C_out, C_in, 3, 3) to (C_out, 9*C_in).

THE LABEL PROBLEM. A loss normally needs labels, and target labels are exactly
what the unsupervised setting forbids. Three loss choices are provided:

  "entropy"  L = mean predictive entropy.        NO labels. Fully label-free.
  "pseudo"   L = cross-entropy vs the model's    NO labels. Label-free.
             own argmax predictions.
  "oracle"   L = cross-entropy vs true labels.   USES LABELS. Diagnostic only --
                                                 an upper bound, not a feature.

Use "entropy" or "pseudo" for anything that goes into the unsupervised set.
Report "oracle" separately to show how much the label-free proxy costs you.
"""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def pick_layer(model, name="layer4.1.conv2"):
    """Return the weight Parameter of a named conv layer."""
    mod = model
    for part in name.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod.weight


@torch.enable_grad()
def layer_gradient(model, loader, layer="layer4.1.conv2", loss="entropy",
                   device="cuda", max_batches=20, desc=""):
    """
    Accumulate dL/dW at the given layer over up to max_batches batches.

    Returns G of shape (C_out, 9 * C_in) -- the unrolled 3x3 patch space from
    the proposal's formalism.
    """
    model = model.to(device).eval()          # eval: BatchNorm stays frozen
    for p in model.parameters():
        p.requires_grad_(False)
    W = pick_layer(model, layer)
    W.requires_grad_(True)

    total = torch.zeros_like(W)
    n = 0
    for bi, (xb, yb) in enumerate(tqdm(loader, desc=desc, leave=False)):
        if bi >= max_batches:
            break
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)

        if loss == "entropy":
            p = F.softmax(logits, dim=1)
            L = -(p * torch.log(p + 1e-12)).sum(1).mean()
        elif loss == "pseudo":
            L = F.cross_entropy(logits, logits.argmax(1).detach())
        elif loss == "oracle":
            L = F.cross_entropy(logits, yb.long().to(device))
        else:
            raise ValueError(loss)

        g = torch.autograd.grad(L, W, retain_graph=False)[0]
        total += g.detach() * len(xb)
        n += len(xb)

    W.requires_grad_(False)
    G = (total / max(n, 1)).cpu().numpy()
    return G.reshape(G.shape[0], -1)         # (C_out, 9*C_in)


def source_basis(model, loader, layer="layer4.1.conv2", r=32, device="cuda",
                 desc=""):
    """
    The 'source-learned basis' U. Computed as the top-r right singular subspace
    of the SOURCE gradient, using the source's own labels -- which you have,
    because you trained on that data. No target labels involved.
    """
    from subspace import top_right_subspace
    G = layer_gradient(model, loader, layer, "oracle", device,
                       max_batches=20, desc=desc)
    return top_right_subspace(G, r), G
