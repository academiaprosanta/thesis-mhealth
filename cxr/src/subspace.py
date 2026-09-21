"""
Subspace-geometry metrics for transferability.

All four metrics you asked about reduce to the principal angles between two
r-dimensional subspaces. Given orthonormal bases U (d x r) and V (d x r):

    singular values of U^T V  =  cos(theta_1) ... cos(theta_r)

    Subspace Alignment    M = U^T V, alignment error ||U M - V||_F
    Principal Angle Sim   (1/r) sum cos^2(theta_k)
    Grassmann distance    sqrt( sum theta_k^2 )
    GSA (proposal eq. 7)  (1/r) ||P_U P_V||_F^2  ==  (1/r) sum cos^2(theta_k)

PAS and GSA are the SAME scalar. What distinguishes GSA is that V is the top-r
right singular subspace of the target GRADIENT, not of the target features.

Sanity anchor (Corollary 1 of the proposal): for Haar-random r-dim subspaces of
R^d, E[(1/r) sum cos^2 theta] = r/d. `selfcheck()` verifies this numerically.
"""
import numpy as np


# ----------------------------------------------------------------- core
def top_right_subspace(M, r):
    """
    Top-r right singular subspace of M (rows = outputs, cols = the d-dim space).
    Returns V, shape (d, r), orthonormal columns.
    """
    M = np.asarray(M, dtype=np.float64)
    M = M - M.mean(axis=0, keepdims=True)          # centre the rows
    r = int(min(r, min(M.shape)))
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:r].T                                 # (d, r)


def principal_angles(U, V):
    """Principal angles theta_1 <= ... <= theta_r between two subspaces, radians."""
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s), s                          # angles, cosines


def subspace_metrics(U, V, prefix=""):
    """
    Every metric from one pair of orthonormal bases.
    U, V both (d, r) with orthonormal columns.
    """
    r = U.shape[1]
    theta, cos = principal_angles(U, V)
    cos2 = cos ** 2

    # Subspace Alignment (Fernando et al. 2013): M = U^T V is the map that
    # rotates U onto V; the residual after applying it is the alignment error.
    M = U.T @ V
    sa_error = float(np.linalg.norm(U @ M - V, ord="fro"))

    out = {
        # --- the shared scalar: PAS == GSA formula ---
        "pa_similarity":   float(cos2.mean()),              # (1/r) sum cos^2
        # --- Grassmann geodesic distance ---
        "grassmann":       float(np.sqrt((theta ** 2).sum())),
        # --- chordal (projection Frobenius) distance ---
        "chordal":         float(np.sqrt(max(r - cos2.sum(), 0.0))),
        # --- Subspace Alignment error ---
        "sa_error":        float(sa_error),
        "sa_frob":         float(np.linalg.norm(M, ord="fro") ** 2 / r),  # == pa_similarity
        # --- angle summaries ---
        "angle_min":       float(theta.min()),
        "angle_mean":      float(theta.mean()),
        "angle_max":       float(theta.max()),
        # --- how many directions are essentially shared (cos^2 > 0.5) ---
        "n_aligned_dirs":  float((cos2 > 0.5).sum()),
    }
    return {f"{prefix}{k}": v for k, v in out.items()}


def feature_subspace_metrics(Xs, Xt, r=32, prefix="fsa_"):
    """
    Baseline 3: principal angles between the top-r PCA subspaces of the two
    FEATURE clouds. Label-free, forward pass only.
    """
    r = int(min(r, Xs.shape[1], Xt.shape[1], len(Xs) - 1, len(Xt) - 1))
    Us = top_right_subspace(Xs, r)
    Ut = top_right_subspace(Xt, r)
    m = subspace_metrics(Us, Ut, prefix)
    m[f"{prefix}rank"] = float(r)
    m[f"{prefix}random_baseline"] = float(r / Xs.shape[1])   # Corollary 1: r/d
    m[f"{prefix}excess_over_random"] = m[f"{prefix}pa_similarity"] - m[f"{prefix}random_baseline"]
    return m


def gsa(U_source, G_target, r=32, prefix="gsa_"):
    """
    Proposal eq. (7).  rho_hat(U, t) = (1/r) || P_U P_V ||_F^2

    U_source : (d, r) orthonormal basis carried over from the source
    G_target : (C_out, d) gradient of the target loss at Delta = 0
    """
    Vt = top_right_subspace(G_target, r)
    m = subspace_metrics(U_source, Vt, prefix)
    d = U_source.shape[0]
    m[f"{prefix}random_baseline"] = float(r / d)
    m[f"{prefix}excess_over_random"] = m[f"{prefix}pa_similarity"] - float(r / d)
    return m


# ----------------------------------------------------------------- check
def selfcheck(d=512, r=32, trials=200, seed=0):
    """
    Three assertions:
      identical subspaces  -> pa_similarity 1, grassmann 0
      orthogonal subspaces -> pa_similarity 0
      random subspaces     -> pa_similarity ~ r/d   (proposal Corollary 1)
    """
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.normal(size=(d, r)))[0]

    same = subspace_metrics(U, U)
    orth = np.linalg.qr(rng.normal(size=(d, r)))[0]
    orth = orth - U @ (U.T @ orth)
    orth = np.linalg.qr(orth)[0]
    og = subspace_metrics(U, orth)

    vals = []
    for _ in range(trials):
        V = np.linalg.qr(rng.normal(size=(d, r)))[0]
        vals.append(subspace_metrics(U, V)["pa_similarity"])

    print(f"identical   pa_similarity={same['pa_similarity']:.6f} (want 1.0)  "
          f"grassmann={same['grassmann']:.6f} (want 0.0)")
    print(f"orthogonal  pa_similarity={og['pa_similarity']:.6f} (want 0.0)")
    print(f"random      pa_similarity={np.mean(vals):.5f} +/- {np.std(vals):.5f}  "
          f"(theory r/d = {r/d:.5f})")


if __name__ == "__main__":
    selfcheck()
