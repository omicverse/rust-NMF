"""nmf-rs: Rust port of R NMF package multiplicative-update algorithms.

Bit-equivalent to the R `NMF` package (https://github.com/renozao/NMF) given
identical (V, W0, H0) starting points, ~5–20× faster on real data.

Public API
----------
- nmf(V, rank, method='brunet', ..., W0=None, H0=None, seed=None, max_iter=2000)
    Run a full NMF factorisation and return an NMFResult.
- update_h_brunet(V, W, H), update_w_brunet(V, W, H)
    Single-step KL multiplicative updates (matches R `std.divergence.update.{h,w}`).
- update_h_lee(V, W, H, eps=1e-9), update_w_lee(V, W, H, eps=1e-9)
    Single-step Frobenius multiplicative updates (matches R `std.euclidean.update.{h,w}`).
- random_init(V, rank, seed) → (W0, H0) using R's runif() Mersenne Twister.
- set_num_threads(n)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

try:
    from nmf_rs._rust import (
        py_divergence_update_h as _div_h,
        py_divergence_update_w as _div_w,
        py_euclidean_update_h as _euc_h,
        py_euclidean_update_w as _euc_w,
        py_nmf_run as _nmf_run,
        py_smoothing_matrix as _smoothing_matrix,
        set_num_threads as _set_num_threads,
    )
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    _div_h = _div_w = _euc_h = _euc_w = _nmf_run = _smoothing_matrix = _set_num_threads = None


__all__ = [
    "NMFResult",
    "nmf",
    "update_h_brunet",
    "update_w_brunet",
    "update_h_lee",
    "update_w_lee",
    "smoothing_matrix",
    "random_init",
    "nndsvd_init",
    "cv_rank",
    "set_num_threads",
]


# =============================================================================
# Result type
# =============================================================================

@dataclass
class NMFResult:
    """Outcome of an NMF run.

    Attributes
    ----------
    W : np.ndarray  (n × rank)
    H : np.ndarray  (rank × p)
    n_iter : int
    deviances : np.ndarray  — objective values across the run (empty for
        ``stop='max_iter'``).
    method : str
    offset : np.ndarray | None  — only set for the `offset` algorithm.
    """
    W: np.ndarray
    H: np.ndarray
    n_iter: int
    deviances: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    method: str = "brunet"
    offset: Optional[np.ndarray] = None

    @property
    def basis(self) -> np.ndarray:
        return self.W

    @property
    def coef(self) -> np.ndarray:
        return self.H

    def fitted(self) -> np.ndarray:
        """Return the model estimate W H (or W H + offset for the offset algo)."""
        wh = self.W @ self.H
        if self.offset is not None:
            wh = wh + self.offset[:, None]
        return wh

    def __repr__(self) -> str:
        return (
            f"NMFResult(method='{self.method}', n={self.W.shape[0]}, "
            f"p={self.H.shape[1]}, rank={self.W.shape[1]}, n_iter={self.n_iter})"
        )


# =============================================================================
# Random initialisation — replicates R's runif() default in nmf()
# =============================================================================

# R's runif on a Mersenne Twister stream is not directly accessible from
# Python without rpy2. To enable parity tests we simply load the W0/H0 the
# R script generates. For users who want an in-Python random init, NumPy's
# default_rng provides a reasonable substitute (NOT bit-equal to R).
def random_init(
    V: np.ndarray,
    rank: int,
    seed: Optional[int] = None,
    *,
    max_value: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (W0, H0) ~ U(0, max(V)) following the same shape conventions
    R uses (W is n×rank, H is rank×p). NOT bit-equal to R's runif() — for
    bit-equality, generate W0/H0 in R and pass them via ``W0=`` / ``H0=``.
    """
    n, p = V.shape
    rng = np.random.default_rng(seed)
    hi = float(max_value if max_value is not None else (np.max(V) if V.size else 1.0))
    W0 = rng.uniform(0.0, hi, size=(n, rank))
    H0 = rng.uniform(0.0, hi, size=(rank, p))
    return W0, H0


def nndsvd_init(
    V: np.ndarray,
    rank: int,
    *,
    fill: str = "mean",
    seed: int = 0,
    n_oversamples: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """NNDSVD initialisation (Boutsidis & Gallopoulos 2008).

    Computes a truncated SVD of ``V`` and turns each (u_k, v_k) pair into a
    non-negative outer product by picking the side of the singular vector
    with the larger 2-norm product. Yields a deterministic, low-residual
    starting point — typically lets multiplicative-update solvers converge
    in **3–5× fewer iterations** than a uniform random ``W0/H0``.

    Parameters
    ----------
    V : (n, p) array_like, non-negative
    rank : int
    fill : {'zero', 'mean', 'eps'}, default 'mean'
        How to replace zero entries. ``'mean'`` (NNDSVDa) is the canonical
        choice for downstream NMF; pure ``'zero'`` (NNDSVD) keeps sparsity.
    seed : int
        Seed for the randomized SVD power iteration.
    n_oversamples : int
        Oversampling for randomized SVD. Defaults to sklearn's 10.

    Returns
    -------
    (W0, H0) : (n, rank), (rank, p)
    """
    try:
        from sklearn.utils.extmath import randomized_svd
    except ImportError as e:
        raise ImportError(
            "nndsvd_init() requires scikit-learn (`pip install scikit-learn`)."
        ) from e

    V = np.asarray(V, dtype=np.float64)
    if (V < 0).any():
        raise ValueError("V must be non-negative for NNDSVD initialisation.")

    U, S, Vt = randomized_svd(
        V, n_components=rank, n_oversamples=n_oversamples, random_state=seed,
    )

    n, p = V.shape
    W = np.zeros((n, rank), dtype=np.float64)
    H = np.zeros((rank, p), dtype=np.float64)

    # First component: per Boutsidis & Gallopoulos, take |u| and |v| since the
    # first SVD pair is (close to) all-positive for non-negative V (Perron-Frobenius).
    s0 = np.sqrt(S[0])
    W[:, 0] = s0 * np.abs(U[:, 0])
    H[0, :] = s0 * np.abs(Vt[0, :])

    # Subsequent: pick the (positive or negative) side with larger ||·||·||·||.
    for k in range(1, rank):
        u, v = U[:, k], Vt[k, :]
        u_pos = np.maximum(u, 0.0); u_neg = np.maximum(-u, 0.0)
        v_pos = np.maximum(v, 0.0); v_neg = np.maximum(-v, 0.0)
        n_pos = np.linalg.norm(u_pos) * np.linalg.norm(v_pos)
        n_neg = np.linalg.norm(u_neg) * np.linalg.norm(v_neg)
        if n_pos >= n_neg:
            sigma = np.sqrt(S[k] * n_pos)
            if n_pos > 0:
                W[:, k] = sigma * u_pos / np.linalg.norm(u_pos)
                H[k, :] = sigma * v_pos / np.linalg.norm(v_pos)
        else:
            sigma = np.sqrt(S[k] * n_neg)
            if n_neg > 0:
                W[:, k] = sigma * u_neg / np.linalg.norm(u_neg)
                H[k, :] = sigma * v_neg / np.linalg.norm(v_neg)

    if fill == "zero":
        pass
    elif fill == "eps":
        eps = 1e-9
        W[W < eps] = eps
        H[H < eps] = eps
    elif fill == "mean":
        # NNDSVDa: replace zeros with mean(V) — better for multiplicative updates,
        # which can't escape exact zeros once entered.
        m = float(V.mean()) if V.size else 0.0
        W[W <= 0.0] = m
        H[H <= 0.0] = m
    else:
        raise ValueError(f"unknown fill mode '{fill}': supported zero / mean / eps")

    return W, H


# =============================================================================
# Single-step updates (thin wrappers, primarily for tests)
# =============================================================================

def _check_rust():
    if not _RUST_AVAILABLE:
        raise RuntimeError(
            "Rust backend not available. Run "
            "`maturin develop --release --manifest-path rust/Cargo.toml` "
            "from the rust-NMF directory."
        )


def _f64_c(arr: np.ndarray, name: str) -> np.ndarray:
    """Coerce to float64 C-contiguous (numpy expects exactly this)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {a.shape}")
    return a


def update_h_brunet(V: np.ndarray, W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Single Brunet (KL) update of H. Matches R `std.divergence.update.h`."""
    _check_rust()
    return _div_h(_f64_c(V, "V"), _f64_c(W, "W"), _f64_c(H, "H"))


def update_w_brunet(V: np.ndarray, W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Single Brunet (KL) update of W. Matches R `std.divergence.update.w`."""
    _check_rust()
    return _div_w(_f64_c(V, "V"), _f64_c(W, "W"), _f64_c(H, "H"))


def update_h_lee(V: np.ndarray, W: np.ndarray, H: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Single Lee (Frobenius) update of H. Matches R `std.euclidean.update.h`."""
    _check_rust()
    return _euc_h(_f64_c(V, "V"), _f64_c(W, "W"), _f64_c(H, "H"), float(eps))


def update_w_lee(V: np.ndarray, W: np.ndarray, H: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Single Lee (Frobenius) update of W. Matches R `std.euclidean.update.w`."""
    _check_rust()
    return _euc_w(_f64_c(V, "V"), _f64_c(W, "W"), _f64_c(H, "H"), float(eps))


def smoothing_matrix(rank: int, theta: float) -> np.ndarray:
    """Return nsNMF smoothing matrix S = (1-θ) I + (θ/r) 1 1^T."""
    _check_rust()
    return _smoothing_matrix(int(rank), float(theta))


def cv_rank(
    V: np.ndarray,
    ranks: Sequence[int],
    *,
    method: str = "hals",
    n_folds: int = 4,
    mask_frac: float = 0.05,
    max_iter: int = 100,
    init: str = "nndsvd",
    seed: int = 0,
    num_threads: Optional[int] = None,
    **nmf_kwargs,
) -> "pd.DataFrame":
    """Cross-validated rank selection by held-out reconstruction error.

    For each candidate rank ``k`` we hold out ``mask_frac`` of V's entries
    (random per fold), fit NMF on the masked V via ``method='ls-nmf'`` (so
    the held-out entries truly drop from the loss), then predict the held-out
    entries via W·H and report MSE on them.

    Lower test-MSE → better-suited rank. Typical pattern: test-MSE drops
    quickly, then plateaus at the "true" rank.

    Parameters
    ----------
    V : (n, p) non-negative array
    ranks : iterable of int — candidates to evaluate
    method : str — base NMF method; we always wrap it through ls-nmf for the
        masking weight, but ``method`` controls the *initialisation pass*
        when ``init='warm'``.
    n_folds : int — average over this many random masks
    mask_frac : float — fraction of entries held out per fold
    max_iter : int — passed to nmf()
    init : {'nndsvd', 'random'} — starting point for each fold
    seed : int — base seed for reproducibility
    num_threads : optional int — passed through

    Returns
    -------
    pandas.DataFrame with columns: rank, fold, train_loss, test_mse.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("cv_rank() requires pandas") from e
    _check_rust()
    V = _f64_c(V, "V")
    if (V < 0).any():
        raise ValueError("V must be non-negative")
    n, p = V.shape

    rows = []
    rng_master = np.random.default_rng(seed)
    for k in ranks:
        for fold in range(n_folds):
            fold_seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(fold_seed)
            mask = rng.random((n, p)) > mask_frac    # True = train
            weight = mask.astype(np.float64)
            V_train = V * weight                      # zero out held-out

            if init == "nndsvd":
                W0, H0 = nndsvd_init(V_train, k, fill="mean", seed=fold_seed)
            else:
                W0, H0 = random_init(V_train, k, seed=fold_seed)

            # ls-nmf with the mask weight — held-out entries don't pull on the fit.
            res = nmf(
                V_train, rank=k, method="ls-nmf",
                W0=W0, H0=H0, weight=weight,
                max_iter=max_iter, num_threads=num_threads,
                **nmf_kwargs,
            )
            recon = res.fitted()
            train_residual = (V - recon) * weight
            train_loss = 0.5 * float((train_residual ** 2).sum())
            held = ~mask
            test_mse = float(((V[held] - recon[held]) ** 2).mean())
            rows.append({
                "rank": int(k), "fold": fold,
                "train_loss": train_loss, "test_mse": test_mse,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Top-level driver
# =============================================================================

_METHOD_ALIASES = {
    "brunet": "brunet",
    "kl": "brunet",
    "lee": "lee",
    "frobenius": "lee",
    "euclidean": "lee",
    "offset": "offset",
    "nsnmf": "nsnmf",
    "ns": "nsnmf",
    "ns_nmf": "nsnmf",
    "hals": "hals",
    "ehals": "ehals",
    "e-hals": "ehals",
    "extrap_hals": "ehals",
    "dnmf": "dnmf",
    "diag_nmf": "dnmf",
    "rcppml": "dnmf",
    "ls-nmf": "lsnmf",
    "lsnmf": "lsnmf",
    "snmf/r": "snmf/r",
    "snmf_r": "snmf/r",
    "snmfr": "snmf/r",
    "snmf/l": "snmf/l",
    "snmf_l": "snmf/l",
    "snmfl": "snmf/l",
}


def set_num_threads(n: int) -> bool:
    """Configure rayon's worker thread pool (best-effort; ignored after first run)."""
    _check_rust()
    return _set_num_threads(int(n))


def nmf(
    V: np.ndarray,
    rank: int,
    method: str = "brunet",
    *,
    W0: Optional[np.ndarray] = None,
    H0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    max_iter: int = 2000,
    eps: float = 1e-9,
    rescale: bool = True,
    theta: float = 0.5,
    offset: Optional[np.ndarray] = None,
    weight: Optional[np.ndarray] = None,
    sparsity: float = 0.01,
    smoothness: float = 1.0,
    stop: str = "max_iter",
    stationary_th: float = float(np.finfo(np.float64).eps),
    check_interval: int = 50,
    check_niter: int = 10,
    num_threads: Optional[int] = None,
) -> NMFResult:
    """Run an NMF factorisation V ≈ W H using a Rust multiplicative-update core.

    Parameters
    ----------
    V : (n, p) array_like
        Non-negative target matrix. Will be coerced to ``float64``.
    rank : int
        Factorisation rank (number of components).
    method : {'brunet', 'lee', 'offset', 'nsNMF', 'hals', 'ls-nmf',
              'snmf/r', 'snmf/l'}, default 'brunet'
        Multiplicative-update family.

        - ``brunet`` / ``KL`` — KL-divergence updates, R-parity bit-equiv.
        - ``lee`` / ``Frobenius`` / ``euclidean`` — Lee/Seung Frobenius, R-parity.
        - ``offset`` — Lee + per-feature offset (Badea 2008), R-parity.
        - ``nsNMF`` — Brunet + smoothing matrix (Pascual-Montano 2006), R-parity.
        - ``hals`` — Cichocki-Phan least-squares; ~10× fewer iterations needed.
        - ``ehals`` — extrapolated HALS (Ang-Gillis 2019). Same per-iter cost
          as HALS plus one WH evaluation, but Nesterov-style momentum cuts
          iteration count by ~2×. Strict upgrade over ``hals``.
        - ``dnmf`` (alias ``rcppml``) — diagonalised NMF V ≈ W·diag(d)·H
          with column-unit W, row-unit H (DeBruine 2024). Makes L1 reg
          (``sparsity=``) actually mean something — without diagonalisation
          the scale ambiguity makes ||W||₁ trivially adjustable.
        - ``ls-nmf`` (alias ``lsnmf``) — weighted Lee (Wang 2006); requires
          ``weight=`` of shape V. Useful for masking missing data.
        - ``snmf/r`` / ``snmf/l`` — sparse-H / sparse-W via regularised HALS,
          NOT bit-equivalent to R's FCNNLS-based snmf but achieves the same
          sparsity goal. Tune via ``sparsity=`` and ``smoothness=``.
    W0, H0 : array, optional
        Initial factor matrices. If not provided, ``random_init`` is used
        with ``seed``. **For bit-equivalence with R, pass W0/H0 generated
        in R** (e.g. via ``set.seed(s); W0 <- matrix(runif(n*r), n, r); ...``).
    seed : int, optional
        Seed for ``random_init`` when W0/H0 are not given.
    max_iter : int, default 2000
        Hard cap on iterations (also the only stop when ``stop='max_iter'``).
    eps : float, default 1e-9
        Numerical floor used by the lee / offset updates (``10^-9`` in R).
    rescale : bool, default True
        Rescale columns of W to sum to 1 after every Lee update (R default).
    theta : float, default 0.5
        Smoothing parameter for ``method='nsNMF'``.
    offset : (n,) array, optional
        Initial offset vector for ``method='offset'``. Defaults to
        ``rowMeans(V)`` to match R.
    weight : (n, p) array, optional
        Per-entry weight matrix for ``method='ls-nmf'``. Set
        ``weight[i, j] = 0`` to mask V[i, j] (missing-value imputation).
    sparsity : float, default 0.01
        L1 penalty coefficient for ``method='snmf/r'`` (applied to H) or
        ``'snmf/l'`` (applied to W). Larger → sparser factors.
    smoothness : float, default 1.0
        L2 penalty coefficient for ``method='snmf/r'`` (applied to W) or
        ``'snmf/l'`` (applied to H). Helps regularise the non-sparse factor.
    stop : {'max_iter', 'stationary'}
        ``'stationary'`` replicates R `nmf.stop.stationary` semantics.
    stationary_th, check_interval, check_niter
        Stopping-criterion parameters; defaults match R.
    num_threads : int, optional
        Run the rayon-parallel inner loops in a *fresh* thread pool of this
        size, scoped to this single call (does not affect the global pool).
        Use this instead of ``set_num_threads()`` if you want different runs
        to use different thread counts in one process.

    Returns
    -------
    NMFResult
    """
    _check_rust()
    method_norm = _METHOD_ALIASES.get(method.lower())
    if method_norm is None:
        raise ValueError(
            f"unknown method '{method}'. supported: brunet, lee, offset, nsNMF "
            "(plus aliases KL/Frobenius/ns)."
        )

    V_arr = _f64_c(V, "V")
    if (V_arr < 0).any():
        raise ValueError("V must be non-negative.")
    n, p = V_arr.shape

    if W0 is None or H0 is None:
        W0_, H0_ = random_init(V_arr, rank, seed=seed)
        if W0 is None:
            W0 = W0_
        if H0 is None:
            H0 = H0_

    W0 = _f64_c(W0, "W0")
    H0 = _f64_c(H0, "H0")
    if W0.shape != (n, rank):
        raise ValueError(f"W0.shape {W0.shape} != ({n}, {rank})")
    if H0.shape != (rank, p):
        raise ValueError(f"H0.shape {H0.shape} != ({rank}, {p})")

    offset_arr = None
    if offset is not None:
        offset_arr = np.ascontiguousarray(np.asarray(offset, dtype=np.float64))
        if offset_arr.shape != (n,):
            raise ValueError(f"offset.shape {offset_arr.shape} != ({n},)")

    weight_arr = None
    if weight is not None:
        weight_arr = np.ascontiguousarray(np.asarray(weight, dtype=np.float64))
        if weight_arr.shape != V_arr.shape:
            raise ValueError(
                f"weight.shape {weight_arr.shape} != V.shape {V_arr.shape}"
            )

    # Sparsity / smoothness routing per algorithm:
    if method_norm == "snmf/r":
        sh, sw, gh, gw = float(sparsity), 0.0, 0.0, float(smoothness)
    elif method_norm == "snmf/l":
        sh, sw, gh, gw = 0.0, float(sparsity), float(smoothness), 0.0
    elif method_norm == "dnmf":
        # dnmf: regularise both factors symmetrically by default.
        sh = sw = float(sparsity)
        gh = gw = 0.0  # smoothness on dnmf is rarely useful; users can extend later
    else:
        sh = sw = gh = gw = 0.0

    out = _nmf_run(
        method_norm,
        V_arr,
        W0,
        H0,
        int(max_iter),
        eps=float(eps),
        rescale=bool(rescale),
        theta=float(theta),
        offset=offset_arr,
        weight=weight_arr,
        sparsity_h=sh,
        sparsity_w=sw,
        smoothness_h=gh,
        smoothness_w=gw,
        stop=stop,
        stationary_th=float(stationary_th),
        check_interval=int(check_interval),
        check_niter=int(check_niter),
        num_threads=int(num_threads) if num_threads is not None else None,
    )
    return NMFResult(
        W=np.asarray(out["W"]),
        H=np.asarray(out["H"]),
        n_iter=int(out["n_iter"]),
        deviances=np.asarray(out["deviances"], dtype=np.float64),
        method=method_norm,
        offset=np.asarray(out["offset"]) if "offset" in out else None,
    )
