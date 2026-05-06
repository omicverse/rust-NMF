# rust-NMF

Rust port of the [R `NMF` package](https://github.com/renozao/NMF) — bit-equivalent
multiplicative-update algorithms (`brunet`, `lee`, `offset`, `nsNMF`), parallelised
with rayon. Drop-in replacement for the inner update loops of `NMF::nmf()`.

## Features

- **Bit-equivalent**: given identical `(V, W0, H0)` (and `offset` for the offset
  algorithm), the iterates are bitwise identical to R's C++ kernels.
- **Parallel**: rayon-parallel inner kernels with per-call thread pools
  (`nmf_rs.nmf(..., num_threads=8)`), so the same process can run different
  configurations without re-initialising a global pool.
- **Faster**: on omicverse `pbmc8k` (HVG=2000, cells=7750, rank=10):
  - `lee`:    R 125 s, 100 iters  vs Rust  **1.65 s @ 16t = 76× faster**  (6.5× single-thread).
  - `brunet`: R  79 s, 100 iters  vs Rust  **4.30 s @ 16t = 18× faster**  (1.4× single-thread).
  - `hals` + NNDSVD init, 25 iters @ 16t: **0.58 s = ~215× faster** at the same loss plateau.
  See `examples/benchmark_pbmc8k.ipynb` for the full thread sweep + factor QC.

## Algorithmic shortcuts (when bit-equivalence with R is not required)

| config                            | iters | 1t      | 16t     | vs R Lee |
|-----------------------------------|-------|---------|---------|----------|
| `lee` + random init               | 100   | 19.4 s  | 1.88 s  | 67×      |
| `lee` + NNDSVD init               | 50    | 9.79 s  | 0.94 s  | 133×     |
| `hals` + random init              | 50    | 9.78 s  | 1.04 s  | 121×     |
| **`hals` + NNDSVD init**          | **25** | **4.85 s** | **0.58 s** | **218×** |
| `hals` + NNDSVD (loose tolerance) | 10    | 2.03 s  | 0.45 s  | **280×** |

- **HALS** (`method='hals'` / `'lsnmf'`): Cichocki-Phan one-row-at-a-time
  least-squares updates. Same per-iteration cost as `lee` but converges to
  the same loss plateau in **5–10× fewer iterations**. Reconstruction error
  is actually *lower* than 100-iter Lee (807k vs 815k on pbmc8k); factor
  identity may differ on 1–2 of the 10 components (typical local-optimum
  spread for non-bit-equivalent solvers).
- **NNDSVD initialisation** (`nmf_rs.nndsvd_init(V, rank)`): truncated-SVD
  warm start. Initial loss is ~**5 orders of magnitude** lower than
  `runif(0, max(V))` on real scRNA-seq data — most of the early iterations
  the random init wastes are spent shrinking ‖WH‖ down to ‖V‖.

Combined: `nmf(V, rank, method='hals', W0=W0_nn, H0=H0_nn, max_iter=25)`.

See `examples/hals_nndsvd_demo.ipynb` for the convergence trace + factor-quality
check.
- **Pythonic API**: `nmf_rs.nmf(V, rank, method=...)` returns an `NMFResult` with
  `W`, `H`, `n_iter`, `deviances`, `offset`.

## Install (development)

```bash
module load rust/1.90.0  # Sherlock
maturin develop --release --manifest-path rust/Cargo.toml
```

## Quickstart

```python
import numpy as np
from nmf_rs import nmf

V = np.abs(np.random.default_rng(0).standard_normal((300, 50)))
res = nmf(V, rank=5, method="brunet", max_iter=200, seed=42)
print(res)            # NMFResult(method='brunet', n=300, p=50, rank=5, n_iter=200)
recon = res.fitted()  # W @ H, shape (300, 50)
```

## Parity vs R

```python
# 1) Generate W0, H0 in R with set.seed(s); store as TSV.
# 2) Load same V, W0, H0 in Python:
import numpy as np
from nmf_rs import nmf
V = np.loadtxt("V.tsv");  W0 = np.loadtxt("W0.tsv");  H0 = np.loadtxt("H0.tsv")
res = nmf(V, rank=W0.shape[1], W0=W0, H0=H0, method="brunet", max_iter=200)
W_R = np.loadtxt("W_R.tsv");  H_R = np.loadtxt("H_R.tsv")
np.testing.assert_array_equal(res.W, W_R)  # should pass exactly
```

See `tests/test_parity.py` for the end-to-end pytest suite.

## Algorithms

| `method`   | R equivalent | Update rule                                   |
|------------|--------------|-----------------------------------------------|
| `brunet`   | `nmf_update.brunet` | KL divergence, every-10-iter eps clamp |
| `lee`      | `nmf_update.lee`    | Frobenius/Euclidean, optional col rescale |
| `offset`   | `nmf_update.offset` | Lee + offset vector (Badea 2008) |
| `nsNMF`    | `nmf_update.ns`     | Brunet with smoothing matrix (theta) |

`lsNMF`, `snmf/r`, `snmf/l` are not yet ported.

## License

GPL ≥ 2 (inherited from R `NMF`).
