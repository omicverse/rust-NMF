# rust-NMF

A modern Rust port of [R's `NMF` package](https://github.com/renozao/NMF) plus
2019-2024 SOTA algorithms (HALS, E-HALS, RcppML-style diagonalised NMF). Six
algorithms are bit-equivalent to R within f64 round-off; four are modern
single-cell-tuned methods. Parallelised with rayon + matrixmultiply, on
omicverse `pbmc8k` it's **76-280× faster than R** depending on configuration.

## Quick decision table

| Use case | Recommended | Why |
|---|---|---|
| **Single-cell exploratory (default)** | `lee` + NNDSVD, 25 iter | **ARI 0.89** vs cell types — best biology *and* bit-eq R |
| Pure speed with strong biology | `dnmf` + NNDSVD, 25 iter | ARI 0.85, ~26% faster than `lee+NNDSVD` |
| Reproduce existing R `NMF::nmf()` analysis | `lee`, `brunet`, `offset`, `nsNMF`, `snmf/r`, `snmf/l` | bit-eq R within 1e-12 |
| Largest speed (loss only matters) | `hals` + NNDSVD, 25 iter | ~280× R, ~3× sklearn cd; ARI lower (0.55) |
| Synthetic / ground-truth recovery | `lee` or `brunet`, 100+ iter | lowest Amari error |
| Sparse factors with bit-eq R | `snmf/r` or `snmf/l` | FCNNLS port + L1² regularisation |
| Missing-value imputation | `ls-nmf` with `weight=mask` | weight=0 drops entries from loss |
| Large atlas (>500k cells) | `lee` + NNDSVD or `dnmf` + NNDSVD | NNDSVD warm-start dominant at scale |

**Key empirical finding** (from `examples/benchmark_scaling.ipynb` init-pairing section,
real pbmc8k 3000 cells × 1500 HVG, rank 8): **NNDSVD init is the magic, not the algorithm**.
With NNDSVD seeding, *every* algorithm reaches its biological-signal ceiling in ~25
iterations. Random init takes 100 iters and still ends ~30% worse on ARI. Pair NNDSVD
with `lee`/`brunet`/`dnmf` and you get the best of both worlds (R-equivalence + best
biology, or speed + best biology).

## Four one-liner recipes

```python
import numpy as np
from nmf_rs import nmf, nndsvd_init

# 1. EXPLORATORY single-cell, best ARI vs cell types (0.89), still bit-eq R
W0, H0 = nndsvd_init(V, rank=10, fill='mean')
res = nmf(V, rank=10, method='lee',             # 'brunet' also reaches ARI 0.86
          W0=W0, H0=H0, max_iter=25, num_threads=16)

# 2. SPEED + biology, modern algorithm (ARI 0.85, 26% faster than recipe 1)
W0, H0 = nndsvd_init(V, rank=10, fill='mean')
res = nmf(V, rank=10, method='dnmf',
          W0=W0, H0=H0, max_iter=25, num_threads=16)

# 3. REPRODUCE R analysis exactly — bit-eq, max|ΔW| < 1e-12
res = nmf(V, rank=10, method='brunet',           # or 'lee', 'offset', 'nsNMF', 'snmf/r', 'snmf/l'
          W0=W0_from_R, H0=H0_from_R, max_iter=200, num_threads=16)

# 4. PUREST SPEED at atlas scale (>500k cells), ARI tradeoff
W0, H0 = nndsvd_init(V, rank=20, fill='mean')
res = nmf(V, rank=20, method='hals',
          W0=W0, H0=H0, max_iter=25, num_threads=16)
```

## Algorithm catalogue

| `method` | Origin | Year | Update rule | Bit-eq R | Best for |
|---|---|---|---|---|---|
| `brunet` (alias `KL`) | Brunet et al. PNAS | 2004 | KL divergence multiplicative | ✅ 1e-12 | R reproduction; count-like data |
| `lee` (alias `Frobenius`) | Lee & Seung | 2001 | Frobenius multiplicative | ✅ 1e-12 | R reproduction; ground-truth recovery |
| `offset` | Badea | 2008 | Lee + per-feature baseline | ✅ 1e-12 | per-gene background subtraction |
| `nsNMF` | Pascual-Montano | 2006 | Brunet + smoothing S | ✅ 1e-12 | sparser, more interpretable factors |
| `ls-nmf` (alias `lsnmf`) | Wang, Kossenkov, Ochs | 2006 | Weighted Frobenius; needs `weight=` | ≈ (BLAS variance) | missing-value imputation |
| `snmf/r` | Kim & Park | 2007 | FCNNLS-based ANLS; sparse H | ✅ 1e-12 | sparse coefficients (cells) |
| `snmf/l` | Kim & Park | 2007 | FCNNLS-based ANLS; sparse W | ✅ 1e-12 | sparse basis (genes) |
| **`hals`** | Cichocki & Phan | 2009 | Block-coord least-squares | ❌ (custom) | speed |
| **`ehals`** | Andersen-Ang & Gillis | 2019 | HALS + Nesterov extrapolation | ❌ | speed + slight per-iter accuracy |
| **`dnmf`** (alias `rcppml`) | DeBruine (RcppML) | 2024 | Diagonalised W·diag(d)·H + L1 | ❌ | **single-cell biological signal** |

The six "✅ 1e-12" rows are bitwise-identical to R `NMF` within f64 round-off
on every parity test (`pytest tests/`).

## Speed (omicverse pbmc8k, HVG=2000, cells=7750, rank=10)

| config | iters | 16t (s) | speed-up vs R `lee` (125 s) |
|---|---|---|---|
| R `lee` + R seed | 100 | 125 s | 1× (baseline) |
| `lee` + matched seed | 100 | **1.65 s** | **76×** ✅ bit-eq |
| `brunet` + matched seed | 100 | **4.30 s** | 18× ✅ bit-eq |
| `hals` + random | 100 | 1.61 s | 78× |
| **`dnmf` + NNDSVD** | **25** | ~0.5 s | ~250× |
| **`hals` + NNDSVD** | 25 | 0.58 s | 215× |
| `hals` + NNDSVD (loose) | 10 | 0.45 s | 280× |
| sklearn cd (BLAS, 17t) | 100 | 0.79 s | 158× |

See `examples/benchmark_pbmc8k.ipynb` and `examples/comparison_vs_cnmf.ipynb`.

## Objective accuracy (anchor-free, no algorithm bias)

The standard NMF evaluation metrics from
[Brunet 2004](https://www.pnas.org/doi/10.1073/pnas.0308531101) and
[Kim & Park 2007](https://academic.oup.com/bioinformatics/article/23/12/1495/225472)
on real pbmc8k (3000 cells × 1500 HVG, rank 8, K=5 random inits per algorithm):

### ARI vs cell types (real pbmc8k, 3k cells × 1.5k HVG, rank 8)

Pivoted by (algorithm × init) to expose the NNDSVD effect:

| algo | NNDSVD (25 iter, K=1) | random (K=5 runs) |
|---|---|---|
| **`lee`** | **🥇 0.889** | 0.568 |
| `brunet` | 0.864 | 0.621 |
| **`dnmf`** | **🥈 0.854** | 0.726 |
| `hals` | 0.547 | 0.428 |
| `ehals` | 0.486 | 0.539 |

### Wall-clock at the same configuration (16 threads)

| algo | NNDSVD@25 (s) | random×5 (mean s/run) |
|---|---|---|
| `lee` | 0.19 | 0.39 |
| `brunet` | 0.41 | 0.96 |
| `dnmf` | 0.14 | 0.24 |
| `hals` | 0.13 | 0.26 |
| `ehals` | 0.12 | 0.26 |

### Three takeaways

1. **NNDSVD init dominates the choice of algorithm.** Every algorithm gets
   higher ARI with NNDSVD than with random — typically 30-60% higher. So
   the question "which algorithm" is downstream of "use NNDSVD".

2. **`lee + NNDSVD` is the gold standard for biological signal**.
   On real single-cell data with `predicted_celltype` labels, `lee + NNDSVD@25`
   reaches ARI = 0.89 — the best across all (algorithm × init) combinations.
   And `lee` is bit-equivalent to R `NMF::nmf(method="lee")` so you can
   reproduce R analyses exactly.

3. **`dnmf + NNDSVD` is the fast alternative**. Same NNDSVD trick on the
   modern (DeBruine 2024) diagonalised algorithm: 0.14 s, ARI 0.85.
   Trades 0.04 ARI for 26% speed.

Caveat: on synthetic data without label structure, the same recipes
undertrain in 25 iters — for ground-truth recovery prefer `lee`/`brunet`
with 100+ iters (medium tier in `examples/benchmark_scaling.ipynb`).

## Tutorial roadmap

| Notebook | Audience | What it covers |
|---|---|---|
| [`tutorial_quickstart.ipynb`](examples/tutorial_quickstart.ipynb) | New users | API basics, all 4 R-bit-eq algorithms, stop criteria |
| [`benchmark_vs_R.ipynb`](examples/benchmark_vs_R.ipynb) | R `NMF` users | Drop-in parity check vs R, max\|ΔW\|<1e-12 |
| [`benchmark_pbmc8k.ipynb`](examples/benchmark_pbmc8k.ipynb) | Single-cell users | Real pbmc8k thread sweep, factor inspection |
| [`hals_nndsvd_demo.ipynb`](examples/hals_nndsvd_demo.ipynb) | Speed-focused users | HALS + NNDSVD convergence trace, factor quality |
| [`comparison_vs_cnmf.ipynb`](examples/comparison_vs_cnmf.ipynb) | cNMF users | sklearn / cNMF apples-to-apples, factor correlation |
| **[`benchmark_scaling.ipynb`](examples/benchmark_scaling.ipynb)** | **Anyone choosing an algorithm** | **3 scales × 7 algos × 6 objective metrics + init pairing** |
| [`tutorial_full.ipynb`](examples/tutorial_full.ipynb) | Comprehensive | R baseline → all algorithms → init strategies → threading → production recipe |

Suggested reading order for new users:
1. **Start** with `tutorial_quickstart.ipynb` for API.
2. Then `benchmark_scaling.ipynb` to **pick the right algorithm for your scale + use case**.
3. If reproducing an R analysis, `benchmark_vs_R.ipynb`.
4. If you've used cNMF before, `comparison_vs_cnmf.ipynb`.
5. For deep-dive: `tutorial_full.ipynb`.

## Install

```bash
# Sherlock or any host with rust toolchain:
module load rust/1.90.0
maturin develop --release --manifest-path rust/Cargo.toml

# Or via pip into an active venv (release wheels coming to PyPI as `nmf-rs`):
VIRTUAL_ENV=/path/to/venv pip install -e .
```

## Reproducing an R `NMF` analysis

```python
# 1. Generate W0, H0 in R with set.seed(s); export as TSV.
# 2. Load same V, W0, H0 in Python:
import numpy as np
from nmf_rs import nmf

V  = np.loadtxt("V.tsv")
W0 = np.loadtxt("W0.tsv")
H0 = np.loadtxt("H0.tsv")
res = nmf(V, rank=W0.shape[1], W0=W0, H0=H0, method="brunet", max_iter=200)

W_R = np.loadtxt("W_from_R.tsv")
H_R = np.loadtxt("H_from_R.tsv")
np.testing.assert_allclose(res.W, W_R, atol=1e-12)   # passes exactly
np.testing.assert_allclose(res.H, H_R, atol=1e-12)
```

See `tests/test_parity.py` and `tests/test_snmf_parity.py` for the
auto-regenerated reference fixtures + R-parity asserts.

## License

GPL ≥ 2 (inherited from R `NMF`).
