# Changelog

All notable changes to `nmf-rs` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); semantics follow
[SemVer](https://semver.org/).

## [Unreleased]

## [0.1.0] — initial release

### Added
- Rust port of R `NMF` package multiplicative-update kernels:
  - `brunet` (KL divergence, default in R)
  - `lee` (Frobenius / Euclidean)
  - `offset` (Lee + per-feature offset, Badea 2008)
  - `nsNMF` (Brunet + smoothing matrix, Pascual-Montano 2006)
- `hals` algorithm (Cichocki-Phan least-squares; aliased as `lsnmf`).
- NNDSVD initialisation `nmf_rs.nndsvd_init(V, rank, fill='mean')`.
- Per-call rayon thread-pool argument `num_threads=N` —
  multiple `nmf()` calls in one process can use different thread counts.
- Single-step kernels exposed as `update_h_brunet`, `update_w_brunet`,
  `update_h_lee`, `update_w_lee`, `smoothing_matrix`.
- `stop='stationary'` replicates R's `nmf.stop.stationary` semantics
  (returns the deviance trace in `NMFResult.deviances`).

### Performance
On omicverse `pbmc8k` (HVG=2000, cells=7750, rank=10, 100 iters):
- `lee`: R 125 s → Rust **1.65 s @ 16t (76×)**, single-thread 6.5×.
- `brunet`: R 79 s → Rust **4.30 s @ 16t (18×)**, single-thread 1.4×.
- `hals` + NNDSVD, 25 iters @ 16t: **0.58 s ≈ 218× R Lee** at the same
  reconstruction-loss plateau.

### Tested
- 6 / 6 parity tests pass: `max|ΔW|` and `max|ΔH|` < 1e-12 vs R `NMF`
  for `brunet`, `lee`, `offset`, `nsNMF`.
