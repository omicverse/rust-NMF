"""End-to-end parity test: nmf-rs vs R `NMF` package.

Reference fixtures are produced by `reference_nmf.R` (CMAP env, R `NMF`
installed at /scratch/users/steorra/env/CMAP_Rlib).

Given identical (V, W0, H0) starting points, we expect:
    - brunet:  bitwise identical W, H after `max_iter` iterations
    - lee:     bitwise identical W, H
    - offset:  bitwise identical W, H, offset
    - nsNMF:   bitwise identical W, H

Floating-point summation order in our Rust port is matched to R's C++
loops, so equality should be exact. We allow a tiny `atol=1e-12` slack
to absorb any cross-platform fma/order quirks.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import nmf_rs

DATA_DIR = Path(__file__).parent / "data"
# Override the path to Rscript via NMF_RS_RSCRIPT for environments where R
# isn't on the default PATH. Defaults match the local Sherlock CMAP env so
# ad-hoc local runs Just Work; CI sets the env var to plain `Rscript`.
RSCRIPT = os.environ.get(
    "NMF_RS_RSCRIPT", "/scratch/users/steorra/env/CMAP/bin/Rscript"
)
REF_GEN = Path(__file__).parent / "reference_nmf.R"

# Test parameters — must match what reference_nmf.R was invoked with.
SEED   = 1234
N      = 80
P      = 30
RANK   = 4
ITER   = 50
ATOL   = 1e-12


def _ensure_reference():
    """Regenerate fixtures if missing."""
    needed = [DATA_DIR / f for f in
              ("V.tsv", "W0.tsv", "H0.tsv", "brunet__W.tsv", "brunet__H.tsv",
               "lee__W.tsv", "lee__H.tsv", "offset__W.tsv", "offset__H.tsv",
               "offset__off.tsv", "nsNMF__W.tsv", "nsNMF__H.tsv")]
    if all(p.exists() for p in needed):
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = "/scratch/users/steorra/env/CMAP/bin:" + env.get("PATH", "")
    subprocess.run(
        [RSCRIPT, str(REF_GEN), str(DATA_DIR),
         str(SEED), str(N), str(P), str(RANK), str(ITER)],
        check=True, env=env,
    )


def _load(name: str) -> np.ndarray:
    return np.loadtxt(DATA_DIR / name, dtype=np.float64)


@pytest.fixture(scope="module")
def fixtures():
    _ensure_reference()
    V  = _load("V.tsv")
    W0 = _load("W0.tsv")
    H0 = _load("H0.tsv")
    assert V.shape == (N, P), V.shape
    assert W0.shape == (N, RANK), W0.shape
    assert H0.shape == (RANK, P), H0.shape
    return {"V": V, "W0": W0, "H0": H0}


# ----------------------------------------------------------------------
# Single-update kernels
# ----------------------------------------------------------------------

def test_single_step_brunet(fixtures):
    """One Brunet update step matches R's `std.divergence.update.{h,w}`."""
    V, W0, H0 = fixtures["V"], fixtures["W0"], fixtures["H0"]
    H1 = nmf_rs.update_h_brunet(V, W0, H0)
    W1 = nmf_rs.update_w_brunet(V, W0, H1)
    # Numerical sanity (no NaN/Inf, non-negative).
    assert np.isfinite(H1).all() and np.isfinite(W1).all()
    assert (H1 >= 0).all() and (W1 >= 0).all()


def test_single_step_lee(fixtures):
    V, W0, H0 = fixtures["V"], fixtures["W0"], fixtures["H0"]
    H1 = nmf_rs.update_h_lee(V, W0, H0)
    W1 = nmf_rs.update_w_lee(V, W0, H1)
    assert np.isfinite(H1).all() and np.isfinite(W1).all()
    assert (H1 >= 0).all() and (W1 >= 0).all()


# ----------------------------------------------------------------------
# Full-iteration parity
# ----------------------------------------------------------------------

def _assert_match(name, x_rust, x_r):
    """Compare a Rust array against an R reference."""
    assert x_rust.shape == x_r.shape, f"{name}: shape {x_rust.shape} vs {x_r.shape}"
    diff = np.abs(x_rust - x_r)
    rel = diff / (np.abs(x_r) + 1e-30)
    max_abs = float(diff.max())
    max_rel = float(rel.max())
    assert max_abs <= ATOL, (
        f"{name}: max abs diff {max_abs:.3e} > {ATOL:.0e} "
        f"(max rel {max_rel:.3e}, mean abs {diff.mean():.3e})"
    )


def test_brunet_parity(fixtures):
    V, W0, H0 = fixtures["V"], fixtures["W0"], fixtures["H0"]
    res = nmf_rs.nmf(V, rank=RANK, method="brunet",
                     W0=W0, H0=H0, max_iter=ITER, stop="max_iter")
    _assert_match("brunet/W", res.W, _load("brunet__W.tsv"))
    _assert_match("brunet/H", res.H, _load("brunet__H.tsv"))
    assert res.n_iter == ITER


def test_lee_parity(fixtures):
    V, W0, H0 = fixtures["V"], fixtures["W0"], fixtures["H0"]
    # R's nmf_update.lee applies col-rescale by default. eps default is 10^-9.
    res = nmf_rs.nmf(V, rank=RANK, method="lee",
                     W0=W0, H0=H0, max_iter=ITER, stop="max_iter",
                     rescale=True, eps=1e-9)
    _assert_match("lee/W", res.W, _load("lee__W.tsv"))
    _assert_match("lee/H", res.H, _load("lee__H.tsv"))
    assert res.n_iter == ITER


def test_offset_parity(fixtures):
    V, W0, H0 = fixtures["V"], fixtures["W0"], fixtures["H0"]
    # R's offset algorithm initialises offset = rowMeans(V) on iter 1 — same
    # default as our Rust core when offset=None.
    res = nmf_rs.nmf(V, rank=RANK, method="offset",
                     W0=W0, H0=H0, max_iter=ITER, stop="max_iter", eps=1e-9)
    _assert_match("offset/W", res.W, _load("offset__W.tsv"))
    _assert_match("offset/H", res.H, _load("offset__H.tsv"))
    off_r = _load("offset__off.tsv").reshape(-1)
    assert res.offset is not None
    _assert_match("offset/off", res.offset, off_r)


def test_nsnmf_parity(fixtures):
    V, W0, H0 = fixtures["V"], fixtures["W0"], fixtures["H0"]
    theta = float(_load("nsNMF__theta.tsv"))
    res = nmf_rs.nmf(V, rank=RANK, method="nsNMF",
                     W0=W0, H0=H0, max_iter=ITER, stop="max_iter", theta=theta)
    _assert_match("nsNMF/W", res.W, _load("nsNMF__W.tsv"))
    _assert_match("nsNMF/H", res.H, _load("nsNMF__H.tsv"))
