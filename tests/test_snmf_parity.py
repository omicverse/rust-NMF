"""R-parity for snmf/R and snmf/L (Kim-Park, FCNNLS-based ANLS).

We mirror R `NMF::nmf_snmf` exactly: same V, same initial W0/H0, same
β/η/maxIter, FCNNLS as the inner NNLS solver. Output should be bitwise
identical to R within f64 round-off (~1e-12).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import nmf_rs

DATA_DIR = Path(__file__).parent / "data_snmf"
RSCRIPT = os.environ.get(
    "NMF_RS_RSCRIPT", "/scratch/users/steorra/env/CMAP/bin/Rscript"
)
REF_GEN = Path(__file__).parent / "reference_snmf.R"

SEED = 1234
N, P, RANK = 80, 30, 4
ITERS = 20
ETA = -1.0
BETA = 0.01
ATOL = 1e-11    # FCNNLS itself loops; some f64 drift expected


def _ensure_ref():
    needed = ["V.tsv", "W0.tsv", "H0.tsv",
              "snmfR_W.tsv", "snmfR_H.tsv",
              "snmfL_W.tsv", "snmfL_H.tsv"]
    if all((DATA_DIR / f).exists() for f in needed):
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = "/scratch/users/steorra/env/CMAP/bin:" + env.get("PATH", "")
    subprocess.run(
        [RSCRIPT, str(REF_GEN), str(DATA_DIR),
         str(SEED), str(N), str(P), str(RANK), str(ITERS),
         str(ETA), str(BETA)],
        check=True, env=env,
    )


def _load(name: str) -> np.ndarray:
    return np.loadtxt(DATA_DIR / name, dtype=np.float64)


@pytest.fixture(scope="module")
def fixtures():
    _ensure_ref()
    return {
        "V":  _load("V.tsv"),
        "W0": _load("W0.tsv"),
        "H0": _load("H0.tsv"),
    }


def _assert_match(name, x_rust, x_r):
    diff = np.abs(x_rust - x_r)
    max_abs = float(diff.max())
    assert max_abs <= ATOL, (
        f"{name}: max |Δ| = {max_abs:.3e} > {ATOL:.0e} "
        f"(mean {diff.mean():.3e}, shape {x_rust.shape})"
    )


def test_snmf_r_parity(fixtures):
    V, W0, H0 = fixtures["V"], fixtures["W0"], fixtures["H0"]
    res = nmf_rs.nmf(V, rank=RANK, method="snmf/r",
                    W0=W0, H0=H0, max_iter=ITERS,
                    sparsity=BETA, smoothness=ETA,  # ETA=-1 → auto = max(V)
                    num_threads=1)
    _assert_match("snmf/R W", res.W, _load("snmfR_W.tsv"))
    _assert_match("snmf/R H", res.H, _load("snmfR_H.tsv"))


def test_snmf_l_parity(fixtures):
    V, W0, H0 = fixtures["V"], fixtures["W0"], fixtures["H0"]
    res = nmf_rs.nmf(V, rank=RANK, method="snmf/l",
                    W0=W0, H0=H0, max_iter=ITERS,
                    sparsity=BETA, smoothness=ETA,
                    num_threads=1)
    _assert_match("snmf/L W", res.W, _load("snmfL_W.tsv"))
    _assert_match("snmf/L H", res.H, _load("snmfL_H.tsv"))


def test_snmf_eta_explicit_matches_auto(fixtures):
    """Passing smoothness = max(V) explicitly should match smoothness = -1."""
    V, W0, H0 = fixtures["V"], fixtures["W0"], fixtures["H0"]
    res_auto = nmf_rs.nmf(V, rank=RANK, method="snmf/r",
                         W0=W0, H0=H0, max_iter=ITERS,
                         sparsity=BETA, smoothness=-1.0, num_threads=1)
    res_explicit = nmf_rs.nmf(V, rank=RANK, method="snmf/r",
                             W0=W0, H0=H0, max_iter=ITERS,
                             sparsity=BETA, smoothness=float(V.max()),
                             num_threads=1)
    np.testing.assert_array_equal(res_auto.W, res_explicit.W)
    np.testing.assert_array_equal(res_auto.H, res_explicit.H)
