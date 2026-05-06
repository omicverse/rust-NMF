"""Smoke / sanity tests for lsNMF, snmf/r, snmf/l.

We don't claim bit-equivalence with R for these:
- lsNMF uses BLAS gemm in R (variable summation order across BLAS impls).
- snmf/r and snmf/l in R use FCNNLS-based ANLS (Van Benthem 2004); our port
  uses regularised HALS instead. Same objective sense (sparse W or H), but
  different solver, so different iterates.

These tests verify:
- lsNMF with `weight=1` gives results close to standard Lee (~within typical
  multiplicative-update tolerance, since R-Lee and lsNMF-with-ones theoretically
  converge to the same fixpoint).
- lsNMF with masked entries (`weight[i,j]=0`) matches the unmasked V on the
  kept entries — i.e. masking actually drops them from the loss.
- snmf/r reduces H density vs plain HALS at non-zero `sparsity`.
- snmf/l reduces W density vs plain HALS at non-zero `sparsity`.
"""
from __future__ import annotations

import numpy as np
import pytest

import nmf_rs


@pytest.fixture(scope="module")
def synth():
    rng = np.random.default_rng(42)
    n, p, rank = 120, 60, 6
    W_true = rng.uniform(0.1, 1.5, (n, rank))
    H_true = rng.uniform(0.1, 1.5, (rank, p))
    V = W_true @ H_true + rng.uniform(0, 0.05, (n, p))
    W0, H0 = nmf_rs.random_init(V, rank, seed=0)
    return {"V": V, "W0": W0, "H0": H0, "rank": rank}


def test_lsnmf_uniform_weight_close_to_lee(synth):
    """lsNMF with weight=1 should converge to roughly the same fixpoint as Lee."""
    V, W0, H0, rank = synth["V"], synth["W0"], synth["H0"], synth["rank"]
    weight = np.ones_like(V)
    res_ls = nmf_rs.nmf(V, rank=rank, method="lsnmf", weight=weight,
                       W0=W0, H0=H0, max_iter=200, num_threads=2)
    res_lee = nmf_rs.nmf(V, rank=rank, method="lee",
                        W0=W0, H0=H0, max_iter=200, num_threads=2)
    err_ls = np.linalg.norm(V - res_ls.fitted())
    err_lee = np.linalg.norm(V - res_lee.fitted())
    # Both should reach a similar reconstruction quality.
    assert err_ls < err_lee * 1.05 and err_ls > err_lee * 0.95, (
        f"lsNMF(w=1) recon {err_ls:.4f} vs Lee {err_lee:.4f}"
    )


def test_lsnmf_masking_drops_entries(synth):
    """Setting weight[i,j]=0 should remove (V[i,j] - WH[i,j])² from the loss —
    so the unmasked subset should be reconstructed *better* than if we'd left
    the masked-out entries in (which would pull the factorisation off-target)."""
    V, W0, H0, rank = synth["V"], synth["W0"], synth["H0"], synth["rank"]
    rng = np.random.default_rng(7)
    mask = rng.random(V.shape) > 0.20  # 20% missing
    weight = mask.astype(np.float64)
    V_masked = V.copy(); V_masked[~mask] = 0  # zero out missing values

    res = nmf_rs.nmf(V_masked, rank=rank, method="lsnmf", weight=weight,
                    W0=W0, H0=H0, max_iter=200, num_threads=2)
    fit = res.fitted()
    rmse_observed = np.sqrt(((V[mask] - fit[mask]) ** 2).mean())
    # On observed entries the model should fit V much better than 0.5
    # (if it learned to fit V_masked=0 everywhere, RMSE on observed would be
    # close to mean(V)≈1).
    assert rmse_observed < 0.3, (
        f"lsNMF on observed entries RMSE = {rmse_observed:.3f}; "
        "expected the masked-out entries not to drag the fit"
    )


def test_snmf_r_makes_h_sparser(synth):
    """snmf/R reduces H density at non-zero β vs unconstrained Lee."""
    V, W0, H0, rank = synth["V"], synth["W0"], synth["H0"], synth["rank"]
    res_lee = nmf_rs.nmf(V, rank=rank, method="lee",
                        W0=W0, H0=H0, max_iter=200, num_threads=2)
    res_s = nmf_rs.nmf(V, rank=rank, method="snmf/r",
                      sparsity=0.5, smoothness=-1.0,
                      W0=W0, H0=H0, max_iter=20, num_threads=2)
    density_h_lee  = float((res_lee.H > 1e-6).mean())
    density_h_snmf = float((res_s.H > 1e-6).mean())
    assert density_h_snmf < density_h_lee, (
        f"snmf/r should make H sparser: lee={density_h_lee:.3f} "
        f"vs snmf/r={density_h_snmf:.3f}"
    )


def test_snmf_l_makes_w_sparser(synth):
    """snmf/L reduces W density at non-zero β vs unconstrained Lee."""
    V, W0, H0, rank = synth["V"], synth["W0"], synth["H0"], synth["rank"]
    res_lee = nmf_rs.nmf(V, rank=rank, method="lee",
                        W0=W0, H0=H0, max_iter=200, num_threads=2)
    res_s = nmf_rs.nmf(V, rank=rank, method="snmf/l",
                      sparsity=0.5, smoothness=-1.0,
                      W0=W0, H0=H0, max_iter=20, num_threads=2)
    density_w_lee  = float((res_lee.W > 1e-6).mean())
    density_w_snmf = float((res_s.W > 1e-6).mean())
    assert density_w_snmf < density_w_lee, (
        f"snmf/l should make W sparser: lee={density_w_lee:.3f} "
        f"vs snmf/l={density_w_snmf:.3f}"
    )


def test_lsnmf_requires_weight(synth):
    V, W0, H0, rank = synth["V"], synth["W0"], synth["H0"], synth["rank"]
    with pytest.raises((ValueError, RuntimeError)):
        nmf_rs.nmf(V, rank=rank, method="lsnmf",
                  W0=W0, H0=H0, max_iter=10, num_threads=1)


def test_method_aliases():
    """All advertised aliases dispatch to the same algorithm."""
    rng = np.random.default_rng(0)
    V = np.abs(rng.normal(size=(40, 20)))
    W0, H0 = nmf_rs.random_init(V, 3, seed=0)
    weight = np.ones_like(V)

    pairs = [
        ("ls-nmf", "lsnmf"),
        ("snmf/r", "snmf_r"),
        ("snmf/l", "snmf_l"),
    ]
    for a, b in pairs:
        kw_a = dict(W0=W0, H0=H0, max_iter=20, num_threads=1)
        kw_b = dict(kw_a)
        if a in ("ls-nmf",):
            kw_a["weight"] = weight
            kw_b["weight"] = weight
        ra = nmf_rs.nmf(V, rank=3, method=a, **kw_a)
        rb = nmf_rs.nmf(V, rank=3, method=b, **kw_b)
        np.testing.assert_array_equal(ra.W, rb.W, err_msg=f"{a} vs {b}")
        np.testing.assert_array_equal(ra.H, rb.H, err_msg=f"{a} vs {b}")
