"""Contracts for the full-loop fixed-factor Brunet APIs.

The existing one-step kernels are R NMF parity anchors.  These tests prove
that the new one-call Rust loops preserve those updates while adding the
Spatial EcoTyper flooring, connectivity stopping, immutability, and metrics
contracts.
"""
from __future__ import annotations

import hashlib
from importlib import metadata

import numpy as np
import pytest

import nmf_rs


def _matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    V = np.array(
        [
            [3.0, 1.0, 4.0, 2.0, 5.0],
            [1.0, 5.0, 2.0, 4.0, 3.0],
            [2.0, 3.0, 1.0, 5.0, 4.0],
            [5.0, 2.0, 3.0, 1.0, 4.0],
        ],
        dtype=np.float64,
    )
    W = np.array(
        [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]],
        dtype=np.float64,
    )
    H = np.array(
        [[0.9, 0.8, 0.2, 0.1, 0.7], [0.1, 0.2, 0.8, 0.9, 0.3]],
        dtype=np.float64,
    )
    return V, W, H


def _repeat_w(V: np.ndarray, W0: np.ndarray, H: np.ndarray, n: int) -> np.ndarray:
    W = W0.copy()
    for iteration in range(1, n + 1):
        W = nmf_rs.update_w_brunet(V, W, H)
        if iteration % 10 == 0:
            W = np.maximum(W, np.finfo(np.float64).eps)
    return W


def _repeat_h(V: np.ndarray, W: np.ndarray, H0: np.ndarray, n: int) -> np.ndarray:
    H = H0.copy()
    for iteration in range(1, n + 1):
        H = nmf_rs.update_h_brunet(V, W, H)
        if iteration % 10 == 0:
            H = np.maximum(H, np.finfo(np.float64).eps)
    return H


@pytest.mark.parametrize("iterations", [1, 10])
def test_fixed_h_matches_r_parity_single_step_kernel(iterations: int) -> None:
    V, W0, H_fixed = _matrices()
    expected = _repeat_w(V, W0, H_fixed, iterations)

    actual, metrics = nmf_rs.fixed_h_brunet(
        V,
        H_fixed,
        W0,
        max_iter=iterations,
        check_interval=10,
        stopconv=40,
        num_threads=1,
    )

    np.testing.assert_array_equal(actual, expected)
    assert metrics["iterations"] == iterations
    assert metrics["eps_floor_applications"] == iterations // 10
    assert not metrics["converged_by_connectivity"]


@pytest.mark.parametrize("iterations", [1, 10])
def test_fixed_w_matches_r_parity_single_step_kernel(iterations: int) -> None:
    V, W_fixed, H0 = _matrices()
    expected = _repeat_h(V, W_fixed, H0, iterations)

    actual, metrics = nmf_rs.fixed_w_brunet(
        V,
        W_fixed,
        H0,
        max_iter=iterations,
        check_interval=10,
        stopconv=40,
        num_threads=1,
    )

    np.testing.assert_array_equal(actual, expected)
    assert metrics["iterations"] == iterations
    assert metrics["eps_floor_applications"] == iterations // 10
    assert not metrics["converged_by_connectivity"]


def test_default_connectivity_contract_stops_at_420() -> None:
    # Identity W makes the first H update equal V and keeps the partition
    # stable, so this fixture isolates the strict `stable > stopconv` rule.
    V = np.array([[4.0, 3.0, 1.0, 1.0], [1.0, 1.0, 3.0, 4.0]])
    W_fixed = np.eye(2, dtype=np.float64)
    H0 = np.array([[0.8, 0.7, 0.2, 0.1], [0.2, 0.3, 0.8, 0.9]])

    H, metrics = nmf_rs.fixed_w_brunet(
        V, W_fixed, H0, max_iter=2000, num_threads=1
    )

    assert H.shape == H0.shape
    assert metrics["iterations"] == 420
    assert metrics["connectivity_checks"] == 42
    assert metrics["stable_connectivity_checks"] == 41
    assert metrics["converged"] is True
    assert metrics["converged_by_connectivity"] is True


def test_non_multiple_max_iter_is_a_hard_ceiling() -> None:
    V, W_fixed, H0 = _matrices()
    H, metrics = nmf_rs.fixed_w_brunet(
        V, W_fixed, H0, max_iter=17, num_threads=1
    )

    np.testing.assert_array_equal(H, _repeat_h(V, W_fixed, H0, 17))
    assert metrics["iterations"] == 17
    assert metrics["eps_floor_applications"] == 1
    assert metrics["connectivity_checks"] == 1
    assert metrics["converged"] is False


@pytest.mark.parametrize("operation", ["fixed_h", "fixed_w"])
def test_fixed_input_is_unchanged_and_attested(operation: str) -> None:
    V, W, H = _matrices()
    fixed = H if operation == "fixed_h" else W
    before = fixed.tobytes(order="C")
    expected_hash = hashlib.sha256(before).hexdigest()

    if operation == "fixed_h":
        _, metrics = nmf_rs.fixed_h_brunet(V, fixed, W, max_iter=10, num_threads=2)
        expected_name = "H"
    else:
        _, metrics = nmf_rs.fixed_w_brunet(V, fixed, H, max_iter=10, num_threads=2)
        expected_name = "W"

    assert fixed.tobytes(order="C") == before
    attestation = metrics["fixed_factor"]
    assert attestation["name"] == expected_name
    assert attestation["sha256_before"] == expected_hash
    assert attestation["sha256_after"] == expected_hash
    assert attestation["byte_for_byte_unchanged"] is True
    assert metrics["num_threads"] == 2
    assert metrics["backend"]["num_threads"] == 2


def test_invalid_inputs_fail_closed() -> None:
    V, W, H = _matrices()

    with pytest.raises(ValueError, match="negative"):
        nmf_rs.fixed_h_brunet(np.where(V == V[0, 0], -1.0, V), H, W, max_iter=1)
    with pytest.raises(ValueError, match="non-finite"):
        nmf_rs.fixed_h_brunet(V, np.where(H == H[0, 0], np.nan, H), W, max_iter=1)
    with pytest.raises(ValueError, match="shape"):
        nmf_rs.fixed_h_brunet(V, H[:, :-1], W, max_iter=1)
    with pytest.raises(ValueError, match="max_iter"):
        nmf_rs.fixed_h_brunet(V, H, W, max_iter=0)
    with pytest.raises(ValueError, match="check_interval"):
        nmf_rs.fixed_w_brunet(V, W, H, max_iter=1, check_interval=0)
    with pytest.raises(ValueError, match="num_threads"):
        nmf_rs.fixed_w_brunet(V, W, H, max_iter=1, num_threads=0)


def test_positive_target_with_zero_reconstruction_fails_closed() -> None:
    # The first feature has no support in the fixed basis.  Its positive
    # target therefore has R NMF KL divergence +Inf even though the H update
    # itself remains finite from the supported second feature.
    V = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    W_fixed = np.array([[0.0], [1.0]], dtype=np.float64)
    H0 = np.ones((1, 2), dtype=np.float64)

    with pytest.raises(RuntimeError, match="non-finite KL objective"):
        nmf_rs.fixed_w_brunet(V, W_fixed, H0, max_iter=1, num_threads=1)


def test_public_api_version_and_legacy_updates_remain_available() -> None:
    assert metadata.version("nmf-rs") == "0.2.0"
    assert callable(nmf_rs.fixed_h_brunet)
    assert callable(nmf_rs.fixed_w_brunet)
    assert callable(nmf_rs.update_h_brunet)
    assert callable(nmf_rs.update_w_brunet)
