#!/usr/bin/env python3
"""Microbenchmark full-loop fixed-factor Brunet vs the legacy Python loop.

Example:
    python examples/benchmark_fixed_factor.py --operation fixed-w --iterations 420
"""
from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable

import numpy as np

import nmf_rs


def _floor(value: np.ndarray, iteration: int) -> np.ndarray:
    if iteration % 10 == 0:
        return np.maximum(value, np.finfo(np.float64).eps)
    return value


def _objective(V: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    estimate = np.maximum(W @ H, np.finfo(np.float64).tiny)
    positive = V > 0
    return float(
        np.sum(
            np.where(
                positive,
                V * (np.log(np.maximum(V, np.finfo(np.float64).tiny)) - np.log(estimate))
                - V
                + estimate,
                estimate,
            )
        )
    )


def _time(call: Callable[[], np.ndarray], repeats: int) -> tuple[float, np.ndarray]:
    timings: list[float] = []
    output: np.ndarray | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = call()
        timings.append(time.perf_counter() - started)
    assert output is not None
    return statistics.median(timings), output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("fixed-h", "fixed-w"), default="fixed-w")
    parser.add_argument("--features", type=int, default=300)
    parser.add_argument("--observations", type=int, default=2000)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    W0 = rng.uniform(0.05, 1.0, size=(args.features, args.rank))
    H0 = rng.uniform(0.05, 1.0, size=(args.rank, args.observations))
    V = np.maximum(
        W0 @ H0 + rng.normal(0.0, 0.01, size=(args.features, args.observations)),
        0.0,
    )

    if args.operation == "fixed-h":
        def legacy() -> np.ndarray:
            W = W0.copy()
            for iteration in range(1, args.iterations + 1):
                W = _floor(nmf_rs.update_w_brunet(V, W, H0), iteration)
            _objective(V, W, H0)
            return W

        def full_rust() -> np.ndarray:
            W, _ = nmf_rs.fixed_h_brunet(
                V,
                H0,
                W0,
                max_iter=args.iterations,
                check_interval=args.iterations + 1,
                num_threads=args.threads,
            )
            return W
    else:
        def legacy() -> np.ndarray:
            H = H0.copy()
            for iteration in range(1, args.iterations + 1):
                H = _floor(nmf_rs.update_h_brunet(V, W0, H), iteration)
            _objective(V, W0, H)
            return H

        def full_rust() -> np.ndarray:
            H, _ = nmf_rs.fixed_w_brunet(
                V,
                W0,
                H0,
                max_iter=args.iterations,
                check_interval=args.iterations + 1,
                num_threads=args.threads,
            )
            return H

    # Give the legacy one-step calls the same worker budget as the scoped
    # full-loop call before either path initializes Rayon's global pool.
    nmf_rs.set_num_threads(args.threads)
    # Warm up extension loading and the Rayon pool before recording medians.
    full_rust()
    legacy_seconds, legacy_factor = _time(legacy, args.repeats)
    rust_seconds, rust_factor = _time(full_rust, args.repeats)
    np.testing.assert_allclose(rust_factor, legacy_factor, atol=1e-12, rtol=1e-10)

    print(
        f"operation={args.operation} V={V.shape} rank={args.rank} "
        f"iterations={args.iterations} threads={args.threads}"
    )
    print(f"legacy_python_loop_median_seconds={legacy_seconds:.6f}")
    print(f"full_rust_loop_median_seconds={rust_seconds:.6f}")
    print(f"speedup={legacy_seconds / rust_seconds:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
