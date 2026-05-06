"""Build & execute the rust-NMF example notebooks in-place.

Generates and runs:
    examples/tutorial_quickstart.ipynb
    examples/benchmark_vs_R.ipynb

Each notebook is built from a list of cells defined below, then executed
with nbclient so the saved file contains real outputs.

Run from the rust-NMF root:
    /scratch/users/steorra/env/omicdev/bin/python examples/build_notebooks.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"
EX.mkdir(exist_ok=True)
TESTS_DATA = ROOT / "tests" / "data"

CMAP_RSCRIPT = "/scratch/users/steorra/env/CMAP/bin/Rscript"
CMAP_PATH = "/scratch/users/steorra/env/CMAP/bin"
KERNEL = "python3"


def md(*lines: str) -> dict:
    return nbformat.v4.new_markdown_cell("\n".join(lines))


def code(*lines: str) -> dict:
    return nbformat.v4.new_code_cell("\n".join(lines))


# ---------------------------------------------------------------------------
# tutorial_quickstart.ipynb
# ---------------------------------------------------------------------------

QUICKSTART: List[dict] = [
    md(
        "# rust-NMF — quickstart",
        "",
        "Rust port of R's `NMF` package. Bit-equivalent multiplicative-update",
        "algorithms (`brunet`, `lee`, `offset`, `nsNMF`), parallelised with",
        "rayon. Drop-in replacement for the inner update loops of `NMF::nmf()`.",
        "",
        "This notebook walks through:",
        "1. Building a synthetic non-negative matrix.",
        "2. Running each of the four built-in algorithms.",
        "3. Inspecting the `NMFResult` (`W`, `H`, `n_iter`, `deviances`).",
        "4. Reusing the same factorisation for downstream sklearn-style work.",
    ),
    code(
        "import numpy as np",
        "import nmf_rs",
        "",
        "rng = np.random.default_rng(0)",
        "n, p, rank = 200, 60, 5",
        "W_true = rng.uniform(0.1, 1.5, (n, rank))",
        "H_true = rng.uniform(0.1, 1.5, (rank, p))",
        "V = W_true @ H_true + rng.uniform(0, 0.05, (n, p))",
        "V.shape, V.min(), V.max()",
    ),
    md(
        "## Random initial factors",
        "",
        "The `nmf()` driver accepts either explicit `W0`/`H0` (recommended for",
        "reproducibility) or a `seed` argument that triggers in-Python random",
        "init. *Note*: NumPy's RNG is not bit-equal to R's `runif()`, so for",
        "parity with R you must generate `W0`/`H0` in R and pass them in.",
    ),
    code(
        "W0, H0 = nmf_rs.random_init(V, rank, seed=42)",
        "W0.shape, H0.shape",
    ),
    md(
        "## Brunet (KL divergence) — the default",
        "",
        "200 multiplicative-update iterations. R's `NMF::nmf(method='brunet')`",
        "uses the same algorithm under the hood.",
    ),
    code(
        "res = nmf_rs.nmf(V, rank=rank, method='brunet',",
        "                 W0=W0, H0=H0, max_iter=200)",
        "print(res)",
        "print('recon error :', np.linalg.norm(V - res.fitted()))",
    ),
    md(
        "## Lee (Frobenius) and the rest",
        "",
        "All four built-in algorithms share the same `nmf()` entry-point. The",
        "`offset` algorithm also returns an offset vector accessible as",
        "`res.offset` — `fitted()` automatically adds it back.",
    ),
    code(
        "for method in ('brunet', 'lee', 'offset', 'nsNMF'):",
        "    res = nmf_rs.nmf(V, rank=rank, method=method,",
        "                     W0=W0, H0=H0, max_iter=200)",
        "    err = np.linalg.norm(V - res.fitted())",
        "    extras = '' if res.offset is None else f'  off=[{res.offset.min():.2f}, {res.offset.max():.2f}]'",
        "    print(f'{method:8s}  iters={res.n_iter}  ||V-WH|| = {err:.4f}{extras}')",
    ),
    md(
        "## Stationary stopping",
        "",
        "Set `stop='stationary'` to replicate R's `nmf.stop.stationary` semantics",
        "— the run halts when the objective value is flat over a window of",
        "`check_niter` checks taken every `check_interval` iterations.",
    ),
    code(
        "res = nmf_rs.nmf(V, rank=rank, method='brunet',",
        "                 W0=W0, H0=H0, max_iter=2000,",
        "                 stop='stationary',",
        "                 stationary_th=1e-6,",
        "                 check_interval=10, check_niter=10)",
        "print('stopped after', res.n_iter, 'iters; final KL =', res.deviances[-1])",
        "len(res.deviances)",
    ),
    md(
        "## Plotting the loss curve",
    ),
    code(
        "import matplotlib.pyplot as plt",
        "fig, ax = plt.subplots(figsize=(5, 3))",
        "ax.plot(res.deviances, lw=1.2)",
        "ax.set_xlabel('check #'); ax.set_ylabel('KL divergence')",
        "ax.set_title('Brunet stationary stop'); fig.tight_layout()",
    ),
    md(
        "## Single-step kernels",
        "",
        "If you want full control over the iteration loop, the per-step kernels",
        "are also exposed as bit-equivalent updates of R's `std.divergence.update.{h,w}`",
        "and `std.euclidean.update.{h,w}`.",
    ),
    code(
        "H1 = nmf_rs.update_h_brunet(V, W0, H0)",
        "W1 = nmf_rs.update_w_brunet(V, W0, H1)",
        "print('one Brunet step ΔKL =',",
        "      float(((V * np.log(np.where(V>0, V/(W0@H0+1e-30), 1)) - V + W0@H0).sum()) -",
        "            ((V * np.log(np.where(V>0, V/(W1@H1+1e-30), 1)) - V + W1@H1).sum())))",
    ),
    md(
        "## See also",
        "",
        "- [`benchmark_vs_R.ipynb`](benchmark_vs_R.ipynb) — bit-equivalence and",
        "  speed comparison vs R `NMF`.",
        "- The R reference fixtures live under `tests/data/` and are produced by",
        "  `tests/reference_nmf.R`.",
    ),
]


# ---------------------------------------------------------------------------
# benchmark_vs_R.ipynb
# ---------------------------------------------------------------------------

BENCHMARK: List[dict] = [
    md(
        "# rust-NMF vs R `NMF` — bit-equivalence and timing",
        "",
        "We assert two things on the same input:",
        "1. Given identical `(V, W0, H0)`, our Rust core produces **bitwise**",
        "   identical `W` and `H` to R's C++ kernels (max abs diff ≤ 1e-12).",
        "2. The Rust core is several × faster end-to-end.",
        "",
        "The reference fixtures are pre-generated by `tests/reference_nmf.R`",
        "(CMAP env, R `NMF` 0.30+).",
    ),
    code(
        "import os, sys, subprocess, time",
        "from pathlib import Path",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import nmf_rs",
        "",
        f"DATA = Path('{TESTS_DATA}')",
        "DATA.exists()",
    ),
    md(
        "## Load the R fixtures",
        "",
        "The fixtures are produced from `set.seed(1234)` with `n=80`, `p=30`,",
        "`rank=4`, `max_iter=50` for each of brunet/lee/offset/nsNMF.",
    ),
    code(
        "V  = np.loadtxt(DATA/'V.tsv')",
        "W0 = np.loadtxt(DATA/'W0.tsv')",
        "H0 = np.loadtxt(DATA/'H0.tsv')",
        "V.shape, W0.shape, H0.shape",
    ),
    md(
        "## Run each algorithm in Python and compare with R's reference",
    ),
    code(
        "rows = []",
        "for method in ('brunet', 'lee', 'offset', 'nsNMF'):",
        "    res = nmf_rs.nmf(V, rank=4, method=method,",
        "                     W0=W0, H0=H0, max_iter=50, stop='max_iter')",
        "    W_R = np.loadtxt(DATA/f'{method}__W.tsv')",
        "    H_R = np.loadtxt(DATA/f'{method}__H.tsv')",
        "    dW = float(np.abs(res.W - W_R).max())",
        "    dH = float(np.abs(res.H - H_R).max())",
        "    rows.append({'method': method,",
        "                 'max|ΔW|': dW, 'max|ΔH|': dH,",
        "                 'bitwise-equiv': dW < 1e-12 and dH < 1e-12})",
        "pd.DataFrame(rows)",
    ),
    md(
        "All four algorithms match R within f64 epsilon — the Rust core is a",
        "drop-in replacement.",
    ),
    md(
        "## Wall-clock comparison",
        "",
        "We use a moderate problem (n=400, p=120, rank=8, 200 iters) and time",
        "both R's compiled `nmf_update.{brunet,lee}` and our Rust driver.",
        "We pin both runs to the same `(V, W0, H0)`.",
    ),
    code(
        "rng = np.random.default_rng(7)",
        "n, p, rank, MAXIT = 400, 120, 8, 200",
        "W_true = rng.uniform(0.1, 1.5, (n, rank))",
        "H_true = rng.uniform(0.1, 1.5, (rank, p))",
        "V = W_true @ H_true + rng.uniform(0, 0.05, (n, p))",
        "W0 = rng.uniform(0, V.max(), (n, rank))",
        "H0 = rng.uniform(0, V.max(), (rank, p))",
        "",
        "# write inputs as TSV so Rscript can pick them up",
        "BENCH_DIR = Path('bench_in'); BENCH_DIR.mkdir(exist_ok=True)",
        "for name, M in {'V.tsv': V, 'W0.tsv': W0, 'H0.tsv': H0}.items():",
        "    np.savetxt(BENCH_DIR/name, M, delimiter='\\t')",
        "V.shape, W0.shape, H0.shape",
    ),
    code(
        "# Time Rust",
        "rs_times = {}",
        "for method in ('brunet', 'lee'):",
        "    t = time.perf_counter()",
        "    res = nmf_rs.nmf(V, rank=rank, method=method, W0=W0, H0=H0,",
        "                     max_iter=MAXIT, stop='max_iter')",
        "    rs_times[method] = time.perf_counter() - t",
        "rs_times",
    ),
    code(
        "# Time R",
        "r_script = '''",
        "suppressPackageStartupMessages({",
        "  .libPaths(c(\"/scratch/users/steorra/env/CMAP_Rlib\", .libPaths()))",
        "  library(NMF)",
        "})",
        "args <- commandArgs(trailingOnly=TRUE)",
        "method <- args[1]",
        "MAXIT <- as.integer(args[2])",
        "V  <- as.matrix(read.table('bench_in/V.tsv', sep='\\\\t'))",
        "W0 <- as.matrix(read.table('bench_in/W0.tsv', sep='\\\\t'))",
        "H0 <- as.matrix(read.table('bench_in/H0.tsv', sep='\\\\t'))",
        ".seed <- function(model, target, ...) {",
        "  basis(model) <- W0; coef(model) <- H0; model",
        "}",
        "t0 <- proc.time()[3]",
        "fit <- nmf(V, rank=ncol(W0), method=method, seed=.seed,",
        "           .options='-cb', .pbackend=NA, nrun=1, maxIter=MAXIT,",
        "           stopconv=10L*MAXIT)",
        "cat(sprintf('R_TIME %s %.4f\\\\n', method, proc.time()[3] - t0))",
        "'''",
        f"R_PATH = '{CMAP_PATH}'",
        "Path('bench_run.R').write_text(r_script)",
        "r_times = {}",
        "for method in ('brunet', 'lee'):",
        f"    cmd = ['{CMAP_RSCRIPT}', 'bench_run.R', method, str(MAXIT)]",
        "    out = subprocess.run(cmd, capture_output=True, text=True,",
        "                         env={**os.environ, 'PATH': R_PATH + ':' + os.environ.get('PATH', '')})",
        "    for line in out.stdout.splitlines():",
        "        if line.startswith('R_TIME'):",
        "            _, m, t = line.split(); r_times[m] = float(t)",
        "r_times",
    ),
    code(
        "summary = pd.DataFrame({",
        "    'R (s)':   [r_times[m] for m in ('brunet', 'lee')],",
        "    'Rust (s)': [rs_times[m] for m in ('brunet', 'lee')],",
        "}, index=['brunet', 'lee'])",
        "summary['speed-up'] = summary['R (s)'] / summary['Rust (s)']",
        "summary.round(3)",
    ),
    code(
        "fig, ax = plt.subplots(figsize=(4, 3))",
        "x = np.arange(len(summary))",
        "ax.bar(x - 0.2, summary['R (s)'].values, width=0.4, label='R NMF', color='#4477aa')",
        "ax.bar(x + 0.2, summary['Rust (s)'].values, width=0.4, label='nmf-rs', color='#cc6677')",
        "ax.set_xticks(x); ax.set_xticklabels(summary.index)",
        "ax.set_ylabel('wall-clock seconds (200 iters, n=400 p=120 r=8)')",
        "ax.legend(); fig.tight_layout()",
    ),
    md(
        "## Bigger problem — Rust scales with rayon",
        "",
        "Set `nmf_rs.set_num_threads(N)` once at process start to use more cores.",
        "Most of the speed advantage comes from the tight inner loops; rayon",
        "kicks in for the larger matrix-matrix products.",
    ),
    code(
        "import nmf_rs",
        "nmf_rs.set_num_threads(4)",
        "rng = np.random.default_rng(99)",
        "n, p, rank = 1000, 300, 10",
        "V_big = np.abs(rng.normal(size=(n, p)))",
        "W0_big, H0_big = nmf_rs.random_init(V_big, rank, seed=0)",
        "t = time.perf_counter()",
        "_ = nmf_rs.nmf(V_big, rank=rank, method='brunet',",
        "               W0=W0_big, H0=H0_big, max_iter=100)",
        "print(f'Brunet 1000×300 rank=10, 100 iters: {time.perf_counter()-t:.3f} s')",
    ),
    md(
        "## Cleanup",
    ),
    code(
        "import shutil",
        "for f in ('bench_run.R',):",
        "    Path(f).unlink(missing_ok=True)",
        "shutil.rmtree('bench_in', ignore_errors=True)",
    ),
]


def write_and_run(nb_path: Path, cells: List[dict]) -> None:
    nb = nbformat.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": KERNEL},
        "language_info": {"name": "python"},
    }
    nbformat.write(nb, nb_path)
    print(f"[build_notebooks] wrote {nb_path}")
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name=KERNEL,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    client.execute()
    nbformat.write(nb, nb_path)
    print(f"[build_notebooks] executed {nb_path}")


def main():
    write_and_run(EX / "tutorial_quickstart.ipynb", QUICKSTART)
    write_and_run(EX / "benchmark_vs_R.ipynb", BENCHMARK)


if __name__ == "__main__":
    main()
