"""Build & execute examples/benchmark_pbmc8k.ipynb.

End-to-end comparison on the omicverse pbmc8k dataset:
    1. Load pbmc8k.h5ad, select top-N HVGs, normalize+log.
    2. Generate W0/H0 once (in R, written to TSV).
    3. R `NMF::nmf(brunet/lee, max_iter=K)` — time it, save W_R/H_R.
    4. nmf-rs with same V/W0/H0 — time it, save W_rs/H_rs.
    5. Assert bit-equality and compute Pearson correlation column-by-column.
    6. Plot the wall-clock speedup.

Run from rust-NMF root with VIRTUAL_ENV unset / CONDA_PREFIX unset:
    /scratch/users/steorra/env/omicdev/bin/python examples/build_pbmc8k_notebook.py
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"
EX.mkdir(exist_ok=True)

KERNEL = "python3"


def md(*lines: str) -> dict:
    return nbformat.v4.new_markdown_cell("\n".join(lines))


def code(*lines: str) -> dict:
    return nbformat.v4.new_code_cell("\n".join(lines))


CELLS: List[dict] = [
    md(
        "# rust-NMF vs R `NMF` on omicverse PBMC 8k",
        "",
        "Real single-cell benchmark on the omicverse-bundled `pbmc8k.h5ad`",
        "(7750 cells × 20939 genes, 10x Genomics).",
        "",
        "Pipeline:",
        "1. Load + log-normalise + take top-`HVG_N` HVGs.",
        "2. Generate `W0`/`H0` in **R** so both backends use literally the",
        "   same starting point.",
        "3. Run `NMF::nmf(method='brunet')` and `nmf_rs.nmf('brunet')` for",
        "   `MAXIT` iterations. Same for `lee`.",
        "4. Compare bitwise / column-wise Pearson, plot wall-clock.",
    ),
    code(
        "import os, sys, time, subprocess, shutil",
        "from pathlib import Path",
        "import numpy as np, pandas as pd",
        "import scanpy as sc, anndata as ad",
        "import matplotlib.pyplot as plt",
        "import nmf_rs",
        "",
        "PBMC = '/scratch/users/steorra/analysis/omicverse_dev/omicverse/data/pbmc8k.h5ad'",
        "WORK = Path('pbmc_bench'); WORK.mkdir(exist_ok=True)",
        "RSCRIPT = '/scratch/users/steorra/env/CMAP/bin/Rscript'",
        "RLIB    = '/scratch/users/steorra/env/CMAP_Rlib'",
        "PATH_R  = '/scratch/users/steorra/env/CMAP/bin'",
        "HVG_N, RANK, MAXIT = 2000, 10, 100",
        "print('omicverse pbmc8k:', PBMC, 'exists:', Path(PBMC).exists())",
    ),
    md(
        "## 1. Load and preprocess",
        "",
        "Standard scanpy log-normalisation, then top-`HVG_N` HVGs by Seurat",
        "v3 dispersion. Result is `V` of shape (genes × cells), dense float64.",
    ),
    code(
        "ad_orig = ad.read_h5ad(PBMC)",
        "print('raw:', ad_orig.shape, 'X dtype:', ad_orig.X.dtype)",
        "ad_orig.layers['counts'] = ad_orig.X.copy()",
        "sc.pp.normalize_total(ad_orig, target_sum=1e4)",
        "sc.pp.log1p(ad_orig)",
        "sc.pp.highly_variable_genes(ad_orig, n_top_genes=HVG_N, flavor='seurat_v3', layer='counts')",
        "ad_hvg = ad_orig[:, ad_orig.var.highly_variable].copy()",
        "V = np.ascontiguousarray(ad_hvg.X.toarray().T.astype(np.float64))  # (genes × cells)",
        "print('V shape (genes × cells):', V.shape, 'min/max:', V.min(), V.max())",
        "n_genes, n_cells = V.shape",
        "np.savetxt(WORK/'V.tsv', V, delimiter='\\t')",
    ),
    md(
        "## 2. Seed W0/H0 in R",
        "",
        "R's `runif()` Mersenne Twister stream isn't directly accessible from",
        "Python. To enable a strict bit-parity comparison, we generate the",
        "initial factors in R and pickle them as TSV.",
    ),
    code(
        "rscript = f'''",
        "set.seed(2024L)",
        "V <- as.matrix(read.table(\"{WORK}/V.tsv\", sep=\"\\\\t\"))",
        "W0 <- matrix(runif({n_genes}*{RANK}, 0, max(V)), {n_genes}, {RANK})",
        "H0 <- matrix(runif({RANK}*{n_cells}, 0, max(V)), {RANK}, {n_cells})",
        "write.table(W0, \"{WORK}/W0.tsv\", sep=\"\\\\t\", row.names=FALSE, col.names=FALSE)",
        "write.table(H0, \"{WORK}/H0.tsv\", sep=\"\\\\t\", row.names=FALSE, col.names=FALSE)",
        "cat(\"seed_done\\\\n\")",
        "'''",
        "(WORK/'seed.R').write_text(rscript)",
        "out = subprocess.run([RSCRIPT, str(WORK/'seed.R')], capture_output=True, text=True,",
        "                      env={**os.environ, 'PATH': PATH_R + ':' + os.environ.get('PATH','')})",
        "print(out.stdout); print(out.stderr[-200:] if out.stderr else '')",
        "W0 = np.loadtxt(WORK/'W0.tsv'); H0 = np.loadtxt(WORK/'H0.tsv')",
        "W0.shape, H0.shape",
    ),
    md(
        "## 3. Run R `NMF::nmf(brunet)` and `lee`",
    ),
    code(
        "rscript = f'''",
        "suppressPackageStartupMessages({{",
        "  .libPaths(c(\"{RLIB}\", .libPaths()))",
        "  library(NMF)",
        "}})",
        "args <- commandArgs(trailingOnly=TRUE); method <- args[1]",
        "V  <- as.matrix(read.table(\"{WORK}/V.tsv\",  sep=\"\\\\t\"))",
        "W0 <- as.matrix(read.table(\"{WORK}/W0.tsv\", sep=\"\\\\t\"))",
        "H0 <- as.matrix(read.table(\"{WORK}/H0.tsv\", sep=\"\\\\t\"))",
        ".seed <- function(model, target, ...) {{ basis(model)<-W0; coef(model)<-H0; model }}",
        "t0 <- proc.time()[3]",
        "fit <- nmf(V, rank=ncol(W0), method=method, seed=.seed,",
        "           .options=\"-cb\", .pbackend=NA, nrun=1, maxIter={MAXIT},",
        "           stopconv=10L*{MAXIT})",
        "cat(sprintf(\"R_TIME %s %.4f\\\\n\", method, proc.time()[3] - t0))",
        "write.table(basis(fit), sprintf(\"{WORK}/%s__W_R.tsv\", method), sep=\"\\\\t\",",
        "            row.names=FALSE, col.names=FALSE)",
        "write.table(coef(fit),  sprintf(\"{WORK}/%s__H_R.tsv\", method), sep=\"\\\\t\",",
        "            row.names=FALSE, col.names=FALSE)",
        "'''",
        "(WORK/'run.R').write_text(rscript)",
        "r_times = {}",
        "for method in ('brunet', 'lee'):",
        "    out = subprocess.run([RSCRIPT, str(WORK/'run.R'), method], capture_output=True, text=True,",
        "                         env={**os.environ, 'PATH': PATH_R + ':' + os.environ.get('PATH','')})",
        "    print(out.stdout.splitlines()[-1])",
        "    for line in out.stdout.splitlines():",
        "        if line.startswith('R_TIME'):",
        "            _, m, t = line.split(); r_times[m] = float(t)",
        "r_times",
    ),
    md(
        "## 4. Run nmf-rs at several thread counts",
        "",
        "Each call passes `num_threads=N`, which builds a fresh rayon pool",
        "scoped to that single call — so different threading configurations",
        "can coexist in one process (unlike `set_num_threads`, which is a",
        "one-shot global init).",
    ),
    code(
        "THREADS = (1, 2, 4, 8, 16)",
        "rs_times = {nt: {} for nt in THREADS}",
        "for nt in THREADS:",
        "    for method in ('brunet', 'lee'):",
        "        t = time.perf_counter()",
        "        res = nmf_rs.nmf(V, rank=RANK, method=method, W0=W0.copy(), H0=H0.copy(),",
        "                         max_iter=MAXIT, stop='max_iter', num_threads=nt)",
        "        rs_times[nt][method] = time.perf_counter() - t",
        "        if nt == max(THREADS):",
        "            np.savetxt(WORK/f'{method}__W_rs.tsv', res.W, delimiter='\\t')",
        "            np.savetxt(WORK/f'{method}__H_rs.tsv', res.H, delimiter='\\t')",
        "        print(f'{method:6s} {nt}t: {rs_times[nt][method]:.2f} s')",
        "rs_times",
    ),
    md(
        "## 5. Bit-equivalence check + correlation",
        "",
        "Same V/W0/H0 → both backends should be bitwise identical.",
    ),
    code(
        "rows = []",
        "for method in ('brunet', 'lee'):",
        "    W_R = np.loadtxt(WORK/f'{method}__W_R.tsv'); H_R = np.loadtxt(WORK/f'{method}__H_R.tsv')",
        "    W_rs = np.loadtxt(WORK/f'{method}__W_rs.tsv'); H_rs = np.loadtxt(WORK/f'{method}__H_rs.tsv')",
        "    dW = float(np.abs(W_R - W_rs).max());  dH = float(np.abs(H_R - H_rs).max())",
        "    # Column-wise Pearson on W (factors) and on H (samples).",
        "    rW = np.array([np.corrcoef(W_R[:,k], W_rs[:,k])[0,1] for k in range(W_R.shape[1])])",
        "    rH = np.array([np.corrcoef(H_R[k,:], H_rs[k,:])[0,1] for k in range(H_R.shape[0])])",
        "    rows.append({'method': method,",
        "                 'max|ΔW|': dW, 'max|ΔH|': dH,",
        "                 'min corr W (per factor)': float(rW.min()),",
        "                 'min corr H (per factor)': float(rH.min()),",
        "                 'bit-equiv': dW < 1e-9 and dH < 1e-9})",
        "pd.DataFrame(rows)",
    ),
    md(
        "## 6. Wall-clock comparison",
    ),
    code(
        "table = pd.DataFrame({'R (s)': [r_times[m] for m in ('brunet', 'lee')]}, index=['brunet', 'lee'])",
        "for nt in THREADS:",
        "    table[f'Rust {nt}t (s)'] = [rs_times[nt][m] for m in ('brunet', 'lee')]",
        "    table[f'speed-up {nt}t']  = table['R (s)'] / table[f'Rust {nt}t (s)']",
        "table.round(3)",
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 3.2))",
        "x = np.arange(len(table)); width = 0.13",
        "n_bars = 1 + len(THREADS)  # R + each thread count",
        "colors = ['#4477aa', '#ddcc77', '#aaaa55', '#88ccaa', '#cc6677', '#aa3377']",
        "offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * width",
        "ax.bar(x + offsets[0], table['R (s)'].values, width, label='R NMF', color=colors[0])",
        "for k, nt in enumerate(THREADS):",
        "    ax.bar(x + offsets[k+1], table[f'Rust {nt}t (s)'].values, width,",
        "           label=f'nmf-rs {nt}t', color=colors[(k+1) % len(colors)])",
        "ax.set_xticks(x); ax.set_xticklabels(table.index)",
        "ax.set_ylabel(f'wall-clock seconds  (HVG={HVG_N}, cells={n_cells}, rank={RANK}, iters={MAXIT})')",
        "ax.legend(loc='upper right', ncol=2); ax.grid(axis='y', alpha=0.3); fig.tight_layout()",
    ),
    md(
        "## 7. Inspect the factors",
        "",
        "Quick QC of the rank-10 brunet factorisation: top-loaded gene per",
        "factor in `W` and the cell-type composition of the most-active cells",
        "per factor in `H`.",
    ),
    code(
        "W_rs = np.loadtxt(WORK/'brunet__W_rs.tsv')",
        "H_rs = np.loadtxt(WORK/'brunet__H_rs.tsv')",
        "gene_names = np.array(ad_hvg.var_names)",
        "top = pd.DataFrame({f'factor {k}': gene_names[np.argsort(-W_rs[:, k])[:5]] for k in range(RANK)})",
        "top",
    ),
    code(
        "ct_col = 'predicted_celltype' if 'predicted_celltype' in ad_hvg.obs.columns else 'cell_type'",
        "comp = pd.DataFrame({",
        "    f'factor {k}': ad_hvg.obs.iloc[np.argsort(-H_rs[k, :])[:50]][ct_col].value_counts().head(3)",
        "    for k in range(RANK)",
        "})",
        "comp.fillna(0).astype(int)",
    ),
    md(
        "## 8. Cleanup",
    ),
    code(
        "shutil.rmtree(WORK, ignore_errors=True)",
    ),
]


def main():
    nb = nbformat.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": KERNEL},
        "language_info": {"name": "python"},
    }
    nb_path = EX / "benchmark_pbmc8k.ipynb"
    nbformat.write(nb, nb_path)
    print(f"[build] wrote {nb_path}")
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name=KERNEL,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    client.execute()
    nbformat.write(nb, nb_path)
    print(f"[build] executed {nb_path}")


if __name__ == "__main__":
    main()
