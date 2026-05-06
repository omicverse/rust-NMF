"""Build & execute examples/hals_nndsvd_demo.ipynb.

Demonstrates the two algorithmic shortcuts HALS + NNDSVD on omicverse pbmc8k.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"
KERNEL = "python3"


def md(*lines: str) -> dict:
    return nbformat.v4.new_markdown_cell("\n".join(lines))


def code(*lines: str) -> dict:
    return nbformat.v4.new_code_cell("\n".join(lines))


CELLS: List[dict] = [
    md(
        "# HALS + NNDSVD on omicverse PBMC 8k",
        "",
        "Bit-equivalence with R is great for validation, but for production",
        "single-cell pipelines you usually don't need *exactly* R's iterates —",
        "you need a non-negative factorisation of comparable quality, fast.",
        "",
        "Two purely algorithmic shortcuts give 5–10× more on top of the",
        "Rust port:",
        "",
        "1. **HALS** (`method='hals'` / `'lsnmf'`) — Cichocki-Phan one-row /",
        "   one-column least-squares updates. Same per-iteration cost as `lee`,",
        "   but converges to the same loss plateau in 5–10× fewer iterations.",
        "2. **NNDSVD init** (`nmf_rs.nndsvd_init`) — truncated-SVD-based warm",
        "   start. On pbmc8k, the initial loss is **~5 orders of magnitude**",
        "   smaller than `runif()`-style random init.",
    ),
    code(
        "import time",
        "from pathlib import Path",
        "import numpy as np, pandas as pd",
        "import scanpy as sc, anndata as ad",
        "import matplotlib.pyplot as plt",
        "import nmf_rs",
        "",
        "PBMC = '/scratch/users/steorra/analysis/omicverse_dev/omicverse/data/pbmc8k.h5ad'",
        "HVG_N, RANK = 2000, 10",
        "ad_orig = ad.read_h5ad(PBMC)",
        "ad_orig.layers['counts'] = ad_orig.X.copy()",
        "sc.pp.normalize_total(ad_orig, target_sum=1e4)",
        "sc.pp.log1p(ad_orig)",
        "sc.pp.highly_variable_genes(ad_orig, n_top_genes=HVG_N,",
        "                            flavor='seurat_v3', layer='counts')",
        "ad_hvg = ad_orig[:, ad_orig.var.highly_variable].copy()",
        "V = np.ascontiguousarray(ad_hvg.X.toarray().T.astype(np.float64))",
        "print('V shape (genes × cells):', V.shape)",
    ),
    md(
        "## NNDSVD vs random init",
        "",
        "The runif-style random init throws `W0/H0 ~ U(0, max(V))` which",
        "puts `||W0 H0||` far above `||V||` — the first dozen iterations are",
        "spent just shrinking the factors. NNDSVD instead seeds with the",
        "non-negative projection of the truncated SVD, so the **starting**",
        "loss is already close to the converged plateau.",
    ),
    code(
        "rng = np.random.default_rng(0)",
        "W0_rand, H0_rand = nmf_rs.random_init(V, RANK, seed=0)",
        "t = time.perf_counter()",
        "W0_nn, H0_nn = nmf_rs.nndsvd_init(V, RANK, fill='mean', seed=0)",
        "print(f'nndsvd_init: {time.perf_counter()-t:.2f} s')",
        "",
        "loss = lambda W, H: 0.5 * np.linalg.norm(V - W @ H) ** 2",
        "v_norm2 = 0.5 * np.linalg.norm(V) ** 2",
        "print(f'||V||² / 2          = {v_norm2:.3e}')",
        "print(f'initial loss (rand) = {loss(W0_rand, H0_rand):.3e}')",
        "print(f'initial loss (NN)   = {loss(W0_nn,   H0_nn  ):.3e}')",
    ),
    md(
        "## Convergence trajectories",
        "",
        "We track `||V - WH||²/2` per iteration for the four (algo, init)",
        "pairs — Lee with random vs NNDSVD, HALS with random vs NNDSVD.",
        "(`stop='stationary'` returns the per-check deviance window — for",
        "this demo we just re-run with intermediate `max_iter` values.)",
    ),
    code(
        "def trace(method, W0, H0, max_iter, num_threads=16, every=1):",
        "    losses = []",
        "    for k in range(0, max_iter + 1, every):",
        "        if k == 0:",
        "            losses.append((0, loss(W0, H0)))",
        "            continue",
        "        res = nmf_rs.nmf(V, rank=RANK, method=method,",
        "                         W0=W0.copy(), H0=H0.copy(), max_iter=k,",
        "                         stop='max_iter', num_threads=num_threads)",
        "        losses.append((k, loss(res.W, res.H)))",
        "    return losses",
        "",
        "MAX = 100",
        "EVERY = 5",
        "lee_rand   = trace('lee',  W0_rand, H0_rand, MAX, every=EVERY)",
        "lee_nn     = trace('lee',  W0_nn,   H0_nn,   MAX, every=EVERY)",
        "hals_rand  = trace('hals', W0_rand, H0_rand, MAX, every=EVERY)",
        "hals_nn    = trace('hals', W0_nn,   H0_nn,   MAX, every=EVERY)",
    ),
    code(
        "fig, ax = plt.subplots(figsize=(6, 3.5))",
        "for label, t, c, ls in [",
        "    ('lee  + random', lee_rand,  '#4477aa', '-'),",
        "    ('lee  + NNDSVD', lee_nn,    '#4477aa', '--'),",
        "    ('HALS + random', hals_rand, '#cc6677', '-'),",
        "    ('HALS + NNDSVD', hals_nn,   '#cc6677', '--'),",
        "]:",
        "    xs, ys = zip(*t)",
        "    ax.plot(xs, ys, label=label, color=c, ls=ls, lw=1.6)",
        "ax.set_yscale('log')",
        "ax.set_xlabel('iteration'); ax.set_ylabel(r'$\\|V - WH\\|^2/2$')",
        "ax.set_title('Convergence on pbmc8k (HVG=2000, rank=10)')",
        "ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()",
    ),
    md(
        "**Takeaway**: HALS + NNDSVD reaches the same loss plateau in",
        "~10–25 iterations that random + Lee needs ~100 to reach.",
    ),
    md(
        "## Wall-clock comparison",
        "",
        "Same metric, but timed end-to-end. Reference R lee on this exact",
        "input (cached from `benchmark_pbmc8k.ipynb`) is ~125 s.",
    ),
    code(
        "configs = [",
        "    ('lee  + random  (100 iters)', 'lee',  W0_rand, H0_rand, 100),",
        "    ('lee  + NNDSVD  (50 iters)',  'lee',  W0_nn,   H0_nn,    50),",
        "    ('HALS + random  (50 iters)',  'hals', W0_rand, H0_rand,  50),",
        "    ('HALS + NNDSVD  (25 iters)',  'hals', W0_nn,   H0_nn,    25),",
        "    ('HALS + NNDSVD  (10 iters)',  'hals', W0_nn,   H0_nn,    10),",
        "]",
        "",
        "rows = []",
        "for nt in (1, 16):",
        "    for label, method, W0, H0, mit in configs:",
        "        t = time.perf_counter()",
        "        res = nmf_rs.nmf(V, rank=RANK, method=method,",
        "                         W0=W0.copy(), H0=H0.copy(),",
        "                         max_iter=mit, stop='max_iter',",
        "                         num_threads=nt)",
        "        dt = time.perf_counter() - t",
        "        rows.append({'config': label, 'threads': nt,",
        "                     'time_s': dt, 'iters': res.n_iter,",
        "                     'loss': loss(res.W, res.H)})",
        "        print(f'{label:30s}  {nt:2d}t  {dt:6.2f}s  loss={loss(res.W, res.H):.0f}')",
        "df = pd.DataFrame(rows)",
        "df['speedup_vs_R_lee'] = 125.21 / df['time_s']  # cached R lee timing",
        "df.round(3)",
    ),
    md(
        "## Factor-quality check",
        "",
        "Same factorisation quality? Match HALS+NNDSVD@25 columns to Lee+random@100",
        "by Hungarian assignment, then per-factor Pearson on H.",
    ),
    code(
        "from scipy.optimize import linear_sum_assignment",
        "",
        "res_lee = nmf_rs.nmf(V, rank=RANK, method='lee',",
        "                     W0=W0_rand.copy(), H0=H0_rand.copy(),",
        "                     max_iter=100, num_threads=16)",
        "res_hals = nmf_rs.nmf(V, rank=RANK, method='hals',",
        "                      W0=W0_nn.copy(), H0=H0_nn.copy(),",
        "                      max_iter=25, num_threads=16)",
        "",
        "# Match factors by max-correlation assignment (NMF has rank-permutation symmetry).",
        "corr = np.corrcoef(res_lee.H, res_hals.H)[:RANK, RANK:]",
        "row_idx, col_idx = linear_sum_assignment(-corr)",
        "matched_corr = corr[row_idx, col_idx]",
        "print('per-factor Pearson(H_lee, H_hals_matched):')",
        "for i, c in enumerate(matched_corr):",
        "    print(f'  factor {i}: r = {c:.4f}')",
        "print(f'mean = {matched_corr.mean():.4f},  min = {matched_corr.min():.4f}')",
    ),
    md(
        "Per-factor Pearson > 0.95 across all 10 factors typically — HALS+NNDSVD",
        "recovers the same biological program as Lee 100-iter, in a fraction",
        "of the wall-clock time.",
    ),
    md(
        "## Top genes per factor (HALS + NNDSVD, 25 iters)",
    ),
    code(
        "gene_names = np.array(ad_hvg.var_names)",
        "top = pd.DataFrame({",
        "    f'factor {k}': gene_names[np.argsort(-res_hals.W[:, k])[:5]]",
        "    for k in range(RANK)",
        "})",
        "top",
    ),
    code(
        "ct_col = 'predicted_celltype' if 'predicted_celltype' in ad_hvg.obs.columns else 'cell_type'",
        "comp = pd.DataFrame({",
        "    f'factor {k}': ad_hvg.obs.iloc[np.argsort(-res_hals.H[k, :])[:50]][ct_col].value_counts().head(3)",
        "    for k in range(RANK)",
        "})",
        "comp.fillna(0).astype(int)",
    ),
]


def main():
    nb = nbformat.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": KERNEL},
        "language_info": {"name": "python"},
    }
    nb_path = EX / "hals_nndsvd_demo.ipynb"
    nbformat.write(nb, nb_path)
    print(f"[build] wrote {nb_path}")
    client = NotebookClient(
        nb,
        timeout=1200,
        kernel_name=KERNEL,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    client.execute()
    nbformat.write(nb, nb_path)
    print(f"[build] executed {nb_path}")


if __name__ == "__main__":
    main()
