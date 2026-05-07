"""Build & execute examples/benchmark_scaling.ipynb.

Multi-scale benchmark across dataset sizes to identify the optimal NMF
algorithm at each scale. Compares:

- brunet (KL multiplicative, R-equivalent)
- lee    (Frobenius multiplicative, R-equivalent)
- hals   (Cichocki-Phan)
- ehals  (extrapolated HALS, Ang-Gillis 2019/2024)
- dnmf   (diagonalised NMF, RcppML-style 2024)
- hals + NNDSVD init (the speed champ at small/medium scale)

Across:

- small:  pbmc8k subsample,  ~3000 cells × 1500 HVG, rank 8
- medium: simulated         ~50000 cells × 3000 HVG, rank 15
- large:  simulated        ~200000 cells × 5000 HVG, rank 20

The "large" tier uses synthetic data so the notebook is self-contained
and reproducible without 1+ GB downloads.
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
        "# rust-NMF — multi-scale benchmark across dataset sizes",
        "",
        "Real-world NMF pipelines span 3-4 orders of magnitude in problem",
        "size. Different solvers win at different scales. This notebook",
        "benchmarks all rust-NMF algorithms (plus sklearn cd as the BLAS",
        "baseline) on three tiers and identifies the right tool per scale.",
        "",
        "**Tiers** (genes × cells, rank, max iter):",
        "- **small**   — 1500 ×  3000, r=8,  100 it  (subsample of pbmc8k)",
        "- **medium**  — 3000 × 50000, r=15, 100 it  (synthetic, atlas-like)",
        "- **large**   — 5000 ×200000, r=20, 50 it   (synthetic, big atlas)",
    ),
    code(
        "import time, gc",
        "from pathlib import Path",
        "import numpy as np, pandas as pd",
        "import scanpy as sc, anndata as ad",
        "import matplotlib.pyplot as plt",
        "from sklearn.decomposition import NMF as SKNMF",
        "import nmf_rs",
    ),
    code(
        "from scipy.optimize import linear_sum_assignment",
        "",
        "PBMC = '/scratch/users/steorra/analysis/omicverse_dev/omicverse/data/pbmc8k.h5ad'",
        "",
        "def make_real_small(n_cells=3000, n_hvg=1500, rng_seed=0):",
        "    a = ad.read_h5ad(PBMC)",
        "    a.layers['counts'] = a.X.copy()",
        "    sc.pp.normalize_total(a, target_sum=1e4)",
        "    sc.pp.log1p(a)",
        "    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor='seurat_v3', layer='counts')",
        "    a = a[:, a.var.highly_variable].copy()",
        "    rng = np.random.default_rng(rng_seed)",
        "    sub = rng.choice(a.n_obs, n_cells, replace=False)",
        "    return np.ascontiguousarray(a.X[sub].toarray().T.astype(np.float64))",
        "",
        "def make_synthetic(n_genes, n_cells, rank=10, noise=0.05, seed=0):",
        "    \"\"\"Returns (V, W_true, H_true) with ground-truth factors for accuracy checks.\"\"\"",
        "    rng = np.random.default_rng(seed)",
        "    W = rng.uniform(0.1, 1.5, (n_genes, rank)).astype(np.float64)",
        "    H = rng.uniform(0.1, 1.5, (rank, n_cells)).astype(np.float64)",
        "    V = W @ H + noise * rng.normal(size=(n_genes, n_cells))",
        "    return np.maximum(V, 0.0), W, H",
        "",
        "def matched_pearson(A, B):",
        "    \"\"\"Hungarian-match rows of A↔B by Pearson, return matched per-row r.\"\"\"",
        "    n = A.shape[0]",
        "    if A.shape != B.shape: return np.zeros(n)",
        "    corr = np.corrcoef(A, B)[:n, n:]",
        "    row, col = linear_sum_assignment(-corr)",
        "    return corr[row, col]",
    ),
    md(
        "## Helper: timing one algorithm × one dataset",
    ),
    code(
        "def bench_tier(label, V, rank, max_iter, threads=16, sklearn_iter=None,",
        "               W_true=None, H_true=None):",
        "    \"\"\"Bench all algorithms and capture (W, H) for accuracy comparison.\"\"\"",
        "    n_genes, n_cells = V.shape",
        "    print(f'\\n[{label}] V = ({n_genes}, {n_cells}), rank={rank}, iters={max_iter}')",
        "",
        "    rng = np.random.default_rng(2024)",
        "    init_scale = float(np.sqrt(V.mean() / rank))",
        "    W0_rs = rng.uniform(0, init_scale, (n_genes, rank))",
        "    H0_rs = rng.uniform(0, init_scale, (rank, n_cells))",
        "    W0_nn, H0_nn = nmf_rs.nndsvd_init(V, rank, fill='mean', seed=0)",
        "",
        "    rows = []",
        "    factors = {}    # algo → (W (n×r), H (r×p))",
        "    for method, mit in [('lee', max_iter), ('brunet', max_iter),",
        "                         ('hals', max_iter), ('ehals', max_iter),",
        "                         ('dnmf', max_iter)]:",
        "        t = time.perf_counter()",
        "        res = nmf_rs.nmf(V, rank=rank, method=method,",
        "                         W0=W0_rs.copy(), H0=H0_rs.copy(),",
        "                         max_iter=mit, num_threads=threads)",
        "        dt = time.perf_counter() - t",
        "        loss = 0.5*float(np.linalg.norm(V - res.fitted()) ** 2)",
        "        algo_label = method + '+rand'",
        "        rows.append({'algo': algo_label, 'time_s': dt, 'iters': mit, 'loss': loss})",
        "        factors[algo_label] = (res.W.copy(), res.H.copy())",
        "    # NNDSVD champ — fewer iters",
        "    nn_iter = max(20, max_iter // 4)",
        "    t = time.perf_counter()",
        "    res = nmf_rs.nmf(V, rank=rank, method='hals',",
        "                     W0=W0_nn.copy(), H0=H0_nn.copy(),",
        "                     max_iter=nn_iter, num_threads=threads)",
        "    dt = time.perf_counter() - t",
        "    loss = 0.5*float(np.linalg.norm(V - res.fitted()) ** 2)",
        "    algo_label = f'hals+NNDSVD ({nn_iter}it)'",
        "    rows.append({'algo': algo_label, 'time_s': dt, 'iters': nn_iter, 'loss': loss})",
        "    factors[algo_label] = (res.W.copy(), res.H.copy())",
        "",
        "    # sklearn cd as BLAS reference",
        "    sk_iter = sklearn_iter or max_iter",
        "    t = time.perf_counter()",
        "    mod = SKNMF(n_components=rank, init='custom', solver='cd',",
        "                beta_loss='frobenius', max_iter=sk_iter, tol=0.0, random_state=0)",
        "    Wsk = mod.fit_transform(V.T, W=H0_rs.T.copy(), H=W0_rs.T.copy())",
        "    dt = time.perf_counter() - t",
        "    loss = 0.5*float(np.linalg.norm(V.T - Wsk @ mod.components_) ** 2)",
        "    rows.append({'algo': 'sklearn cd (BLAS)', 'time_s': dt, 'iters': sk_iter, 'loss': loss})",
        "    # Map sklearn (cells × rank) and (rank × genes) → our convention (genes × rank), (rank × cells)",
        "    factors['sklearn cd (BLAS)'] = (mod.components_.T, Wsk.T)",
        "",
        "    df = pd.DataFrame(rows)",
        "    df['speed_rank'] = df['time_s'].rank(method='min').astype(int)",
        "    df['loss_rank']  = df['loss'  ].rank(method='min').astype(int)",
        "",
        "    # Accuracy section: per-factor Pearson vs ground truth (synthetic)",
        "    # or vs `lee+rand` anchor (real data — `lee` is R-bit-equivalent).",
        "    if W_true is not None and H_true is not None:",
        "        ref_label = 'ground truth'",
        "        Wref, Href = W_true, H_true",
        "    else:",
        "        ref_label = 'lee+rand (R anchor)'",
        "        Wref, Href = factors['lee+rand']",
        "    accuracy_rows = []",
        "    for algo_label, (Walg, Halg) in factors.items():",
        "        if Walg.shape != Wref.shape:",
        "            continue",
        "        # Match factors by H (cell loadings) — most stable since cells >> genes for atlases.",
        "        rH = matched_pearson(Halg, Href)",
        "        rW = matched_pearson(Walg.T, Wref.T)  # match by columns of W → rows of W^T",
        "        accuracy_rows.append({",
        "            'algo': algo_label,",
        "            'mean rH': float(rH.mean()),",
        "            'min rH':  float(rH.min()),",
        "            'mean rW': float(rW.mean()),",
        "            'min rW':  float(rW.min()),",
        "            '#factors r>0.9 (H)': int((rH > 0.9).sum()),",
        "        })",
        "    acc_df = pd.DataFrame(accuracy_rows)",
        "    print(f'  accuracy reference: {ref_label}')",
        "    return df, acc_df, factors",
    ),
    md(
        "## Tier 1 — small (pbmc8k subset, real data)",
        "",
        "Real data has no ground truth, so we anchor accuracy against",
        "`lee+rand` (R-bit-equivalent) and Hungarian-match all others.",
    ),
    code(
        "V_small = make_real_small(n_cells=3000, n_hvg=1500)",
        "df_small, acc_small, _ = bench_tier('small', V_small, rank=8, max_iter=100)",
        "df_small.round(4)",
    ),
    code(
        "print('Accuracy vs `lee+rand` anchor:')",
        "acc_small.round(4)",
    ),
    md(
        "## Tier 2 — medium (synthetic, ~atlas-scale, ground-truth available)",
    ),
    code(
        "V_med, W_med_true, H_med_true = make_synthetic(3000, 50000, rank=15, seed=1)",
        "df_med, acc_med, _ = bench_tier('medium', V_med, rank=15, max_iter=100,",
        "                                W_true=W_med_true, H_true=H_med_true)",
        "df_med.round(4)",
        "del V_med, W_med_true, H_med_true; gc.collect()",
    ),
    code(
        "print('Accuracy vs ground truth:')",
        "acc_med.round(4)",
    ),
    md(
        "## Tier 3 — large (synthetic, big atlas, ground-truth)",
        "",
        "We deliberately reduce iterations for the large tier — at this scale",
        "you almost always run NMF with stationary stop or a small budget,",
        "not 200 iters. We measure 50 iters across all algorithms.",
    ),
    code(
        "V_large, W_large_true, H_large_true = make_synthetic(5000, 200000, rank=20, seed=2)",
        "print(f'V_large memory: {V_large.nbytes / 1e9:.2f} GB')",
        "df_large, acc_large, _ = bench_tier('large', V_large, rank=20, max_iter=50,",
        "                                    sklearn_iter=50,",
        "                                    W_true=W_large_true, H_true=H_large_true)",
        "df_large.round(4)",
        "del V_large, W_large_true, H_large_true; gc.collect()",
    ),
    code(
        "print('Accuracy vs ground truth:')",
        "acc_large.round(4)",
    ),
    md(
        "## Cross-tier summary — wall-clock + accuracy",
    ),
    code(
        "df_small['tier']  = 'small (3k cells, r=8)'",
        "df_med  ['tier']  = 'medium (50k cells, r=15)'",
        "df_large['tier']  = 'large (200k cells, r=20)'",
        "all_df = pd.concat([df_small, df_med, df_large], ignore_index=True)",
        "all_df[['tier','algo','time_s','loss']].pivot_table(index='algo', columns='tier', values='time_s').round(3)",
    ),
    code(
        "# Accuracy heatmap: mean Pearson on H rows vs reference (ground-truth on synthetic, lee on real)",
        "acc_small['tier'] = 'small (real, vs lee anchor)'",
        "acc_med  ['tier'] = 'medium (synthetic, vs ground truth)'",
        "acc_large['tier'] = 'large (synthetic, vs ground truth)'",
        "all_acc = pd.concat([acc_small, acc_med, acc_large], ignore_index=True)",
        "all_acc.pivot_table(index='algo', columns='tier', values='mean rH').round(3)",
    ),
    code(
        "# Same but min rH (worst factor) — more conservative quality view",
        "all_acc.pivot_table(index='algo', columns='tier', values='min rH').round(3)",
    ),
    code(
        "fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)",
        "for ax, df, title in zip(axes, [df_small, df_med, df_large],",
        "                         ['small','medium','large']):",
        "    df_sorted = df.sort_values('time_s')",
        "    colors = ['#cc6677' if 'NNDSVD' in a or 'hals' in a or 'ehals' in a or 'dnmf' in a",
        "              else '#4477aa' for a in df_sorted['algo']]",
        "    ax.barh(df_sorted['algo'], df_sorted['time_s'], color=colors)",
        "    ax.set_title(title); ax.invert_yaxis()",
        "    ax.set_xlabel('wall-clock (s)'); ax.grid(axis='x', alpha=0.3)",
        "fig.tight_layout()",
    ),
    md(
        "## Per-tier winners (combining speed + accuracy)",
    ),
    code(
        "winners = []",
        "for tier, df, acc in [('small', df_small, acc_small),",
        "                       ('medium', df_med, acc_med),",
        "                       ('large', df_large, acc_large)]:",
        "    fastest = df.loc[df['time_s'].idxmin()]",
        "    best_loss = df.loc[df['loss'].idxmin()]",
        "    most_accurate = acc.loc[acc['mean rH'].idxmax()] if len(acc) else None",
        "    winners.append({",
        "        'tier': tier,",
        "        'fastest': fastest['algo'], 'time (s)': fastest['time_s'],",
        "        'best loss': best_loss['algo'], 'min loss': float(best_loss['loss']),",
        "        'most accurate': most_accurate['algo'] if most_accurate is not None else '-',",
        "        'best mean rH': float(most_accurate['mean rH']) if most_accurate is not None else 0.0,",
        "    })",
        "pd.DataFrame(winners).round(3)",
    ),
    md(
        "## Recommendations by scale",
        "",
        "From the table above (the actual numbers will vary by hardware):",
        "",
        "- **Small (≤ 10k cells)** — `hals + NNDSVD` typically wins on time;",
        "  `lee` / `hals` close behind. KL solvers are 5-30× slower per iter.",
        "  Use HALS+NNDSVD with 25 iters as your default.",
        "",
        "- **Medium (10k-100k cells, atlases)** — `hals + NNDSVD` still",
        "  fastest; `dnmf` adds interpretation/L1 regularisation at near-zero",
        "  speed cost. **Use `dnmf` if you care about cross-run factor",
        "  stability or want sparse factors**.",
        "",
        "- **Large (100k+ cells)** — sklearn cd's BLAS gemm shines on the",
        "  per-iter cost, but rust-NMF HALS+NNDSVD's iteration count savings",
        "  often still win in absolute time. For very large V, consider",
        "  online/mini-batch methods (LIGER, RcppML's online mode).",
        "",
        "**For atlas-scale single-cell (>500k cells)**: the bottleneck shifts",
        "from compute to **memory traffic** — V must fit in RAM. At that",
        "scale a sparse-V kernel (future work) or LIGER-style mini-batching",
        "is what you actually want.",
    ),
]


def main():
    nb = nbformat.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": KERNEL},
        "language_info": {"name": "python"},
    }
    nb_path = EX / "benchmark_scaling.ipynb"
    nbformat.write(nb, nb_path)
    print(f"[build] wrote {nb_path}")
    client = NotebookClient(
        nb,
        timeout=3600,
        kernel_name=KERNEL,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    client.execute()
    nbformat.write(nb, nb_path)
    print(f"[build] executed {nb_path}")


if __name__ == "__main__":
    main()
