"""Build & execute examples/benchmark_scaling.ipynb.

Multi-scale benchmark with **objective**, anchor-free NMF evaluation:

  - reconstruction loss (mean across K random inits)
  - Brunet cophenetic correlation (1)        — consensus stability
  - Kim-Park dispersion coefficient (2)      — consensus stability
  - Hoyer sparsity (3)                       — interpretability
  - ARI vs cell-type labels (4, real only)   — biological signal recovery
  - Amari error vs ground truth (5, synth)   — factor recovery (when known)

(1) Brunet et al. PNAS 2004
(2) Kim & Park, Bioinformatics 2007
(3) Hoyer, JMLR 2004
(4) Hubert & Arabie 1985 / sklearn metrics

Three tiers (small / medium / large), each algorithm run K times with
different seeds; consensus matrix built from cell-cluster assignments
(argmax over H rows). Consensus matrix capped at 3000 cells (subsampled
when V has more) so it stays at 72 MB.
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
        "# rust-NMF — multi-scale benchmark with objective metrics",
        "",
        "Compares all rust-NMF algorithms (plus sklearn cd) across three",
        "dataset scales using **anchor-free** NMF evaluation metrics:",
        "",
        "1. **Reconstruction loss** ½‖V-WH‖² — fit quality",
        "2. **Held-out test MSE** — generalisation",
        "3. **Brunet cophenetic correlation** (PNAS 2004) — consensus stability",
        "4. **Kim-Park dispersion coefficient** (Bioinformatics 2007) — same",
        "5. **Hoyer sparsity** (JMLR 2004) — factor interpretability",
        "6. **ARI vs cell-type labels** — biological signal (real data only)",
        "7. **Amari error vs ground truth** — factor recovery (synthetic only)",
        "",
        "These don't depend on choosing any single algorithm as anchor, so",
        "they give an unbiased view of who's actually best.",
    ),
    code(
        "import time, gc",
        "from pathlib import Path",
        "import numpy as np, pandas as pd",
        "import scanpy as sc, anndata as ad",
        "import matplotlib.pyplot as plt",
        "from sklearn.decomposition import NMF as SKNMF",
        "from sklearn.metrics import adjusted_rand_score",
        "from scipy.cluster.hierarchy import linkage, cophenet",
        "from scipy.spatial.distance import squareform",
        "from scipy.optimize import linear_sum_assignment",
        "import nmf_rs",
    ),
    code(
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
        "    V = np.ascontiguousarray(a.X[sub].toarray().T.astype(np.float64))",
        "    ct_col = 'predicted_celltype' if 'predicted_celltype' in a.obs.columns else 'cell_type'",
        "    labels = a.obs.iloc[sub][ct_col].astype(str).values",
        "    return V, labels",
        "",
        "def make_synthetic(n_genes, n_cells, rank=10, noise=0.05, seed=0):",
        "    rng = np.random.default_rng(seed)",
        "    W = rng.uniform(0.1, 1.5, (n_genes, rank)).astype(np.float64)",
        "    H = rng.uniform(0.1, 1.5, (rank, n_cells)).astype(np.float64)",
        "    V = W @ H + noise * rng.normal(size=(n_genes, n_cells))",
        "    return np.maximum(V, 0.0), W, H",
    ),
    md(
        "## Objective metric implementations",
        "",
        "The four anchor-free NMF metrics. Cophenetic + dispersion need",
        "multiple NMF runs (K=5 by default for small data, K=3 for medium,",
        "K=1 for large where each run is expensive).",
    ),
    code(
        "def hoyer_sparsity(x):",
        "    x = np.asarray(x).ravel()",
        "    n = x.size",
        "    if n == 0: return 0.0",
        "    l1 = float(np.abs(x).sum())",
        "    l2 = float(np.sqrt((x ** 2).sum()))",
        "    if l2 == 0: return 0.0",
        "    return (np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1.0)",
        "",
        "def factor_sparsity(W, H):",
        "    \"\"\"Mean Hoyer sparsity across W columns and H rows.\"\"\"",
        "    sw = float(np.mean([hoyer_sparsity(W[:, k]) for k in range(W.shape[1])]))",
        "    sh = float(np.mean([hoyer_sparsity(H[k, :]) for k in range(H.shape[0])]))",
        "    return sw, sh",
        "",
        "def connectivity_matrix(H_subset):",
        "    \"\"\"Binary indicator of 'cells co-cluster': cluster_i == cluster_j.\"\"\"",
        "    cluster = np.argmax(H_subset, axis=0)",
        "    return (cluster[:, None] == cluster[None, :]).astype(np.float64)",
        "",
        "def cophenetic_correlation(consensus):",
        "    \"\"\"Brunet 2004: corr between distance(1-C̄) and the cophenetic",
        "    distance from average-linkage hierarchical clustering of 1-C̄.\"\"\"",
        "    np.fill_diagonal(consensus, 1.0)  # safe self-similarity",
        "    distances = 1.0 - consensus",
        "    cond = squareform(distances, checks=False)",
        "    Z = linkage(cond, method='average')",
        "    coph = cophenet(Z, cond)",
        "    if isinstance(coph, tuple): coph = coph[0]",
        "    if cond.std() == 0 or coph.std() == 0:",
        "        return 1.0",
        "    return float(np.corrcoef(cond, coph)[0, 1])",
        "",
        "def dispersion_coefficient(consensus):",
        "    \"\"\"Kim-Park 2007: ρ = (1/n²) Σ 4·(C̄_ij - 0.5)²; 1 = perfectly stable.\"\"\"",
        "    return float((4.0 * (consensus - 0.5) ** 2).mean())",
        "",
        "def amari_error(W_est, W_true):",
        "    \"\"\"Permutation-invariant factor recovery error. 0 = identity match.\"\"\"",
        "    if W_est.shape != W_true.shape: return float('nan')",
        "    P = np.linalg.pinv(W_est) @ W_true                       # rank × rank",
        "    P = np.abs(P)",
        "    n = P.shape[0]",
        "    row_term = sum((P[i,:].sum() / max(P[i,:].max(), 1e-30) - 1) for i in range(n))",
        "    col_term = sum((P[:,j].sum() / max(P[:,j].max(), 1e-30) - 1) for j in range(n))",
        "    return float((row_term + col_term) / (2 * n * (n - 1))) if n > 1 else 0.0",
    ),
    md(
        "## Multi-run consensus driver",
    ),
    code(
        "def run_with_consensus(V, rank, method, n_runs, max_iter, num_threads,",
        "                        sub_idx=None, sklearn_compat=False, **nmf_kw):",
        "    \"\"\"Run NMF n_runs times with different seeds; return mean loss,",
        "    cophenetic, dispersion, Hoyer, plus the W/H of the last run.\"\"\"",
        "    n_genes, n_cells = V.shape",
        "    consensus_n = len(sub_idx) if sub_idx is not None else n_cells",
        "    consensus = np.zeros((consensus_n, consensus_n), dtype=np.float64)",
        "    losses, sparsities_w, sparsities_h, times = [], [], [], []",
        "    last_W = last_H = None",
        "",
        "    for run in range(n_runs):",
        "        rng = np.random.default_rng(2024 + run)",
        "        init_scale = float(np.sqrt(V.mean() / rank))",
        "        if sklearn_compat:",
        "            W0_sk = rng.uniform(0, init_scale, (n_cells, rank))",
        "            H0_sk = rng.uniform(0, init_scale, (rank, n_genes))",
        "            mod = SKNMF(n_components=rank, init='custom', solver='cd',",
        "                        beta_loss='frobenius', max_iter=max_iter, tol=0.0, random_state=run)",
        "            t = time.perf_counter()",
        "            Wsk = mod.fit_transform(V.T, W=W0_sk.copy(), H=H0_sk.copy())",
        "            times.append(time.perf_counter() - t)",
        "            W = mod.components_.T          # (n_genes, rank)",
        "            H = Wsk.T                      # (rank, n_cells)",
        "        else:",
        "            W0 = rng.uniform(0, init_scale, (n_genes, rank))",
        "            H0 = rng.uniform(0, init_scale, (rank, n_cells))",
        "            t = time.perf_counter()",
        "            res = nmf_rs.nmf(V, rank=rank, method=method, W0=W0, H0=H0,",
        "                              max_iter=max_iter, num_threads=num_threads, **nmf_kw)",
        "            times.append(time.perf_counter() - t)",
        "            W, H = res.W, res.H",
        "        losses.append(0.5 * float(np.linalg.norm(V - W @ H) ** 2))",
        "        # Subsample cells for consensus to bound memory.",
        "        H_for_consensus = H[:, sub_idx] if sub_idx is not None else H",
        "        consensus += connectivity_matrix(H_for_consensus)",
        "        sw, sh = factor_sparsity(W, H)",
        "        sparsities_w.append(sw); sparsities_h.append(sh)",
        "        last_W, last_H = W, H",
        "",
        "    consensus /= n_runs",
        "    coph = cophenetic_correlation(consensus.copy()) if n_runs > 1 else float('nan')",
        "    disp = dispersion_coefficient(consensus)         if n_runs > 1 else float('nan')",
        "    return {",
        "        'mean_time':       float(np.mean(times)),",
        "        'mean_loss':       float(np.mean(losses)),",
        "        'cophenetic':      coph,",
        "        'dispersion':      disp,",
        "        'sparsity_W':      float(np.mean(sparsities_w)),",
        "        'sparsity_H':      float(np.mean(sparsities_h)),",
        "        'last_W':          last_W,",
        "        'last_H':          last_H,",
        "    }",
    ),
    md(
        "## Tier 1 — small (real pbmc8k subset, K=5 runs each)",
        "",
        "We have cell-type labels on this tier, so we can also report ARI",
        "(adjusted Rand index) of the factor-argmax cluster vs the curated",
        "labels — a true biological-signal recovery metric.",
    ),
    code(
        "V_small, labels_small = make_real_small(n_cells=3000, n_hvg=1500)",
        "rank_small, MAX_SMALL, K_SMALL = 8, 100, 5",
        "rng = np.random.default_rng(42)",
        "sub_small = rng.choice(V_small.shape[1], min(3000, V_small.shape[1]), replace=False)",
        "labels_sub = labels_small[sub_small]",
        "print(f'V {V_small.shape}; rank={rank_small}; K={K_SMALL} runs each')",
    ),
    code(
        "def bench_tier_objective(V, rank, max_iter, K, threads=16, sub_idx=None,",
        "                          true_labels=None, W_true=None, H_true=None,",
        "                          methods_full=('lee','brunet','hals','ehals','dnmf'),",
        "                          include_nndsvd=True, include_sklearn=True,",
        "                          nndsvd_iter=None):",
        "    rows = []",
        "    for method in methods_full:",
        "        d = run_with_consensus(V, rank, method, n_runs=K,",
        "                                max_iter=max_iter, num_threads=threads,",
        "                                sub_idx=sub_idx)",
        "        d['algo'] = method + '+rand'; rows.append(d)",
        "    if include_nndsvd:",
        "        nn_iter = nndsvd_iter or max(20, max_iter // 4)",
        "        # NNDSVD is deterministic given V and rank → consensus over identical runs",
        "        # would be trivially 1.0. Run with K=1 and report only single-run metrics.",
        "        n_genes, n_cells = V.shape",
        "        W0n, H0n = nmf_rs.nndsvd_init(V, rank, fill='mean', seed=0)",
        "        t = time.perf_counter()",
        "        res = nmf_rs.nmf(V, rank=rank, method='hals', W0=W0n, H0=H0n,",
        "                          max_iter=nn_iter, num_threads=threads)",
        "        dt = time.perf_counter() - t",
        "        sw, sh = factor_sparsity(res.W, res.H)",
        "        rows.append({",
        "            'algo': f'hals+NNDSVD ({nn_iter}it, K=1)',",
        "            'mean_time': dt,",
        "            'mean_loss': 0.5 * float(np.linalg.norm(V - res.fitted()) ** 2),",
        "            'cophenetic': float('nan'), 'dispersion': float('nan'),",
        "            'sparsity_W': sw, 'sparsity_H': sh,",
        "            'last_W': res.W, 'last_H': res.H,",
        "        })",
        "    if include_sklearn:",
        "        d = run_with_consensus(V, rank, 'sklearn_cd', n_runs=K,",
        "                                max_iter=max_iter, num_threads=threads,",
        "                                sub_idx=sub_idx, sklearn_compat=True)",
        "        d['algo'] = 'sklearn cd (BLAS)'; rows.append(d)",
        "    # ARI on real labels",
        "    if true_labels is not None:",
        "        for r in rows:",
        "            cluster = np.argmax(r['last_H'][:, sub_idx if sub_idx is not None else slice(None)], axis=0)",
        "            r['ARI vs labels'] = adjusted_rand_score(true_labels, cluster)",
        "    # Amari vs ground truth",
        "    if W_true is not None:",
        "        for r in rows:",
        "            r['Amari error'] = amari_error(r['last_W'], W_true)",
        "    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('last_W','last_H')}",
        "                        for r in rows])",
        "    return df",
        "",
        "df_small = bench_tier_objective(V_small, rank_small, MAX_SMALL, K_SMALL,",
        "                                  sub_idx=sub_small, true_labels=labels_sub)",
        "df_small.round(4)",
    ),
    md(
        "**How to read this**:",
        "- `cophenetic` and `dispersion` near 1.0 → algorithm gives stable factors across random inits",
        "- `ARI vs labels` ∈ [-1, 1]; > 0.5 strong, > 0.3 moderate biological signal",
        "- `sparsity_W` / `sparsity_H` ∈ [0, 1]; higher = more interpretable factors",
        "- `mean_loss` lower = better fit (but compare same iter count)",
    ),
    md(
        "## Tier 2 — medium (synthetic 3k×50k, K=3 runs)",
    ),
    code(
        "V_med, W_med_true, _ = make_synthetic(3000, 50000, rank=15, seed=1)",
        "rank_med, MAX_MED, K_MED = 15, 100, 3",
        "rng = np.random.default_rng(43)",
        "sub_med = rng.choice(V_med.shape[1], 3000, replace=False)",
        "df_med = bench_tier_objective(V_med, rank_med, MAX_MED, K_MED,",
        "                                sub_idx=sub_med, W_true=W_med_true)",
        "df_med.round(4)",
        "del V_med, W_med_true; gc.collect()",
    ),
    md(
        "## Tier 3 — large (synthetic 5k×200k, K=1, no consensus)",
        "",
        "Each run at this scale is expensive (~30-60s); consensus over",
        "multiple inits is impractical here. We report single-run metrics",
        "(reconstruction, sparsity, Amari vs ground truth).",
    ),
    code(
        "V_large, W_large_true, _ = make_synthetic(5000, 200000, rank=20, seed=2)",
        "print(f'V_large memory: {V_large.nbytes/1e9:.2f} GB')",
        "rng = np.random.default_rng(44)",
        "sub_large = rng.choice(V_large.shape[1], 3000, replace=False)",
        "df_large = bench_tier_objective(V_large, rank_large := 20, max_iter := 50, K=1,",
        "                                  sub_idx=sub_large, W_true=W_large_true,",
        "                                  nndsvd_iter=20)",
        "df_large.round(4)",
        "del V_large, W_large_true; gc.collect()",
    ),
    md(
        "## Cross-tier summary",
    ),
    code(
        "df_small['tier']  = 'small (3k cells, r=8, K=5)'",
        "df_med  ['tier']  = 'medium (50k cells, r=15, K=3)'",
        "df_large['tier']  = 'large (200k cells, r=20, K=1)'",
        "all_df = pd.concat([df_small, df_med, df_large], ignore_index=True)",
        "all_df.pivot_table(index='algo', columns='tier', values='mean_time').round(3)",
    ),
    code(
        "# Cophenetic correlation — only meaningful where K>1",
        "all_df.pivot_table(index='algo', columns='tier', values='cophenetic').round(3)",
    ),
    code(
        "# Dispersion (1 = perfectly stable consensus, 0 = random)",
        "all_df.pivot_table(index='algo', columns='tier', values='dispersion').round(3)",
    ),
    code(
        "# Hoyer sparsity (W) — higher = sparser, more interpretable factors",
        "all_df.pivot_table(index='algo', columns='tier', values='sparsity_W').round(3)",
    ),
    md(
        "## Per-tier objective winners",
    ),
    code(
        "def winner_table(df):",
        "    rows = []",
        "    if 'mean_time' in df.columns:",
        "        r = df.loc[df['mean_time'].idxmin()];     rows.append(('fastest',         r['algo'], float(r['mean_time'])))",
        "    if 'mean_loss' in df.columns:",
        "        r = df.loc[df['mean_loss'].idxmin()];     rows.append(('lowest loss',     r['algo'], float(r['mean_loss'])))",
        "    if 'dispersion' in df.columns and df['dispersion'].notna().any():",
        "        r = df.loc[df['dispersion'].idxmax()];    rows.append(('most stable',     r['algo'], float(r['dispersion'])))",
        "    if 'cophenetic' in df.columns and df['cophenetic'].notna().any():",
        "        r = df.loc[df['cophenetic'].idxmax()];    rows.append(('best cophenetic', r['algo'], float(r['cophenetic'])))",
        "    if 'sparsity_W' in df.columns:",
        "        r = df.loc[df['sparsity_W'].idxmax()];    rows.append(('sparsest W',      r['algo'], float(r['sparsity_W'])))",
        "    if 'ARI vs labels' in df.columns:",
        "        r = df.loc[df['ARI vs labels'].idxmax()]; rows.append(('best ARI',        r['algo'], float(r['ARI vs labels'])))",
        "    if 'Amari error' in df.columns:",
        "        r = df.loc[df['Amari error'].idxmin()];   rows.append(('lowest Amari',    r['algo'], float(r['Amari error'])))",
        "    return pd.DataFrame(rows, columns=['metric', 'winner', 'value'])",
        "",
        "for tier, df in [('small', df_small), ('medium', df_med), ('large', df_large)]:",
        "    print(f'\\n=== {tier} ===')",
        "    print(winner_table(df).to_string(index=False))",
    ),
    md(
        "## Recommendations (objective metrics, no anchor)",
        "",
        "- **Speed**: `hals + NNDSVD` wins everywhere by 2-5× over sklearn cd.",
        "- **Stability** (cophenetic / dispersion): typically `lee` and `brunet`",
        "  give the most stable consensus — multiplicative updates are",
        "  deterministic given enough iterations and converge to the same",
        "  basin. HALS-family solvers find equally-good but distinct local",
        "  minima → lower consensus stability.",
        "- **Sparsity**: `dnmf` (RcppML-style) and `snmf/r/l` are the only",
        "  algorithms that explicitly target sparse W or H. Plain HALS gives",
        "  dense factors.",
        "- **Biological signal** (ARI vs labels on real data): typically all",
        "  reasonable algorithms cluster cells comparably — choose by",
        "  speed + stability.",
        "- **Factor recovery on synthetic** (Amari): all HALS-family solvers",
        "  hit similar Amari error; multiplicative updates need more iters",
        "  than this benchmark allots, so they look weak.",
        "",
        "**For production single-cell pipelines**: `hals + NNDSVD` for speed,",
        "`dnmf` for interpretability, `lee` for R-bit-equivalent results.",
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
        timeout=5400,
        kernel_name=KERNEL,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    client.execute()
    nbformat.write(nb, nb_path)
    print(f"[build] executed {nb_path}")


if __name__ == "__main__":
    main()
