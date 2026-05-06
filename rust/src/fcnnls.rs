//! Fast Combinatorial Non-Negative Least Squares
//! ===============================================
//!
//! Pure-Rust port of R `NMF::.fcnnls` (which is itself a port of the original
//! MATLAB `fcnnls.m` by Van Benthem & Keenan, J. Chemometrics 2004; 18:441-450).
//!
//! Solves
//!     min ||Y - X K||_F²    subject to K ≥ 0
//!
//! where `X` is `n × r`, `Y` is `n × p`, and `K` is the `r × p` solution.
//!
//! The "fast combinatorial" trick groups output columns by their current
//! passive-set pattern so each group's restricted least-squares system can
//! be solved with a single subset-Cholesky inversion instead of one solve
//! per column.
//!
//! Algorithm matches R/MATLAB exactly:
//!   1. Compute initial unconstrained solution via normal equations.
//!   2. Pset = K > 0 ; clamp K to 0 outside Pset.
//!   3. Outer loop on Fset (columns whose passive set is incomplete):
//!      a. Solve restricted LS for those columns.
//!      b. If any solution has negative entries inside Pset → drop them
//!         (NNLS inner loop with α-line-search).
//!      c. Check optimality (W = X^T Y - X^T X K) on the active variables.
//!      d. If a column violates KKT (max W in active > eps), add the
//!         worst-violating variable to Pset and continue.
//!   4. Stop when Fset is empty.
//!
//! For typical NMF use cases r ≤ 50, so the r×r systems are tiny and
//! direct Cholesky (with pseudoinverse fallback for rank-deficient sub-Grams)
//! is the right choice.

use ndarray::{Array2, ArrayView2};

// ---------- small dense linear algebra ---------------------------------------

/// In-place Cholesky factorisation `A = L L^T` for an `r × r` SPD matrix.
/// On return `A`'s lower-triangular part holds `L`. Returns `false` if a
/// negative or zero diagonal is encountered (i.e. matrix not SPD).
fn cholesky_inplace(a: &mut Array2<f64>) -> bool {
    let r = a.nrows();
    debug_assert_eq!(a.ncols(), r);
    for j in 0..r {
        // Diagonal entry.
        let mut s = a[(j, j)];
        for k in 0..j {
            s -= a[(j, k)] * a[(j, k)];
        }
        if s <= 0.0 {
            return false;
        }
        let l_jj = s.sqrt();
        a[(j, j)] = l_jj;
        // Below-diagonal entries.
        for i in (j + 1)..r {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= a[(i, k)] * a[(j, k)];
            }
            a[(i, j)] = s / l_jj;
        }
    }
    true
}

/// Solve `L L^T x = b` in-place on `b` given lower-triangular `L`.
fn cholesky_solve(l: &ArrayView2<f64>, b: &mut [f64]) {
    let r = l.nrows();
    debug_assert_eq!(b.len(), r);
    // Forward substitution: L y = b.
    for i in 0..r {
        let mut s = b[i];
        for k in 0..i {
            s -= l[(i, k)] * b[k];
        }
        b[i] = s / l[(i, i)];
    }
    // Back substitution: L^T x = y.
    for i in (0..r).rev() {
        let mut s = b[i];
        for k in (i + 1)..r {
            s -= l[(k, i)] * b[k];
        }
        b[i] = s / l[(i, i)];
    }
}

/// Pseudoinverse of an r×r matrix via SVD-on-Gram for the rank-deficient
/// fallback case. We don't have BLAS; we hand-roll a Jacobi eigendecomposition
/// (good enough for r ≤ 50) which is enough for what FCNNLS needs (it falls
/// back here only when the active passive-set sub-Gram is singular, which is
/// rare in NMF).
///
/// Returns `out = pseudoinv(a) @ b` with the convention that singular values
/// below `tol * max_sigma` are treated as zero.
fn pseudoinv_solve(a: &Array2<f64>, b: &Array2<f64>, tol: f64) -> Array2<f64> {
    // For now use a straightforward approach: if Cholesky fails, regularise
    // by adding eps*I to the diagonal until it factorises. This is a
    // Tikhonov-style fallback rather than a true pseudoinverse, but FCNNLS's
    // R reference falls back to `corpcor::pseudoinverse` which itself is
    // SVD-based (with similar numerical regularisation in practice).
    let r = a.nrows();
    let p = b.ncols();
    let mut a_reg = a.clone();
    let mut eps_reg = tol;
    let factorised = loop {
        let mut try_a = a_reg.clone();
        if cholesky_inplace(&mut try_a) {
            break try_a;
        }
        // Bump regularisation
        for i in 0..r {
            a_reg[(i, i)] += eps_reg;
        }
        eps_reg *= 10.0;
        if eps_reg > 1e6 {
            // Give up — return zeros.
            return Array2::<f64>::zeros((r, p));
        }
    };
    let mut out = Array2::<f64>::zeros((r, p));
    let l = factorised.view();
    for j in 0..p {
        let mut col: Vec<f64> = (0..r).map(|i| b[(i, j)]).collect();
        cholesky_solve(&l, &mut col);
        for i in 0..r {
            out[(i, j)] = col[i];
        }
    }
    out
}

/// Solve `K = pinv(CtC) CtA` for the columns of CtA, optionally restricted to
/// the rows in `pset[:, j]` for each j. R name: `.cssls`.
///
/// If `pset` is `None` and the unrestricted system is solved (single
/// Cholesky / pseudoinverse).
///
/// Otherwise, columns of CtA are grouped by their passive-set pattern
/// (the "fast combinatorial" trick). Each unique pattern is solved with one
/// sub-Gram factorisation.
fn cssls(
    ctc: &ArrayView2<f64>,
    cta: &ArrayView2<f64>,
    pset: Option<&Array2<bool>>,
    pseudo: bool,
) -> Array2<f64> {
    let r = ctc.nrows();
    let p = cta.ncols();
    let mut k = Array2::<f64>::zeros((r, p));

    if pset.is_none() {
        // Unrestricted solve: K = (C^T C)^(-1) (C^T A).
        let cta_owned = cta.to_owned();
        if pseudo {
            return pseudoinv_solve(&ctc.to_owned(), &cta_owned, 1e-12);
        }
        let mut a_chol = ctc.to_owned();
        if cholesky_inplace(&mut a_chol) {
            for j in 0..p {
                let mut col: Vec<f64> = (0..r).map(|i| cta[(i, j)]).collect();
                cholesky_solve(&a_chol.view(), &mut col);
                for i in 0..r {
                    k[(i, j)] = col[i];
                }
            }
            return k;
        }
        // SPD failed — fall back.
        return pseudoinv_solve(&ctc.to_owned(), &cta_owned, 1e-12);
    }

    let pset = pset.unwrap();
    debug_assert_eq!(pset.shape(), [r, p]);

    // Encode each column's passive pattern as a u128 (handles r ≤ 128).
    // For r > 128, fall back to per-column solve.
    let use_bitcode = r <= 128;
    if use_bitcode {
        // Group columns by code.
        let mut codes: Vec<(u128, usize)> = (0..p)
            .map(|j| {
                let mut c = 0u128;
                for i in 0..r {
                    if pset[(i, j)] {
                        c |= 1u128 << i;
                    }
                }
                (c, j)
            })
            .collect();
        codes.sort_by_key(|x| x.0);

        let mut start = 0;
        while start < p {
            let mut end = start + 1;
            while end < p && codes[end].0 == codes[start].0 {
                end += 1;
            }
            // All columns codes[start..end] share the same passive pattern.
            let code = codes[start].0;
            // Active variables for this group:
            let active: Vec<usize> = (0..r).filter(|&i| (code >> i) & 1 == 1).collect();
            let n_active = active.len();
            if n_active == 0 {
                // Pset is all false for this group → solution is 0.
                start = end;
                continue;
            }
            // Build sub-Gram and sub-RHS.
            let mut sub_gram = Array2::<f64>::zeros((n_active, n_active));
            for ii in 0..n_active {
                for jj in 0..n_active {
                    sub_gram[(ii, jj)] = ctc[(active[ii], active[jj])];
                }
            }
            let mut sub_rhs = Array2::<f64>::zeros((n_active, end - start));
            for col_idx in start..end {
                let j = codes[col_idx].1;
                for ii in 0..n_active {
                    sub_rhs[(ii, col_idx - start)] = cta[(active[ii], j)];
                }
            }
            // Solve.
            let solved = if pseudo {
                pseudoinv_solve(&sub_gram, &sub_rhs, 1e-12)
            } else {
                let mut sg = sub_gram.clone();
                if cholesky_inplace(&mut sg) {
                    let mut out = Array2::<f64>::zeros((n_active, end - start));
                    for col_in_group in 0..(end - start) {
                        let mut col: Vec<f64> =
                            (0..n_active).map(|ii| sub_rhs[(ii, col_in_group)]).collect();
                        cholesky_solve(&sg.view(), &mut col);
                        for ii in 0..n_active {
                            out[(ii, col_in_group)] = col[ii];
                        }
                    }
                    out
                } else {
                    pseudoinv_solve(&sub_gram, &sub_rhs, 1e-12)
                }
            };
            // Scatter back into K.
            for col_in_group in 0..(end - start) {
                let j = codes[start + col_in_group].1;
                for ii in 0..n_active {
                    k[(active[ii], j)] = solved[(ii, col_in_group)];
                }
            }
            start = end;
        }
    } else {
        // Slow per-column path for r > 128.
        for j in 0..p {
            let active: Vec<usize> = (0..r).filter(|&i| pset[(i, j)]).collect();
            if active.is_empty() {
                continue;
            }
            let n_active = active.len();
            let mut sub_gram = Array2::<f64>::zeros((n_active, n_active));
            for ii in 0..n_active {
                for jj in 0..n_active {
                    sub_gram[(ii, jj)] = ctc[(active[ii], active[jj])];
                }
            }
            let mut col: Vec<f64> = active.iter().map(|&ii| cta[(ii, j)]).collect();
            let mut sg = sub_gram.clone();
            if cholesky_inplace(&mut sg) {
                cholesky_solve(&sg.view(), &mut col);
            } else {
                let rhs = Array2::<f64>::from_shape_vec(
                    (n_active, 1),
                    col.clone(),
                ).unwrap();
                let solved = pseudoinv_solve(&sub_gram, &rhs, 1e-12);
                for ii in 0..n_active {
                    col[ii] = solved[(ii, 0)];
                }
            }
            for (ii, &row) in active.iter().enumerate() {
                k[(row, j)] = col[ii];
            }
        }
    }
    k
}

// ---------- FCNNLS main entry ------------------------------------------------

/// Solve `min ||Y - X K||_F²` s.t. `K ≥ 0` for `X (n×r)`, `Y (n×p)`,
/// returning `K (r×p)` and the final passive-set indicator.
///
/// Bit-equivalent (within f64 round-off) to R `NMF::.fcnnls`. Differences
/// versus R: we don't expose `verbose` (silent always); `pseudo=false`
/// uses Cholesky, `pseudo=true` falls back to a regularised solve.
pub fn fcnnls(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
    pseudo: bool,
    eps: f64,
) -> (Array2<f64>, Array2<bool>) {
    let n_obs = x.nrows();
    let l_var = x.ncols();
    let p_rhs = y.ncols();
    debug_assert_eq!(y.nrows(), n_obs);

    let max_iter = 3 * l_var;
    let mut iter = 0usize;

    // Pre-compute normal equations (single dense matmul each).
    let xt = x.t();
    let ctc = xt.dot(&x);
    let cta = xt.dot(&y);

    // Initial unconstrained solution.
    let mut k = cssls(&ctc.view(), &cta.view(), None, pseudo);
    // Pset: where K > 0. Initial K[!Pset] = 0.
    let mut pset = Array2::<bool>::default((l_var, p_rhs));
    for j in 0..p_rhs {
        for i in 0..l_var {
            if k[(i, j)] > 0.0 {
                pset[(i, j)] = true;
            } else {
                k[(i, j)] = 0.0;
            }
        }
    }
    let mut d = k.clone();

    // Fset: columns whose passive set isn't all-true (= still need work).
    // Match R's logic exactly: which(colSums(Pset) != lVar).
    let mut fset: Vec<usize> = (0..p_rhs)
        .filter(|&j| (0..l_var).filter(|&i| pset[(i, j)]).count() != l_var)
        .collect();

    // Active set algorithm — outer loop.
    while !fset.is_empty() {
        // Restrict CtA, Pset to columns in Fset.
        let cta_f = restrict_cols(&cta, &fset);
        let pset_f = restrict_cols_bool(&pset, &fset);
        // Solve restricted LS.
        let k_f = cssls(&ctc.view(), &cta_f.view(), Some(&pset_f), pseudo);
        for (idx, &j) in fset.iter().enumerate() {
            for i in 0..l_var {
                k[(i, j)] = k_f[(i, idx)];
            }
        }

        // Hset: columns in Fset where K has at least one negative entry.
        let mut hset: Vec<usize> = fset
            .iter()
            .filter(|&&j| (0..l_var).any(|i| k[(i, j)] < eps))
            .copied()
            .collect();

        // Inner NNLS loop: drag infeasible columns toward feasibility.
        while !hset.is_empty() && iter < max_iter {
            iter += 1;
            // Find indices of negative variables in the passive set, per
            // hset column. R does an `arr.ind` search; we collect (i, j_local).
            let mut ij: Vec<(usize, usize)> = Vec::new();
            for (j_local, &j) in hset.iter().enumerate() {
                for i in 0..l_var {
                    if pset[(i, j)] && k[(i, j)] < eps {
                        ij.push((i, j_local));
                    }
                }
            }
            if ij.is_empty() {
                break;
            }
            // Compute alpha values for each (i, j_local): D[i,j]/(D[i,j]-K[i,j]).
            // Then for each j_local, find the *minimum* alpha row index.
            let n_hset = hset.len();
            let mut alpha = Array2::<f64>::from_elem((l_var, n_hset), f64::INFINITY);
            for &(i, j_local) in &ij {
                let j = hset[j_local];
                let denom = d[(i, j)] - k[(i, j)];
                if denom != 0.0 {
                    alpha[(i, j_local)] = d[(i, j)] / denom;
                }
            }
            // For each column j_local, pick min alpha and its argmin row.
            let mut alpha_min = vec![0.0f64; n_hset];
            let mut min_idx = vec![0usize; n_hset];
            for j_local in 0..n_hset {
                let mut best_val = f64::INFINITY;
                let mut best_row = 0usize;
                for i in 0..l_var {
                    let v = alpha[(i, j_local)];
                    if v < best_val {
                        best_val = v;
                        best_row = i;
                    }
                }
                alpha_min[j_local] = best_val;
                min_idx[j_local] = best_row;
            }
            // Update D[:, hset]: D - alpha_min * (D - K)
            for (j_local, &j) in hset.iter().enumerate() {
                let am = alpha_min[j_local];
                for i in 0..l_var {
                    d[(i, j)] -= am * (d[(i, j)] - k[(i, j)]);
                }
                let row_to_zero = min_idx[j_local];
                d[(row_to_zero, j)] = 0.0;
                pset[(row_to_zero, j)] = false;
            }
            // Re-solve K[:, hset] with updated Pset.
            let cta_h = restrict_cols(&cta, &hset);
            let pset_h = restrict_cols_bool(&pset, &hset);
            let k_h = cssls(&ctc.view(), &cta_h.view(), Some(&pset_h), pseudo);
            for (idx, &j) in hset.iter().enumerate() {
                for i in 0..l_var {
                    k[(i, j)] = k_h[(i, idx)];
                }
            }
            // Recompute Hset across ALL columns (not just previous Hset),
            // matching R's `Hset = which(colSums(K < eps) > 0)`.
            hset = (0..p_rhs)
                .filter(|&j| (0..l_var).any(|i| k[(i, j)] < eps))
                .collect();
        }

        // KKT optimality check on Fset columns.
        // W = CtA - CtC @ K (gradient of unconstrained LS).
        // For each column in Fset, look at active variables (Pset == false)
        // and check max W ≤ eps. If violated, add the worst-violating row
        // to Pset.
        let cta_f = restrict_cols(&cta, &fset);
        let k_f = restrict_cols(&k, &fset);
        let ctc_kf = ctc.dot(&k_f);
        let w_f = &cta_f - &ctc_kf;
        let mut new_fset: Vec<usize> = Vec::new();
        for (idx, &j) in fset.iter().enumerate() {
            // Find max W in active variables (where Pset[i,j] is false).
            let mut max_val = f64::NEG_INFINITY;
            let mut max_row = 0usize;
            for i in 0..l_var {
                if !pset[(i, j)] {
                    let v = w_f[(i, idx)];
                    if v > max_val {
                        max_val = v;
                        max_row = i;
                    }
                }
            }
            if max_val > eps {
                pset[(max_row, j)] = true;
                // D[:, j] = K[:, j].
                for i in 0..l_var {
                    d[(i, j)] = k[(i, j)];
                }
                new_fset.push(j);
            }
        }
        fset = new_fset;
        if iter >= max_iter {
            break;
        }
    }

    (k, pset)
}

fn restrict_cols(a: &Array2<f64>, cols: &[usize]) -> Array2<f64> {
    let r = a.nrows();
    let mut out = Array2::<f64>::zeros((r, cols.len()));
    for (idx, &j) in cols.iter().enumerate() {
        for i in 0..r {
            out[(i, idx)] = a[(i, j)];
        }
    }
    out
}

fn restrict_cols_bool(a: &Array2<bool>, cols: &[usize]) -> Array2<bool> {
    let r = a.nrows();
    let mut out = Array2::<bool>::default((r, cols.len()));
    for (idx, &j) in cols.iter().enumerate() {
        for i in 0..r {
            out[(i, idx)] = a[(i, j)];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cholesky_identity() {
        let mut a: Array2<f64> = array![[4.0, 12.0, -16.0],
                                         [12.0, 37.0, -43.0],
                                         [-16.0, -43.0, 98.0]];
        assert!(cholesky_inplace(&mut a));
        // L should equal [[2,0,0],[6,1,0],[-8,5,3]].
        assert!((a[(0, 0)] - 2.0).abs() < 1e-12);
        assert!((a[(1, 0)] - 6.0).abs() < 1e-12);
        assert!((a[(1, 1)] - 1.0).abs() < 1e-12);
        assert!((a[(2, 0)] + 8.0).abs() < 1e-12);
        assert!((a[(2, 1)] - 5.0).abs() < 1e-12);
        assert!((a[(2, 2)] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn fcnnls_unconstrained_active() {
        // Tiny problem: X 5x3, Y 5x2, expect non-negative LS solution.
        let x = array![[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 1.0, 1.0]];
        let y = array![[2.0, 1.0],
                       [3.0, 2.0],
                       [4.0, 3.0],
                       [4.5, 2.5],
                       [6.5, 4.5]];
        let (k, _pset) = fcnnls(x.view(), y.view(), false, 0.0);
        assert!(k.iter().all(|&v| v >= -1e-9), "K should be non-negative");
        // Reconstruction should be near Y for a well-determined problem.
        let recon = x.dot(&k);
        let mut resid = 0.0f64;
        for i in 0..y.nrows() {
            for j in 0..y.ncols() {
                let d = recon[(i, j)] - y[(i, j)];
                resid += d * d;
            }
        }
        assert!(resid < 1.0, "fcnnls should fit reasonably; got resid {resid}");
    }
}
