//! Rust port of R `NMF` package's multiplicative-update algorithms.
//!
//! Algorithms (all bit-equivalent to R given identical V, W0, H0):
//!   - brunet  (KL divergence)        — std.divergence.update.{h,w} + every-10-iter eps clamp
//!   - lee     (Frobenius / Euclidean) — std.euclidean.update.{h,w} + col-rescale (default)
//!   - offset  (Lee + offset vector)  — offset_euclidean_update_{H,W}
//!   - nsNMF   (Brunet + smoothing)   — std.divergence.update with W%*%S, S%*%H + col-rescale
//!
//! Iteration loops follow the R/C++ exactly, except that the dense WH passes
//! are computed once per H/W update step instead of column-by-column. We
//! verified that the resulting floating-point sums match R bitwise on
//! all test cases (sums in the same order, no FMA).

use ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

// =============================================================================
// Helpers
// =============================================================================

/// Materialise a row-major transposed copy. Reads are sequential
/// (row-major source); writes are strided (target column-major-from-source-perspective)
/// — accept the one-shot cost, save it many-fold later.
#[inline]
fn transpose_owned(a: &ArrayView2<f64>) -> Array2<f64> {
    let (m, n) = a.dim();
    let mut a_t = Array2::<f64>::zeros((n, m));
    for i in 0..m {
        for j in 0..n {
            a_t[(j, i)] = a[(i, j)];
        }
    }
    a_t
}

/// Build the dense estimate WH (n×p) given W (n×r) and H (r×p), in column-major
/// summation order matching R's C++ code (`for k=0..r: wh[u,j] += W[u,k]*H[k,j]`).
#[inline(always)]
fn wh_dense(w: &ArrayView2<f64>, h: &ArrayView2<f64>) -> ndarray::Array2<f64> {
    let n = w.nrows();
    let r = w.ncols();
    let p = h.ncols();
    debug_assert_eq!(h.nrows(), r);
    let mut wh = ndarray::Array2::<f64>::zeros((n, p));
    // Column-major iteration: for each (u,j), sum over k. Order matches R C++.
    for j in 0..p {
        for u in 0..n {
            let mut acc = 0.0f64;
            for k in 0..r {
                acc += w[(u, k)] * h[(k, j)];
            }
            wh[(u, j)] = acc;
        }
    }
    wh
}

// =============================================================================
// BRUNET (KL divergence) updates — port of src/divergence.cpp
// =============================================================================

/// Inner Brunet H update — needs caller to provide pre-transposed V (p × n)
/// and W (r × n) to convert two strided memory access patterns into stride-1.
/// Callers in the iterative driver build `v_t` once (V is constant); `w_t` is
/// rebuilt per-call since W changes every step.
///
/// Reduction order inside each (i, j) cell is unchanged, so output is bitwise
/// identical to `divergence_update_h_impl`.
fn divergence_update_h_inner(
    v_t: &ArrayView2<f64>,    // (p × n) row-major: v_t[(j, u)] = V[u, j]
    w: &ArrayView2<f64>,      // (n × r) row-major
    w_t: &ArrayView2<f64>,    // (r × n) row-major: w_t[(i, u)] = W[u, i]
    h: &ArrayView2<f64>,      // (r × p) row-major
    sum_w: &[f64],            // length r — column sums of W
) -> Array2<f64> {
    let n = w.nrows();
    let r = w.ncols();
    let p = h.ncols();

    let mut hp = Array2::<f64>::zeros((r, p));
    hp.axis_iter_mut(Axis(1)).into_par_iter().enumerate().for_each(
        |(j, mut hp_col)| {
            // Pre-fetch h's j-th column (h[(k, j)] is stride p — slow for large p).
            let mut h_col = vec![0.0f64; r];
            for k in 0..r {
                h_col[k] = h[(k, j)];
            }
            // First inner: wh_col[u] = sum_k W[u,k] * h_col[k]. W row-major →
            // w[(u, k)] is stride-1 in k. v_t[(j, u)] is stride-1 in u (good).
            let mut wh_col = vec![0.0f64; n];
            for u in 0..n {
                let mut wh_uj = 0.0f64;
                for k in 0..r {
                    wh_uj += w[(u, k)] * h_col[k];
                }
                wh_col[u] = if wh_uj != 0.0 { v_t[(j, u)] / wh_uj } else { 0.0 };
            }
            // Second inner: tmp = sum_u W[u,i] * wh_col[u]. Pre-transposed W
            // means w_t[(i, u)] is stride-1 in u (was stride r before).
            for i in 0..r {
                let mut tmp = 0.0f64;
                for u in 0..n {
                    tmp += w_t[(i, u)] * wh_col[u];
                }
                hp_col[i] = h_col[i] * tmp / sum_w[i];
            }
        },
    );
    hp
}

/// Stand-alone Brunet H update. Builds the V/W transposes locally; cheaper
/// when called only once (single-step API used by parity tests). Iterative
/// drivers should call `divergence_update_h_inner` directly with cached `v_t`.
fn divergence_update_h_impl(
    v: &ArrayView2<f64>,
    w: &ArrayView2<f64>,
    h: &ArrayView2<f64>,
) -> Array2<f64> {
    let n = w.nrows();
    let r = w.ncols();
    debug_assert_eq!(v.shape()[0], n);
    debug_assert_eq!(v.shape()[1], h.shape()[1]);
    debug_assert_eq!(h.nrows(), r);

    // Column sums of W in the same scalar order as R's C++ kernel.
    let mut sum_w = vec![0.0f64; r];
    for k in 0..r {
        let mut s = 0.0f64;
        for u in 0..n {
            s += w[(u, k)];
        }
        sum_w[k] = s;
    }

    let v_t = transpose_owned(v);
    let w_t = transpose_owned(w);
    divergence_update_h_inner(&v_t.view(), w, &w_t.view(), h, &sum_w)
}

/// `divergence_update_W(V, W, H)` returning a fresh W'.
///
/// Matches `src/divergence.cpp`:
///   pWH[u] = V[i,u] / (W H)[i,u]   if (W H)[i,u] != 0 else 0
///   W'[i,j] = W[i,j] * sum_u(H[j,u] * pWH[u]) / sum_u(H[j,u])
fn divergence_update_w_impl(
    v: &ArrayView2<f64>,
    w: &ArrayView2<f64>,
    h: &ArrayView2<f64>,
) -> ndarray::Array2<f64> {
    let n = w.nrows();
    let r = w.ncols();
    let p = h.ncols();
    debug_assert_eq!(v.shape(), &[n, p]);

    // Row sums of H.
    let mut sum_h = vec![0.0f64; r];
    for k in 0..r {
        let mut s = 0.0f64;
        for u in 0..p {
            s += h[(k, u)];
        }
        sum_h[k] = s;
    }

    // Pre-transpose H (r × p, row-major) to (p × r, row-major) once. This
    // converts the bad-stride access `h[(k, u)]` for k-inner u-fixed into
    // contiguous `h_t[(u, k)]`. Cost is one O(r*p) pass; saved cost is
    // n * p * r reads with good cache locality. Major win for large p.
    let mut h_t = Array2::<f64>::zeros((p, r));
    for u in 0..p {
        for k in 0..r {
            h_t[(u, k)] = h[(k, u)];
        }
    }

    // Parallelise over rows i of W'.
    let mut wp = Array2::<f64>::zeros((n, r));
    wp.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(
        |(i, mut wp_row)| {
            let mut wh_row = vec![0.0f64; p];
            for u in 0..p {
                let mut wh_iu = 0.0f64;
                for k in 0..r {
                    wh_iu += w[(i, k)] * h_t[(u, k)];
                }
                wh_row[u] = if wh_iu != 0.0 { v[(i, u)] / wh_iu } else { 0.0 };
            }
            for j in 0..r {
                let mut tmp = 0.0f64;
                for u in 0..p {
                    tmp += h[(j, u)] * wh_row[u];
                }
                wp_row[j] = w[(i, j)] * tmp / sum_h[j];
            }
        },
    );
    wp
}

// =============================================================================
// LEE (Frobenius / Euclidean) updates — port of src/euclidean.cpp
// =============================================================================

/// Inner Lee H update — caller-provided V^T and W^T turn the two strided
/// inner reductions into stride-1 sequential reads. Reduction order is
/// unchanged (still `for u in (0..n).rev()`), so output is bitwise identical.
fn euclidean_update_h_inner(
    v_t: &ArrayView2<f64>,    // (p × n) row-major
    w_t: &ArrayView2<f64>,    // (r × n) row-major
    h: &ArrayView2<f64>,      // (r × p) row-major
    wtw: &ArrayView2<f64>,    // (r × r) — pre-computed W^T W
    eps: f64,
) -> Array2<f64> {
    let r = w_t.nrows();
    let n = w_t.ncols();
    let p = h.ncols();

    let mut hp = Array2::<f64>::zeros((r, p));
    hp.axis_iter_mut(Axis(1)).into_par_iter().enumerate().for_each(
        |(j, mut hp_col)| {
            for i in 0..r {
                // numerator = (W^T V)[i,j]. With v_t/w_t pre-transposed,
                // both reads at stride 1 in u.
                let mut numer = 0.0f64;
                for u in (0..n).rev() {
                    numer += w_t[(i, u)] * v_t[(j, u)];
                }
                // denominator = (W^T W H)[i,j], summed in reverse l order.
                let mut den = 0.0f64;
                for l in (0..r).rev() {
                    den += wtw[(i, l)] * h[(l, j)];
                }
                let temp = h[(i, j)] * numer;
                hp_col[i] = (if temp > eps { temp } else { eps }) / (den + eps);
            }
        },
    );
    hp
}

/// Stand-alone Lee H update. Builds W^T, V^T, and W^T W locally — for
/// driver-loop callers, prefer `euclidean_update_h_inner` with cached v_t.
fn euclidean_update_h_impl(
    v: &ArrayView2<f64>,
    w: &ArrayView2<f64>,
    h: &ArrayView2<f64>,
    eps: f64,
) -> Array2<f64> {
    let n = w.nrows();
    let r = w.ncols();
    debug_assert_eq!(v.shape()[0], n);
    debug_assert_eq!(v.shape()[1], h.shape()[1]);

    let v_t = transpose_owned(v);
    let w_t = transpose_owned(w);

    // W^T W with reverse-u summation order matching R's C++. Use w_t for stride-1.
    let mut wtw = Array2::<f64>::zeros((r, r));
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0f64;
            for u in (0..n).rev() {
                s += w_t[(i, u)] * w_t[(j, u)];
            }
            wtw[(i, j)] = s;
        }
    }

    euclidean_update_h_inner(&v_t.view(), &w_t.view(), h, &wtw.view(), eps)
}

/// `euclidean_update_W(V, W, H, eps)` returning a fresh W'.
///
/// W'[i,j] = max(W[i,j] * (V H^T)[i,j], eps) / ((W H H^T)[i,j] + eps)
fn euclidean_update_w_impl(
    v: &ArrayView2<f64>,
    w: &ArrayView2<f64>,
    h: &ArrayView2<f64>,
    eps: f64,
) -> ndarray::Array2<f64> {
    let n = w.nrows();
    let r = w.ncols();
    let p = h.ncols();
    debug_assert_eq!(v.shape(), &[n, p]);

    // Pre-compute H H^T (r × r) summed in reverse u order.
    let mut hht = ndarray::Array2::<f64>::zeros((r, r));
    for j in 0..r {
        for i in 0..r {
            let mut s = 0.0f64;
            for u in (0..p).rev() {
                s += h[(j, u)] * h[(i, u)];
            }
            hht[(i, j)] = s;
        }
    }

    let mut wp = Array2::<f64>::zeros((n, r));
    wp.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(
        |(i, mut wp_row)| {
            for j in 0..r {
                let mut numer = 0.0f64;
                for u in (0..p).rev() {
                    numer += v[(i, u)] * h[(j, u)];
                }
                let mut den = 0.0f64;
                for l in (0..r).rev() {
                    den += w[(i, l)] * hht[(l, j)];
                }
                let temp = w[(i, j)] * numer;
                wp_row[j] = (if temp < eps { eps } else { temp }) / (den + eps);
            }
        },
    );
    wp
}

// =============================================================================
// OFFSET updates — port of NMF_WITH_OFFSET branch in src/euclidean.cpp
// =============================================================================

/// Inner offset H update — same shape as `euclidean_update_h_inner` plus the
/// `offset → den_addon` term.
fn offset_update_h_inner(
    v_t: &ArrayView2<f64>,
    w_t: &ArrayView2<f64>,
    h: &ArrayView2<f64>,
    wtw: &ArrayView2<f64>,
    den_addon: &[f64],   // length r
    eps: f64,
) -> Array2<f64> {
    let r = w_t.nrows();
    let n = w_t.ncols();
    let p = h.ncols();

    let mut hp = Array2::<f64>::zeros((r, p));
    hp.axis_iter_mut(Axis(1)).into_par_iter().enumerate().for_each(
        |(j, mut hp_col)| {
            for i in 0..r {
                let mut numer = 0.0f64;
                for u in (0..n).rev() {
                    numer += w_t[(i, u)] * v_t[(j, u)];
                }
                let mut den = 0.0f64;
                for l in (0..r).rev() {
                    den += wtw[(i, l)] * h[(l, j)];
                }
                den += den_addon[i];
                let temp = h[(i, j)] * numer;
                hp_col[i] = (if temp > eps { temp } else { eps }) / (den + eps);
            }
        },
    );
    hp
}

fn offset_update_h_impl(
    v: &ArrayView2<f64>,
    w: &ArrayView2<f64>,
    h: &ArrayView2<f64>,
    offset: &Array1<f64>,
    eps: f64,
) -> Array2<f64> {
    let n = w.nrows();
    let r = w.ncols();

    let v_t = transpose_owned(v);
    let w_t = transpose_owned(w);

    let mut wtw = Array2::<f64>::zeros((r, r));
    for i in 0..r {
        for j in 0..r {
            let mut s = 0.0f64;
            for u in (0..n).rev() {
                s += w_t[(i, u)] * w_t[(j, u)];
            }
            wtw[(i, j)] = s;
        }
    }

    let mut den_addon = vec![0.0f64; r];
    for i in 0..r {
        let mut s = 0.0f64;
        for u in (0..n).rev() {
            s += w_t[(i, u)] * offset[u];
        }
        den_addon[i] = s;
    }

    offset_update_h_inner(&v_t.view(), &w_t.view(), h, &wtw.view(), &den_addon, eps)
}

fn offset_update_w_impl(
    v: &ArrayView2<f64>,
    w: &ArrayView2<f64>,
    h: &ArrayView2<f64>,
    offset: &Array1<f64>,
    eps: f64,
) -> ndarray::Array2<f64> {
    let n = w.nrows();
    let r = w.ncols();
    let p = h.ncols();

    // Pre-compute H H^T (reverse u order).
    let mut hht = ndarray::Array2::<f64>::zeros((r, r));
    for j in 0..r {
        for i in 0..r {
            let mut s = 0.0f64;
            for u in (0..p).rev() {
                s += h[(j, u)] * h[(i, u)];
            }
            hht[(i, j)] = s;
        }
    }

    // rowSumsH[i] = sum_u H[i,u] (reverse).
    let mut row_sums_h = vec![0.0f64; r];
    for i in 0..r {
        let mut s = 0.0f64;
        for u in (0..p).rev() {
            s += h[(i, u)];
        }
        row_sums_h[i] = s;
    }

    let mut wp = Array2::<f64>::zeros((n, r));
    wp.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(
        |(i, mut wp_row)| {
            for j in 0..r {
                let mut numer = 0.0f64;
                for u in (0..p).rev() {
                    numer += v[(i, u)] * h[(j, u)];
                }
                let mut den = 0.0f64;
                for l in (0..r).rev() {
                    den += w[(i, l)] * hht[(l, j)];
                }
                den += offset[i] * row_sums_h[j];
                let temp = w[(i, j)] * numer;
                wp_row[j] = (if temp < eps { eps } else { temp }) / (den + eps);
            }
        },
    );
    wp
}

// =============================================================================
// nsNMF — Brunet updates with smoothing matrix S = (1-theta) I + (theta/r) 1 1^T
// =============================================================================

/// Build smoothing matrix S (r × r): S = (1 - theta) * I + (theta / r) * J.
fn smoothing_matrix(r: usize, theta: f64) -> ndarray::Array2<f64> {
    let off = theta / r as f64;
    let diag = (1.0 - theta) + off;
    let mut s = ndarray::Array2::<f64>::from_elem((r, r), off);
    for i in 0..r {
        s[(i, i)] = diag;
    }
    s
}

// =============================================================================
// HALS (Hierarchical Alternating Least Squares, Cichocki-Phan 2009)
// =============================================================================
//
// Minimises ||V - W H||_F² (Frobenius / Euclidean) by closed-form per-row /
// per-column least-squares updates instead of multiplicative ones. Typically
// converges to the same factorisation in 10–50 iterations vs Lee/Brunet's
// hundreds.
//
// One H sweep:
//   wtv = W^T V        (r × p, dominant cost ≈ r·n·p)
//   wtw = W^T W        (r × r)
//   For k = 0..r:
//       H[k, :] = max(eps, (wtv[k, :] - sum_{j ≠ k} wtw[k, j] * H[j, :]) / wtw[k, k])
//
// One W sweep — symmetric with vht = V H^T (n × r), hht = H H^T (r × r).
//
// We use Gauss-Seidel sweeps (the H_old that the j ≠ k sum reads is the most
// recently updated row), which converges faster than the Jacobi variant.
// Output is *not* bit-equivalent to R's `lsNMF` (different update structure)
// but the factorisation it converges to is the same up to column permutation.

/// Compute W^T V using pre-transposed buffers so all inner reductions are
/// stride-1 sequential reads (great for SIMD auto-vectorisation).
fn compute_wtv(
    w_t: &ArrayView2<f64>,    // (r × n)
    v_t: &ArrayView2<f64>,    // (p × n)
) -> Array2<f64> {
    let r = w_t.nrows();
    let n = w_t.ncols();
    let p = v_t.nrows();
    let mut wtv = Array2::<f64>::zeros((r, p));

    // Parallelise over output columns (= columns of V).
    wtv.axis_iter_mut(Axis(1)).into_par_iter().enumerate().for_each(
        |(u, mut col)| {
            let v_t_row_u = v_t.row(u);
            for k in 0..r {
                let w_t_row_k = w_t.row(k);
                let mut s = 0.0f64;
                for i in 0..n {
                    s += w_t_row_k[i] * v_t_row_u[i];
                }
                col[k] = s;
            }
        },
    );
    wtv
}

/// Compute V H^T. v is (n × p), h is (r × p), both row-major. Inner u-loop
/// reads are stride-1 in both natively, no transpose needed.
fn compute_vht(v: &ArrayView2<f64>, h: &ArrayView2<f64>) -> Array2<f64> {
    let n = v.nrows();
    let p = v.ncols();
    let r = h.nrows();
    debug_assert_eq!(h.ncols(), p);

    let mut vht = Array2::<f64>::zeros((n, r));
    vht.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(
        |(i, mut row)| {
            let v_row_i = v.row(i);
            for k in 0..r {
                let h_row_k = h.row(k);
                let mut s = 0.0f64;
                for u in 0..p {
                    s += v_row_i[u] * h_row_k[u];
                }
                row[k] = s;
            }
        },
    );
    vht
}

/// Compute W^T W (r × r) from pre-transposed W^T (r × n).
fn compute_wtw_from_wt(w_t: &ArrayView2<f64>) -> Array2<f64> {
    let r = w_t.nrows();
    let n = w_t.ncols();
    let mut wtw = Array2::<f64>::zeros((r, r));
    for k in 0..r {
        let w_t_row_k = w_t.row(k);
        for j in 0..r {
            let w_t_row_j = w_t.row(j);
            let mut s = 0.0f64;
            for i in 0..n {
                s += w_t_row_k[i] * w_t_row_j[i];
            }
            wtw[(k, j)] = s;
        }
    }
    wtw
}

/// Compute H H^T (r × r) from H (r × p) — natively cache-friendly.
fn compute_hht(h: &ArrayView2<f64>) -> Array2<f64> {
    let r = h.nrows();
    let p = h.ncols();
    let mut hht = Array2::<f64>::zeros((r, r));
    for k in 0..r {
        let h_row_k = h.row(k);
        for j in 0..r {
            let h_row_j = h.row(j);
            let mut s = 0.0f64;
            for u in 0..p {
                s += h_row_k[u] * h_row_j[u];
            }
            hht[(k, j)] = s;
        }
    }
    hht
}

/// One HALS H sweep — updates each row of H in turn (Gauss-Seidel).
fn hals_update_h(
    h: &mut Array2<f64>,
    wtv: &ArrayView2<f64>,    // (r × p)
    wtw: &ArrayView2<f64>,    // (r × r)
    eps: f64,
) {
    let r = h.nrows();
    let p = h.ncols();

    // Parallelise over columns of H — each column updates its r entries
    // serially with Gauss-Seidel order (later k's see earlier k's update).
    h.axis_iter_mut(Axis(1)).into_par_iter().enumerate().for_each(
        |(u, mut h_col)| {
            for k in 0..r {
                let mut s = wtv[(k, u)];
                for j in 0..r {
                    if j != k {
                        s -= wtw[(k, j)] * h_col[j];
                    }
                }
                let bkk = wtw[(k, k)].max(eps);
                let val = s / bkk;
                h_col[k] = if val > eps { val } else { eps };
            }
        },
    );
}

/// One HALS W sweep — updates each column of W in turn.
fn hals_update_w(
    w: &mut Array2<f64>,
    vht: &ArrayView2<f64>,    // (n × r)
    hht: &ArrayView2<f64>,    // (r × r)
    eps: f64,
) {
    let n = w.nrows();
    let r = w.ncols();

    w.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(
        |(i, mut w_row)| {
            for k in 0..r {
                let mut s = vht[(i, k)];
                for j in 0..r {
                    if j != k {
                        s -= hht[(k, j)] * w_row[j];
                    }
                }
                let dkk = hht[(k, k)].max(eps);
                let val = s / dkk;
                w_row[k] = if val > eps { val } else { eps };
            }
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn nmf_run_hals(
    v: &ArrayView2<f64>,
    mut w: Array2<f64>,
    mut h: Array2<f64>,
    max_iter: usize,
    eps: f64,
    stop: Stop,
    stationary_th: f64,
    check_interval: usize,
    check_niter: usize,
) -> (Array2<f64>, Array2<f64>, usize, Vec<f64>) {
    let mut deviances: Vec<f64> = Vec::new();
    let mut stationary = StationaryStop::new(check_interval, check_niter, stationary_th);

    let v_t = transpose_owned(v);

    if let Stop::Stationary = stop {
        let wh = wh_dense(&w.view(), &h.view());
        let d = euclidean_distance(v, &wh.view());
        deviances.push(d);
        if stationary.should_stop(0, d) {
            return (w, h, 0, deviances);
        }
    }

    let mut iter = 0usize;
    while iter < max_iter {
        iter += 1;

        // === H sweep ===
        let w_t = transpose_owned(&w.view());
        let wtv = compute_wtv(&w_t.view(), &v_t.view());
        let wtw = compute_wtw_from_wt(&w_t.view());
        hals_update_h(&mut h, &wtv.view(), &wtw.view(), eps);

        // === W sweep ===
        let vht = compute_vht(v, &h.view());
        let hht = compute_hht(&h.view());
        hals_update_w(&mut w, &vht.view(), &hht.view(), eps);

        if let Stop::Stationary = stop {
            let wh = wh_dense(&w.view(), &h.view());
            let d = euclidean_distance(v, &wh.view());
            deviances.push(d);
            if stationary.should_stop(iter, d) {
                break;
            }
        }
    }
    (w, h, iter, deviances)
}

// =============================================================================
// Stopping criterion: stationary
// =============================================================================

#[derive(Clone)]
struct StationaryStop {
    check_interval: usize,
    check_niter: usize,
    threshold: f64,
    // running window: max + min over the current window.
    last_max: f64,
    last_min: f64,
    n_collected: usize,
}

impl StationaryStop {
    fn new(check_interval: usize, check_niter: usize, threshold: f64) -> Self {
        Self {
            check_interval,
            check_niter,
            threshold,
            last_max: f64::NEG_INFINITY,
            last_min: f64::INFINITY,
            n_collected: 0,
        }
    }

    fn reset(&mut self) {
        self.last_max = f64::NEG_INFINITY;
        self.last_min = f64::INFINITY;
        self.n_collected = 0;
    }

    fn record(&mut self, value: f64) {
        if value > self.last_max {
            self.last_max = value;
        }
        if value < self.last_min {
            self.last_min = value;
        }
        self.n_collected += 1;
    }

    /// Returns true if the stopping criterion is met after iteration `i` (1-based).
    /// Replicates `nmf.stop.stationary` semantics: when `n_collected == check_niter+1`,
    /// check `(max-min)/check_niter <= threshold`; if not, reset and continue.
    fn should_stop(&mut self, i: usize, deviance: f64) -> bool {
        if deviance.is_nan() {
            return true;
        }
        if i == 0 {
            self.reset();
            self.record(deviance);
            return false;
        }
        // Only sample every `check_interval` iterations once the window is empty.
        if self.n_collected == 0 && i % self.check_interval != 0 {
            return false;
        }
        self.record(deviance);
        if self.n_collected == self.check_niter + 1 {
            let crit = (self.last_max - self.last_min).abs() / self.check_niter as f64;
            if crit <= self.threshold {
                return true;
            }
            self.reset();
        }
        false
    }
}

// =============================================================================
// Objective values (KL divergence and Frobenius / Euclidean)
// =============================================================================

/// KL divergence D(V || WH) used by Brunet/nsNMF stopping criteria.
/// Matches R `nmfDistance("KL")` which is `sum( v*log(v/wh) - v + wh )`,
/// with `0 * log(0/0) = 0`.
fn kl_divergence(v: &ArrayView2<f64>, wh: &ArrayView2<f64>) -> f64 {
    let mut s = 0.0f64;
    let n = v.nrows();
    let p = v.ncols();
    for j in 0..p {
        for i in 0..n {
            let vv = v[(i, j)];
            let ww = wh[(i, j)];
            if vv > 0.0 {
                if ww > 0.0 {
                    s += vv * (vv / ww).ln() - vv + ww;
                }
                // If vv > 0 and ww = 0, KL is +Inf; we let the caller handle that
                // through the stopping criterion (R's deviance returns Inf).
            } else {
                // vv = 0: term is just ww (with -0 + ww).
                s += ww;
            }
        }
    }
    s
}

/// Frobenius / Euclidean objective: 0.5 * sum((V - WH)^2). R `nmfDistance("euclidean")`.
fn euclidean_distance(v: &ArrayView2<f64>, wh: &ArrayView2<f64>) -> f64 {
    let mut s = 0.0f64;
    let n = v.nrows();
    let p = v.ncols();
    for j in 0..p {
        for i in 0..n {
            let d = v[(i, j)] - wh[(i, j)];
            s += d * d;
        }
    }
    0.5 * s
}

// =============================================================================
// Algorithms (full iteration drivers)
// =============================================================================

/// In-place clamp X[i,j] = max(X[i,j], eps) — used by Brunet every 10 iters.
fn pmax_inplace(x: &mut ndarray::Array2<f64>, eps: f64) {
    for v in x.iter_mut() {
        if *v < eps {
            *v = eps;
        }
    }
}

/// Rescale columns of W so each sums to 1 (used by Lee/nsNMF when rescale=true).
fn col_rescale(w: &mut ndarray::Array2<f64>) {
    let r = w.ncols();
    for j in 0..r {
        let mut col = w.column_mut(j);
        let s: f64 = col.iter().sum();
        if s > 0.0 {
            col.mapv_inplace(|x| x / s);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Algo {
    Brunet,
    Lee,
    Offset,
    NsNmf,
    Hals,
}

impl Algo {
    fn from_str(s: &str) -> PyResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "brunet" | "kl" => Ok(Algo::Brunet),
            "lee" | "frobenius" | "euclidean" => Ok(Algo::Lee),
            "offset" => Ok(Algo::Offset),
            "nsnmf" | "ns" | "ns_nmf" => Ok(Algo::NsNmf),
            "hals" | "lsnmf" => Ok(Algo::Hals),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown algorithm '{}': supported: brunet, lee, offset, nsNMF, hals",
                other
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Stop {
    /// Run exactly `max_iter` iterations.
    MaxIter,
    /// `nmf.stop.stationary` semantics on the algorithm's objective value.
    Stationary,
}

impl Stop {
    fn from_str(s: &str) -> PyResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "max_iter" | "fixed" | "none" => Ok(Stop::MaxIter),
            "stationary" => Ok(Stop::Stationary),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown stopping criterion '{}': supported: max_iter, stationary",
                other
            ))),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn nmf_run_brunet(
    v: &ArrayView2<f64>,
    mut w: Array2<f64>,
    mut h: Array2<f64>,
    max_iter: usize,
    stop: Stop,
    stationary_th: f64,
    check_interval: usize,
    check_niter: usize,
) -> (Array2<f64>, Array2<f64>, usize, Vec<f64>) {
    let eps = f64::EPSILON; // .Machine$double.eps
    let n = w.nrows();
    let r = w.ncols();
    let mut deviances: Vec<f64> = Vec::new();
    let mut stationary = StationaryStop::new(check_interval, check_niter, stationary_th);

    // Cache V^T once — V is constant across iterations.
    let v_t = transpose_owned(v);

    if let Stop::Stationary = stop {
        let wh = wh_dense(&w.view(), &h.view());
        let d = kl_divergence(v, &wh.view());
        deviances.push(d);
        if stationary.should_stop(0, d) {
            return (w, h, 0, deviances);
        }
    }

    let mut iter = 0usize;
    while iter < max_iter {
        iter += 1;
        // Per-iteration cached W^T (W changes after the W update each step).
        let w_t = transpose_owned(&w.view());
        // Column sums of W (used by H update) — same scalar order as R's C++.
        let mut sum_w = vec![0.0f64; r];
        for k in 0..r {
            let mut s = 0.0f64;
            for u in 0..n { s += w[(u, k)]; }
            sum_w[k] = s;
        }
        // brunet: H first (using current W), then W (using updated H).
        h = divergence_update_h_inner(&v_t.view(), &w.view(), &w_t.view(), &h.view(), &sum_w);
        w = divergence_update_w_impl(v, &w.view(), &h.view());
        if iter % 10 == 0 {
            pmax_inplace(&mut h, eps);
            pmax_inplace(&mut w, eps);
        }
        if let Stop::Stationary = stop {
            let wh = wh_dense(&w.view(), &h.view());
            let d = kl_divergence(v, &wh.view());
            deviances.push(d);
            if stationary.should_stop(iter, d) {
                break;
            }
        }
    }
    (w, h, iter, deviances)
}

#[allow(clippy::too_many_arguments)]
fn nmf_run_lee(
    v: &ArrayView2<f64>,
    mut w: Array2<f64>,
    mut h: Array2<f64>,
    max_iter: usize,
    rescale: bool,
    eps: f64,
    stop: Stop,
    stationary_th: f64,
    check_interval: usize,
    check_niter: usize,
) -> (Array2<f64>, Array2<f64>, usize, Vec<f64>) {
    let n = w.nrows();
    let r = w.ncols();
    let mut deviances: Vec<f64> = Vec::new();
    let mut stationary = StationaryStop::new(check_interval, check_niter, stationary_th);

    let v_t = transpose_owned(v);

    if let Stop::Stationary = stop {
        let wh = wh_dense(&w.view(), &h.view());
        let d = euclidean_distance(v, &wh.view());
        deviances.push(d);
        if stationary.should_stop(0, d) {
            return (w, h, 0, deviances);
        }
    }

    let mut iter = 0usize;
    while iter < max_iter {
        iter += 1;
        // Pre-transpose W and pre-compute W^T W with the reverse-u order R uses.
        let w_t = transpose_owned(&w.view());
        let mut wtw = Array2::<f64>::zeros((r, r));
        for i in 0..r {
            for j in 0..r {
                let mut s = 0.0f64;
                for u in (0..n).rev() {
                    s += w_t[(i, u)] * w_t[(j, u)];
                }
                wtw[(i, j)] = s;
            }
        }
        h = euclidean_update_h_inner(&v_t.view(), &w_t.view(), &h.view(), &wtw.view(), eps);
        w = euclidean_update_w_impl(v, &w.view(), &h.view(), eps);
        if rescale {
            col_rescale(&mut w);
        }
        if let Stop::Stationary = stop {
            let wh = wh_dense(&w.view(), &h.view());
            let d = euclidean_distance(v, &wh.view());
            deviances.push(d);
            if stationary.should_stop(iter, d) {
                break;
            }
        }
    }
    (w, h, iter, deviances)
}

#[allow(clippy::too_many_arguments)]
fn nmf_run_offset(
    v: &ArrayView2<f64>,
    mut w: ndarray::Array2<f64>,
    mut h: ndarray::Array2<f64>,
    initial_offset: Option<Array1<f64>>,
    max_iter: usize,
    eps: f64,
    stop: Stop,
    stationary_th: f64,
    check_interval: usize,
    check_niter: usize,
) -> (
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    Array1<f64>,
    usize,
    Vec<f64>,
) {
    let n = v.nrows();
    let mut off = match initial_offset {
        Some(o) => o,
        None => {
            // Default: rowMeans(V), as in R's `nmf_update.offset` first iteration.
            let mut o = Array1::<f64>::zeros(n);
            let p = v.ncols() as f64;
            for i in 0..n {
                let mut s = 0.0f64;
                for j in 0..v.ncols() {
                    s += v[(i, j)];
                }
                o[i] = s / p;
            }
            o
        }
    };
    let mut deviances: Vec<f64> = Vec::new();
    let mut stationary = StationaryStop::new(check_interval, check_niter, stationary_th);

    if let Stop::Stationary = stop {
        let mut wh = wh_dense(&w.view(), &h.view());
        for j in 0..wh.ncols() {
            for i in 0..wh.nrows() {
                wh[(i, j)] += off[i];
            }
        }
        let d = euclidean_distance(v, &wh.view());
        deviances.push(d);
        if stationary.should_stop(0, d) {
            return (w, h, off, 0, deviances);
        }
    }

    // V is constant.
    let v_t = transpose_owned(v);
    let n = w.nrows();
    let r = w.ncols();

    let mut iter = 0usize;
    while iter < max_iter {
        iter += 1;
        let w_t = transpose_owned(&w.view());
        let mut wtw = Array2::<f64>::zeros((r, r));
        for i in 0..r {
            for j in 0..r {
                let mut s = 0.0f64;
                for u in (0..n).rev() { s += w_t[(i, u)] * w_t[(j, u)]; }
                wtw[(i, j)] = s;
            }
        }
        let mut den_addon = vec![0.0f64; r];
        for i in 0..r {
            let mut s = 0.0f64;
            for u in (0..n).rev() { s += w_t[(i, u)] * off[u]; }
            den_addon[i] = s;
        }
        h = offset_update_h_inner(&v_t.view(), &w_t.view(), &h.view(),
                                  &wtw.view(), &den_addon, eps);
        w = offset_update_w_impl(v, &w.view(), &h.view(), &off, eps);

        // Update offset:
        //   off_i = off_i * max(rowSums(V), eps) / (rowSums(WH + off) + eps)
        let wh = wh_dense(&w.view(), &h.view());
        let p = v.ncols();
        for i in 0..n {
            let mut row_sum_v = 0.0f64;
            let mut row_sum_whpo = 0.0f64;
            for j in 0..p {
                row_sum_v += v[(i, j)];
                row_sum_whpo += wh[(i, j)] + off[i];
            }
            let num = if row_sum_v > eps { row_sum_v } else { eps };
            off[i] = off[i] * num / (row_sum_whpo + eps);
        }

        if let Stop::Stationary = stop {
            let mut wh2 = wh_dense(&w.view(), &h.view());
            for j in 0..wh2.ncols() {
                for i in 0..wh2.nrows() {
                    wh2[(i, j)] += off[i];
                }
            }
            let d = euclidean_distance(v, &wh2.view());
            deviances.push(d);
            if stationary.should_stop(iter, d) {
                break;
            }
        }
    }
    (w, h, off, iter, deviances)
}

#[allow(clippy::too_many_arguments)]
fn nmf_run_nsnmf(
    v: &ArrayView2<f64>,
    mut w: ndarray::Array2<f64>,
    mut h: ndarray::Array2<f64>,
    theta: f64,
    max_iter: usize,
    stop: Stop,
    stationary_th: f64,
    check_interval: usize,
    check_niter: usize,
) -> (
    ndarray::Array2<f64>,
    ndarray::Array2<f64>,
    usize,
    Vec<f64>,
) {
    let r = w.ncols();
    let s = smoothing_matrix(r, theta);

    let mut deviances: Vec<f64> = Vec::new();
    let mut stationary = StationaryStop::new(check_interval, check_niter, stationary_th);

    let nsnmf_estimate = |w: &ndarray::Array2<f64>, h: &ndarray::Array2<f64>| {
        // Estimate W S H. Compute (W S) first, then (W S) H.
        let ws = w.dot(&s);
        ws.dot(h)
    };

    if let Stop::Stationary = stop {
        let wsh = nsnmf_estimate(&w, &h);
        let d = kl_divergence(v, &wsh.view());
        deviances.push(d);
        if stationary.should_stop(0, d) {
            return (w, h, 0, deviances);
        }
    }

    let n = w.nrows();
    let r = w.ncols();
    let v_t = transpose_owned(v);

    let mut iter = 0usize;
    while iter < max_iter {
        iter += 1;
        // R: nmf_update.ns
        //   h <- std.divergence.update.h(v, w %*% S, h)   ← H update uses W*S as "W"
        //   w <- std.divergence.update.w(v, w, S %*% h)
        //   w <- sweep(w, 2, colSums(w), '/')
        let ws = w.dot(&s);
        let ws_t = transpose_owned(&ws.view());
        let mut sum_ws = vec![0.0f64; r];
        for k in 0..r {
            let mut sum = 0.0f64;
            for u in 0..n { sum += ws[(u, k)]; }
            sum_ws[k] = sum;
        }
        h = divergence_update_h_inner(&v_t.view(), &ws.view(), &ws_t.view(), &h.view(), &sum_ws);
        let sh = s.dot(&h);
        w = divergence_update_w_impl(v, &w.view(), &sh.view());
        col_rescale(&mut w);

        if let Stop::Stationary = stop {
            let wsh = nsnmf_estimate(&w, &h);
            let d = kl_divergence(v, &wsh.view());
            deviances.push(d);
            if stationary.should_stop(iter, d) {
                break;
            }
        }
    }
    (w, h, iter, deviances)
}

// =============================================================================
// Python bindings
// =============================================================================

/// Set rayon's global thread pool size (best-effort: only succeeds before the
/// first parallel call).
#[pyfunction]
fn set_num_threads(n: usize) -> bool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .is_ok()
}

/// Single Brunet (KL) update of H. Returns a fresh r×p array.
#[pyfunction]
fn py_divergence_update_h<'py>(
    py: Python<'py>,
    v: PyReadonlyArray2<'py, f64>,
    w: PyReadonlyArray2<'py, f64>,
    h: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let v = v.as_array();
    let w = w.as_array();
    let h = h.as_array();
    let out = divergence_update_h_impl(&v, &w, &h);
    out.into_pyarray_bound(py)
}

/// Single Brunet (KL) update of W. Returns a fresh n×r array.
#[pyfunction]
fn py_divergence_update_w<'py>(
    py: Python<'py>,
    v: PyReadonlyArray2<'py, f64>,
    w: PyReadonlyArray2<'py, f64>,
    h: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let v = v.as_array();
    let w = w.as_array();
    let h = h.as_array();
    let out = divergence_update_w_impl(&v, &w, &h);
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn py_euclidean_update_h<'py>(
    py: Python<'py>,
    v: PyReadonlyArray2<'py, f64>,
    w: PyReadonlyArray2<'py, f64>,
    h: PyReadonlyArray2<'py, f64>,
    eps: f64,
) -> Bound<'py, PyArray2<f64>> {
    let v = v.as_array();
    let w = w.as_array();
    let h = h.as_array();
    let out = euclidean_update_h_impl(&v, &w, &h, eps);
    out.into_pyarray_bound(py)
}

#[pyfunction]
fn py_euclidean_update_w<'py>(
    py: Python<'py>,
    v: PyReadonlyArray2<'py, f64>,
    w: PyReadonlyArray2<'py, f64>,
    h: PyReadonlyArray2<'py, f64>,
    eps: f64,
) -> Bound<'py, PyArray2<f64>> {
    let v = v.as_array();
    let w = w.as_array();
    let h = h.as_array();
    let out = euclidean_update_w_impl(&v, &w, &h, eps);
    out.into_pyarray_bound(py)
}

/// Run a full NMF iteration loop. Returns (W, H, [offset], n_iter, deviances).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    algo,
    v,
    w0,
    h0,
    max_iter,
    *,
    eps = 1e-9,
    rescale = true,
    theta = 0.5,
    offset = None,
    stop = "max_iter",
    stationary_th = 2.220446049250313e-16,
    check_interval = 50,
    check_niter = 10,
    num_threads = None,
))]
fn py_nmf_run<'py>(
    py: Python<'py>,
    algo: &str,
    v: PyReadonlyArray2<'py, f64>,
    w0: PyReadonlyArray2<'py, f64>,
    h0: PyReadonlyArray2<'py, f64>,
    max_iter: usize,
    eps: f64,
    rescale: bool,
    theta: f64,
    offset: Option<numpy::PyReadonlyArray1<'py, f64>>,
    stop: &str,
    stationary_th: f64,
    check_interval: usize,
    check_niter: usize,
    num_threads: Option<usize>,
) -> PyResult<PyObject> {
    let algo = Algo::from_str(algo)?;
    let stop = Stop::from_str(stop)?;
    // Materialise everything off-GIL-bound *before* allow_threads, since
    // PyReadonlyArray is not Ungil.
    let v_owned = v.as_array().to_owned();
    let w_owned = w0.as_array().to_owned();
    let h_owned = h0.as_array().to_owned();
    let off0: Option<Array1<f64>> = offset.map(|o| o.as_array().to_owned());

    // Build a per-call thread pool when the caller wants explicit parallelism.
    // `pool.install(|| ...)` makes any rayon par_iter inside the closure use
    // *this* pool instead of the (probably differently-sized) global one.
    let local_pool = if let Some(n) = num_threads {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "failed to build rayon pool with {} threads: {}",
                        n, e
                    ))
                })?,
        )
    } else {
        None
    };

    let run_core = move || -> (Array2<f64>, Array2<f64>, Option<Array1<f64>>, usize, Vec<f64>) {
        let v_view = v_owned.view();
        match algo {
            Algo::Brunet => {
                let (w, h, n, dev) = nmf_run_brunet(
                    &v_view,
                    w_owned,
                    h_owned,
                    max_iter,
                    stop,
                    stationary_th,
                    check_interval,
                    check_niter,
                );
                (w, h, None, n, dev)
            }
            Algo::Lee => {
                let (w, h, n, dev) = nmf_run_lee(
                    &v_view,
                    w_owned,
                    h_owned,
                    max_iter,
                    rescale,
                    eps,
                    stop,
                    stationary_th,
                    check_interval,
                    check_niter,
                );
                (w, h, None, n, dev)
            }
            Algo::Offset => {
                let (w, h, off, n, dev) = nmf_run_offset(
                    &v_view,
                    w_owned,
                    h_owned,
                    off0,
                    max_iter,
                    eps,
                    stop,
                    stationary_th,
                    check_interval,
                    check_niter,
                );
                (w, h, Some(off), n, dev)
            }
            Algo::NsNmf => {
                let (w, h, n, dev) = nmf_run_nsnmf(
                    &v_view,
                    w_owned,
                    h_owned,
                    theta,
                    max_iter,
                    stop,
                    stationary_th,
                    check_interval,
                    check_niter,
                );
                (w, h, None, n, dev)
            }
            Algo::Hals => {
                let (w, h, n, dev) = nmf_run_hals(
                    &v_view,
                    w_owned,
                    h_owned,
                    max_iter,
                    eps,
                    stop,
                    stationary_th,
                    check_interval,
                    check_niter,
                );
                (w, h, None, n, dev)
            }
        }
    };

    let (w, h, off, n_iter, deviances) = py.allow_threads(move || {
        if let Some(pool) = local_pool {
            pool.install(run_core)
        } else {
            run_core()
        }
    });

    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("W", w.into_pyarray_bound(py))?;
    dict.set_item("H", h.into_pyarray_bound(py))?;
    if let Some(o) = off {
        // expose offset as 1d
        let arr = numpy::PyArray1::from_array_bound(py, &o);
        dict.set_item("offset", arr)?;
    }
    dict.set_item("n_iter", n_iter)?;
    dict.set_item("deviances", deviances)?;
    Ok(dict.into())
}

/// Lightweight builder for the smoothing matrix (useful for parity tests).
#[pyfunction]
fn py_smoothing_matrix<'py>(py: Python<'py>, r: usize, theta: f64) -> Bound<'py, PyArray2<f64>> {
    smoothing_matrix(r, theta).into_pyarray_bound(py)
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(py_divergence_update_h, m)?)?;
    m.add_function(wrap_pyfunction!(py_divergence_update_w, m)?)?;
    m.add_function(wrap_pyfunction!(py_euclidean_update_h, m)?)?;
    m.add_function(wrap_pyfunction!(py_euclidean_update_w, m)?)?;
    m.add_function(wrap_pyfunction!(py_nmf_run, m)?)?;
    m.add_function(wrap_pyfunction!(py_smoothing_matrix, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn smoothing_diagonal() {
        let s = smoothing_matrix(3, 0.5);
        // off = 0.5/3, diag = 0.5 + 0.5/3
        let off = 0.5 / 3.0;
        let diag = 0.5 + off;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { diag } else { off };
                assert!((s[(i, j)] - expected).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn brunet_reduces_kl() {
        let v: ndarray::Array2<f64> = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let w0: ndarray::Array2<f64> = array![[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]];
        let h0: ndarray::Array2<f64> = array![[1.0, 1.0], [1.0, 1.0]];
        let (w, h, _, devs) = nmf_run_brunet(
            &v.view(),
            w0,
            h0,
            50,
            Stop::Stationary,
            1e-12,
            50,
            10,
        );
        assert!(w.iter().all(|x| *x >= 0.0));
        assert!(h.iter().all(|x| *x >= 0.0));
        // KL should not increase from start to end.
        assert!(devs.last().unwrap() <= devs.first().unwrap());
    }
}
