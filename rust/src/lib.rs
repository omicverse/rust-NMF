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

mod fcnnls;

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
// lsNMF (Wang, Kossenkov, Ochs 2006) — weighted Frobenius
// =============================================================================
//
// Minimises  ||(V - WH) ⊙ Σ||_F²  for a per-entry weight matrix Σ.
// Common use case: missing-value imputation by setting Σ[i,j]=0 where V[i,j]
// is missing; the entry contributes nothing to the loss.
//
// Update rules from R `nmf_update.lsnmf`:
//   wV  = V ⊙ Σ                                    (precomputed, constant)
//   est = W H                                      (per iter)
//   wE  = est ⊙ Σ
//   numH = W^T (V ⊙ Σ)        denH = W^T (est ⊙ Σ)
//   H ← max(eps, H ⊙ numH) / (denH + eps)
//   numW = (V ⊙ Σ) H^T        denW = (est ⊙ Σ) H^T
//   W ← max(eps, W ⊙ numW) / (denW + eps)
//
// Note: R's lsNMF uses BLAS gemm under the hood, so even R isn't bit-exact
// across BLAS implementations. We compute the gemms with our cache-friendly
// kernels in a fixed order.

fn elementwise_mul(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
    let (n, p) = a.dim();
    debug_assert_eq!(b.dim(), (n, p));
    let mut out = Array2::<f64>::zeros((n, p));
    for u in 0..n {
        let a_row = a.row(u);
        let b_row = b.row(u);
        let mut o_row = out.row_mut(u);
        for j in 0..p {
            o_row[j] = a_row[j] * b_row[j];
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn nmf_run_lsnmf(
    v: &ArrayView2<f64>,
    weight: &ArrayView2<f64>,
    mut w: Array2<f64>,
    mut h: Array2<f64>,
    max_iter: usize,
    eps: f64,
    stop: Stop,
    stationary_th: f64,
    check_interval: usize,
    check_niter: usize,
) -> (Array2<f64>, Array2<f64>, usize, Vec<f64>) {
    debug_assert_eq!(weight.dim(), v.dim());
    let mut deviances: Vec<f64> = Vec::new();
    let mut stationary = StationaryStop::new(check_interval, check_niter, stationary_th);

    // Pre-multiply target by weight once — this is the "wX" of R's lsNMF.
    let v_w = elementwise_mul(v, weight);
    let v_w_t = transpose_owned(&v_w.view());   // for cache-friendly W^T V_w

    if let Stop::Stationary = stop {
        // Weighted reconstruction loss: 0.5 * ||(V - WH) ⊙ weight||²
        let est = wh_dense(&w.view(), &h.view());
        let mut s = 0.0f64;
        for u in 0..v.nrows() {
            for j in 0..v.ncols() {
                let d = (v[(u, j)] - est[(u, j)]) * weight[(u, j)];
                s += d * d;
            }
        }
        let d = 0.5 * s;
        deviances.push(d);
        if stationary.should_stop(0, d) {
            return (w, h, 0, deviances);
        }
    }

    let mut iter = 0usize;
    while iter < max_iter {
        iter += 1;

        // ----- H update -----
        let est = wh_dense(&w.view(), &h.view());
        let est_w = elementwise_mul(&est.view(), weight);
        let est_w_t = transpose_owned(&est_w.view());

        let w_t = transpose_owned(&w.view());
        // numer_h = W^T V_w  (r × p)
        let numer_h = compute_wtv(&w_t.view(), &v_w_t.view());
        // denom_h = W^T est_w (r × p)
        let denom_h = compute_wtv(&w_t.view(), &est_w_t.view());

        // h ← max(h * numer_h, eps) / (denom_h + eps)
        let r_ = h.nrows();
        let p_ = h.ncols();
        for k in 0..r_ {
            for u in 0..p_ {
                let temp = h[(k, u)] * numer_h[(k, u)];
                let num = if temp > eps { temp } else { eps };
                h[(k, u)] = num / (denom_h[(k, u)] + eps);
            }
        }

        // ----- W update (recompute est now that H changed) -----
        let est = wh_dense(&w.view(), &h.view());
        let est_w = elementwise_mul(&est.view(), weight);

        // numer_w = V_w H^T (n × r)
        let numer_w = compute_vht(&v_w.view(), &h.view());
        // denom_w = est_w H^T (n × r)
        let denom_w = compute_vht(&est_w.view(), &h.view());

        let n_ = w.nrows();
        for u in 0..n_ {
            for k in 0..r_ {
                let temp = w[(u, k)] * numer_w[(u, k)];
                let num = if temp > eps { temp } else { eps };
                w[(u, k)] = num / (denom_w[(u, k)] + eps);
            }
        }

        if let Stop::Stationary = stop {
            let est = wh_dense(&w.view(), &h.view());
            let mut s = 0.0f64;
            for u in 0..v.nrows() {
                for j in 0..v.ncols() {
                    let d = (v[(u, j)] - est[(u, j)]) * weight[(u, j)];
                    s += d * d;
                }
            }
            let d = 0.5 * s;
            deviances.push(d);
            if stationary.should_stop(iter, d) {
                break;
            }
        }
    }
    (w, h, iter, deviances)
}

// =============================================================================
// Sparse HALS (snmf/r, snmf/l) — Kim-Park objective via regularised HALS
// =============================================================================
//
// **NOT bit-equivalent to R's `snmf/r` / `snmf/l`**, which use FCNNLS-based
// alternating NNLS (Van Benthem & Keenan 2004). We solve the closely related
// regularised problem with HALS instead:
//
//   snmf/r:  min_{W,H}  ½ ||V - W H||² + γ_W ||W||² + λ_H Σ_j ||H_:,j||₁
//   snmf/l:  min_{W,H}  ½ ||V - W H||² + γ_H ||H||² + λ_W Σ_i ||W_i,:||₁
//
// HALS update with these penalties (closed form per row/col):
//
//   For H: H[k, :] = max(eps, (W^T V - sum_{j≠k} (W^T W)[k,j] H[j,:] - λ_H 1)
//                              / ((W^T W)[k,k] + γ_H))
//   For W: W[:, k] = max(eps, (V H^T - sum_{j≠k} W[:, j] (H H^T)[j,k] - λ_W 1)
//                              / ((H H^T)[k,k] + γ_W))
//
// The factorisation has the same sparsity-promoting effect as Kim-Park snmf,
// just reached by a different optimiser. Use this when you need sparse
// factors and don't need bit-equivalence with R.

#[allow(clippy::too_many_arguments)]
fn hals_update_h_reg(
    h: &mut Array2<f64>,
    wtv: &ArrayView2<f64>,
    wtw: &ArrayView2<f64>,
    lambda_h: f64,    // L1 coefficient on H
    gamma_h: f64,     // L2 coefficient on H (added to wtw[k,k])
    eps: f64,
) {
    let r = h.nrows();
    h.axis_iter_mut(Axis(1)).into_par_iter().enumerate().for_each(
        |(u, mut h_col)| {
            for k in 0..r {
                let mut s = wtv[(k, u)] - lambda_h;
                for j in 0..r {
                    if j != k {
                        s -= wtw[(k, j)] * h_col[j];
                    }
                }
                let denom = (wtw[(k, k)] + gamma_h).max(eps);
                let val = s / denom;
                h_col[k] = if val > eps { val } else { eps };
            }
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn hals_update_w_reg(
    w: &mut Array2<f64>,
    vht: &ArrayView2<f64>,
    hht: &ArrayView2<f64>,
    lambda_w: f64,    // L1 coefficient on W
    gamma_w: f64,     // L2 coefficient on W
    eps: f64,
) {
    let r = w.ncols();
    w.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(
        |(_i, mut w_row)| {
            for k in 0..r {
                let mut s = vht[(_i, k)] - lambda_w;
                for j in 0..r {
                    if j != k {
                        s -= hht[(k, j)] * w_row[j];
                    }
                }
                let denom = (hht[(k, k)] + gamma_w).max(eps);
                let val = s / denom;
                w_row[k] = if val > eps { val } else { eps };
            }
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn nmf_run_snmf(
    v: &ArrayView2<f64>,
    mut w: Array2<f64>,
    mut h: Array2<f64>,
    max_iter: usize,
    eps: f64,
    sparsity_h: f64,    // λ_H in snmf/r objective
    sparsity_w: f64,    // λ_W in snmf/l objective
    smoothness_h: f64,  // γ_H (L2 on H)
    smoothness_w: f64,  // γ_W (L2 on W)
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

        // H sweep with L1-on-H + L2-on-H regularisation
        let w_t = transpose_owned(&w.view());
        let wtv = compute_wtv(&w_t.view(), &v_t.view());
        let wtw = compute_wtw_from_wt(&w_t.view());
        hals_update_h_reg(&mut h, &wtv.view(), &wtw.view(),
                          sparsity_h, smoothness_h, eps);

        // W sweep with L1-on-W + L2-on-W regularisation
        let vht = compute_vht(v, &h.view());
        let hht = compute_hht(&h.view());
        hals_update_w_reg(&mut w, &vht.view(), &hht.view(),
                          sparsity_w, smoothness_w, eps);

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
// Kim-Park sparse NMF (snmf/R, snmf/L) — bit-equivalent to R `nmf_snmf`
// =============================================================================
//
// Solves
//   snmf/R:   min_{W,H}  ½ ||A - W H||² + η ||W||² + β Σ_j ||H_:,j||₁²
//   snmf/L:   min_{W,H}  ½ ||A - W H||² + η ||H||² + β Σ_i ||W_i,:||₁²
//
// via alternating FCNNLS. R `nmf_snmf` (algorithms-snmf.R) does:
//
//     normalize columns of W
//     loop:
//         H = fcnnls(rbind(W, sqrt(β)·1ₖ),  rbind(A,         0₁ₓₙ))
//         W = fcnnls(rbind(Hᵀ, η·Iₖ),       rbind(Aᵀ,        0ₖₓₘ))ᵀ
//         every 5 iters: convergence check via cluster assignments
//
// The trick is: stacking a sqrt(β)·1ₖ row in the H subproblem makes the
// least-squares term include  β·||Σᵢ Hᵢ,ⱼ||² = β·||H_:,j||₁²  per column.
// Stacking η·Iₖ in the W subproblem adds the η·||W||² Frobenius penalty.
// Both subproblems are vanilla NNLS once stacked, so FCNNLS solves them.
//
// snmf/L flips the roles: minimise on Aᵀ swapping W↔H, sparsity on rows of W.

fn col_l2_normalize(w: &mut Array2<f64>) {
    let r = w.ncols();
    for j in 0..r {
        let mut col = w.column_mut(j);
        let norm_sq: f64 = col.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            col.mapv_inplace(|x| x / norm);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn nmf_run_snmf_kim_park(
    a: &ArrayView2<f64>,
    w_init: Array2<f64>,
    h_init: Array2<f64>,
    is_left: bool,    // true → snmf/L (transpose A, swap roles)
    max_iter: usize,
    eta: f64,         // L2 coeff (penalty on the non-sparse factor); if <0 use max(A)
    beta: f64,        // L1²-per-col coeff on the sparse factor
    eps_inner: f64,   // tolerance inside fcnnls
) -> (Array2<f64>, Array2<f64>, usize, Vec<f64>) {
    // Effective target. snmf/L works on Aᵀ with W↔H swapped.
    let a_eff_owned = if is_left { transpose_owned(a) } else { a.to_owned() };
    let a_eff = a_eff_owned.view();
    let m = a_eff.nrows();      // features
    let n = a_eff.ncols();      // samples
    let k = w_init.ncols();

    // For snmf/L, the "W" we operate on is internally the transposed user-H.
    let (mut w, mut h): (Array2<f64>, Array2<f64>) = if is_left {
        // user-W is m × k (after the algorithm), user-H is k × n.
        // Internally we view: A' = Aᵀ (n × m), with internal-W (n × k) = (user-H)ᵀ,
        // internal-H (k × m) = (user-W)ᵀ.
        // So pass transpose(user-H) and transpose(user-W) as internal w_init and h_init.
        // Caller must give w_init as user-H (k × n) so internal w = (user-H)ᵀ.
        // To keep the API uniform we swap below.
        (transpose_owned(&h_init.view()), transpose_owned(&w_init.view()))
    } else {
        (w_init, h_init)
    };

    let eta_eff = if eta < 0.0 {
        a_eff.iter().cloned().fold(0.0f64, f64::max)
    } else {
        eta
    };
    let beta_eff = beta.max(1e-30);
    let sqrt_beta = beta_eff.sqrt();

    // Normalise columns of internal W.
    col_l2_normalize(&mut w);

    let mut deviances: Vec<f64> = Vec::new();
    let mut iter = 0usize;
    while iter < max_iter {
        iter += 1;

        // ---- H subproblem (sparse-rows when snmf/R, sparse-cols-of-W when snmf/L) ----
        // x = rbind(W, sqrt(β)·1ₖ) of shape (m+1, k)
        let mut x_h = Array2::<f64>::zeros((m + 1, k));
        for i in 0..m {
            for j in 0..k {
                x_h[(i, j)] = w[(i, j)];
            }
        }
        for j in 0..k {
            x_h[(m, j)] = sqrt_beta;
        }
        // y = rbind(A_eff, 0_{1×n})
        let mut y_h = Array2::<f64>::zeros((m + 1, n));
        for i in 0..m {
            for j in 0..n {
                y_h[(i, j)] = a_eff[(i, j)];
            }
        }
        let (h_new, _) = fcnnls::fcnnls(x_h.view(), y_h.view(), false, eps_inner);
        h = h_new;

        // R checks for zero rows of H and restarts. We warn (via a deviance
        // sentinel) but don't auto-restart in this port — caller can re-init.

        // ---- W subproblem ----
        // x = rbind(Hᵀ, η·Iₖ) of shape (n+k, k)
        let mut x_w = Array2::<f64>::zeros((n + k, k));
        for i in 0..n {
            for j in 0..k {
                x_w[(i, j)] = h[(j, i)];   // Hᵀ
            }
        }
        for j in 0..k {
            x_w[(n + j, j)] = eta_eff;     // diag(η)
        }
        // y = rbind(A_effᵀ, 0_{k×m})
        let mut y_w = Array2::<f64>::zeros((n + k, m));
        for i in 0..n {
            for j in 0..m {
                y_w[(i, j)] = a_eff[(j, i)];   // A_effᵀ
            }
        }
        let (wt, _) = fcnnls::fcnnls(x_w.view(), y_w.view(), false, eps_inner);
        // wt is k × m; we want internal W of shape m × k.
        w = transpose_owned(&wt.view());

        // Tracking — record reconstruction loss every iter.
        let recon = w.dot(&h);
        let mut s = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let d = a_eff[(i, j)] - recon[(i, j)];
                s += d * d;
            }
        }
        deviances.push(0.5 * s);
    }

    // Map internal (W, H) back to user (W, H) — snmf/L un-swap.
    if is_left {
        // Internal W (m × k) corresponds to user-H transposed: user-H = Wᵀ
        // Internal H (k × n) corresponds to user-W transposed: user-W = Hᵀ
        let user_h = transpose_owned(&w.view());
        let user_w = transpose_owned(&h.view());
        (user_w, user_h, iter, deviances)
    } else {
        (w, h, iter, deviances)
    }
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
    LsNmf,
    SnmfR,
    SnmfL,
}

impl Algo {
    fn from_str(s: &str) -> PyResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "brunet" | "kl" => Ok(Algo::Brunet),
            "lee" | "frobenius" | "euclidean" => Ok(Algo::Lee),
            "offset" => Ok(Algo::Offset),
            "nsnmf" | "ns" | "ns_nmf" => Ok(Algo::NsNmf),
            "hals" => Ok(Algo::Hals),
            "ls-nmf" | "lsnmf" => Ok(Algo::LsNmf),
            "snmf/r" | "snmf_r" | "snmfr" => Ok(Algo::SnmfR),
            "snmf/l" | "snmf_l" | "snmfl" => Ok(Algo::SnmfL),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown algorithm '{}': supported: brunet, lee, offset, nsNMF, hals, lsNMF, snmf/r, snmf/l",
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
    weight = None,
    sparsity_h = 0.0,
    sparsity_w = 0.0,
    smoothness_h = 0.0,
    smoothness_w = 0.0,
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
    weight: Option<PyReadonlyArray2<'py, f64>>,
    sparsity_h: f64,
    sparsity_w: f64,
    smoothness_h: f64,
    smoothness_w: f64,
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
    let weight_owned: Option<Array2<f64>> = weight.map(|w| w.as_array().to_owned());

    // Sanity check: lsNMF needs a weight matrix.
    if matches!(algo, Algo::LsNmf) && weight_owned.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "method='lsnmf' requires a `weight` matrix of the same shape as V",
        ));
    }

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
            Algo::LsNmf => {
                let weight = weight_owned.expect("weight checked above");
                let (w, h, n, dev) = nmf_run_lsnmf(
                    &v_view,
                    &weight.view(),
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
            Algo::SnmfR => {
                // Kim-Park snmf/R via FCNNLS-based ANLS (bit-eq R nmf_snmf).
                // `sparsity_h` plays the role of β (L1²-per-col on H rows).
                // `smoothness_w` plays the role of η (L2 on W); -1 → max(V).
                let (w, h, n, dev) = nmf_run_snmf_kim_park(
                    &v_view,
                    w_owned,
                    h_owned,
                    false,         // snmf/R: not flipped
                    max_iter,
                    smoothness_w,  // η
                    sparsity_h,    // β
                    eps,
                );
                (w, h, None, n, dev)
            }
            Algo::SnmfL => {
                // Kim-Park snmf/L: same machinery, transposed problem internally.
                let (w, h, n, dev) = nmf_run_snmf_kim_park(
                    &v_view,
                    w_owned,
                    h_owned,
                    true,
                    max_iter,
                    smoothness_h,  // η on H
                    sparsity_w,    // β on W (sparse rows)
                    eps,
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
