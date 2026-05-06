# Generate reference fixtures from R `NMF` package for parity tests.
#
# Usage:  Rscript reference_nmf.R <out_dir> [seed=1234] [n=80] [p=30] [rank=4] [max_iter=50]
#
# Writes (TSV, no header / row.names, full numeric precision):
#   V.tsv          n×p target
#   W0.tsv         n×r initial basis
#   H0.tsv         r×p initial coef
#   <method>__W.tsv, <method>__H.tsv  — final factors after `max_iter` iterations
# for method ∈ {brunet, lee, offset, nsNMF}, plus offset__off.tsv for offset.
#
# Algorithms run with `.opt='-cb'` so no consensus tracking, and the user-
# supplied W0/H0 is used verbatim (no rescaling pre-step). max.iter caps the
# run; the connectivity stop is disabled by setting stopconv to a huge value.

suppressPackageStartupMessages({
    .libPaths(c("/scratch/users/steorra/env/CMAP_Rlib", .libPaths()))
    library(NMF)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- args[1]
seed_v  <- as.integer(if (length(args) >= 2) args[2] else 1234L)
n       <- as.integer(if (length(args) >= 3) args[3] else 80L)
p       <- as.integer(if (length(args) >= 4) args[4] else 30L)
rank    <- as.integer(if (length(args) >= 5) args[5] else 4L)
max_it  <- as.integer(if (length(args) >= 6) args[6] else 50L)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(seed_v)
# Build a non-negative target matrix with rank-`rank` structure plus noise.
W_true <- matrix(runif(n * rank, 0.1, 1.5), n, rank)
H_true <- matrix(runif(rank * p, 0.1, 1.5), rank, p)
V <- W_true %*% H_true + matrix(runif(n * p, 0, 0.05), n, p)

# Initial factors (also reproducible from the same seed stream).
W0 <- matrix(runif(n * rank, 0.0, max(V)), n, rank)
H0 <- matrix(runif(rank * p, 0.0, max(V)), rank, p)

# helper to write a matrix without R's row/col labels.
.write_mat <- function(M, path) {
    write.table(M, path, sep = "\t", row.names = FALSE, col.names = FALSE,
                quote = FALSE, na = "NA")
}
.write_mat(V,  file.path(out_dir, "V.tsv"))
.write_mat(W0, file.path(out_dir, "W0.tsv"))
.write_mat(H0, file.path(out_dir, "H0.tsv"))

# A custom seeding method that just installs our W0/H0 verbatim. This is the
# cleanest way to give NMF::nmf() a deterministic starting point.
.seed_fn <- function(model, target, ...) {
    basis(model) <- W0
    coef(model)  <- H0
    model
}

run_one <- function(method) {
    cat("[reference_nmf.R] running", method, "...\n")
    fit <- NMF::nmf(V, rank = rank, method = method, seed = .seed_fn,
                    .options = '-cb',
                    .pbackend = NA,
                    nrun = 1, maxIter = max_it,
                    # disable connectivity stop: stopconv high enough that we
                    # always hit maxIter first.
                    stopconv = 10L * max_it)
    list(W = basis(fit), H = coef(fit), fit = fit)
}

for (method in c("brunet", "lee")) {
    out <- run_one(method)
    .write_mat(out$W, file.path(out_dir, paste0(method, "__W.tsv")))
    .write_mat(out$H, file.path(out_dir, paste0(method, "__H.tsv")))
}

# offset: the model is NMFOffset; expose the offset slot too.
{
    fit <- NMF::nmf(V, rank = rank, method = "offset", seed = .seed_fn,
                    .options = '-cb', .pbackend = NA,
                    nrun = 1, maxIter = max_it, stopconv = 10L * max_it)
    .write_mat(basis(fit), file.path(out_dir, "offset__W.tsv"))
    .write_mat(coef(fit),  file.path(out_dir, "offset__H.tsv"))
    .write_mat(matrix(fit@fit@offset, ncol = 1L),
               file.path(out_dir, "offset__off.tsv"))
}

# nsNMF: NMFns model with smoothing theta. Use the default theta=0.5.
{
    fit <- NMF::nmf(V, rank = rank, method = "nsNMF", seed = .seed_fn,
                    .options = '-cb', .pbackend = NA,
                    nrun = 1, maxIter = max_it, stopconv = 10L * max_it)
    .write_mat(basis(fit), file.path(out_dir, "nsNMF__W.tsv"))
    .write_mat(coef(fit),  file.path(out_dir, "nsNMF__H.tsv"))
    .write_mat(matrix(fit@fit@theta, ncol = 1L),
               file.path(out_dir, "nsNMF__theta.tsv"))
}

cat("[reference_nmf.R] done. Wrote fixtures to", out_dir, "\n")
