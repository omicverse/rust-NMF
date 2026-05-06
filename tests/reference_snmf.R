# Generate reference SNMF/R and SNMF/L fixtures from R `NMF::nmf_snmf`.
#
# Usage:  Rscript reference_snmf.R <out_dir> [seed=1234] [n=80] [p=30] [rank=4]
#         [max_iter=20] [eta=-1] [beta=0.01]
#
# Outputs (TSV):
#   V.tsv          n×p target matrix
#   W0.tsv         n×r initial W (column-normalised internally by snmf, but
#                  we store the un-normalised seed since callers must pass it
#                  as-is; Rust then col-normalises to match)
#   snmfR_W.tsv    n×r final W from snmf/R
#   snmfR_H.tsv    r×p final H from snmf/R
#   snmfL_W.tsv    n×r final W from snmf/L
#   snmfL_H.tsv    r×p final H from snmf/L

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
max_it  <- as.integer(if (length(args) >= 6) args[6] else 20L)
eta_v   <- as.numeric(if (length(args) >= 7) args[7] else -1)
beta_v  <- as.numeric(if (length(args) >= 8) args[8] else 0.01)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(seed_v)
W_true <- matrix(runif(n * rank, 0.1, 1.5), n, rank)
H_true <- matrix(runif(rank * p, 0.1, 1.5), rank, p)
V <- W_true %*% H_true + matrix(runif(n * p, 0, 0.05), n, p)

# Initial W (snmf only seeds W internally; H is computed in iter 1).
W0 <- matrix(runif(n * rank, 0.0, max(V)), n, rank)
H0 <- matrix(runif(rank * p, 0.0, max(V)), rank, p)  # only used by callers

.write_mat <- function(M, path) {
    write.table(M, path, sep = "\t", row.names = FALSE, col.names = FALSE,
                quote = FALSE, na = "NA")
}
.write_mat(V,  file.path(out_dir, "V.tsv"))
.write_mat(W0, file.path(out_dir, "W0.tsv"))
.write_mat(H0, file.path(out_dir, "H0.tsv"))

# snmf/R
.seed_R <- function(model, target, ...) {
    basis(model) <- W0
    coef(model)  <- H0
    model
}
fit_r <- NMF::nmf(V, rank = rank, method = "snmf/r", seed = .seed_R,
                  .options = "-cb", .pbackend = NA, nrun = 1,
                  maxIter = max_it, eta = eta_v, beta = beta_v,
                  bi_conv = c(0L, 10L * max_it),  # disable conv stop
                  eps_conv = 1e-30)
.write_mat(basis(fit_r), file.path(out_dir, "snmfR_W.tsv"))
.write_mat(coef(fit_r),  file.path(out_dir, "snmfR_H.tsv"))

# snmf/L — note R passes the SAME init via the seed function regardless of version
fit_l <- NMF::nmf(V, rank = rank, method = "snmf/l", seed = .seed_R,
                  .options = "-cb", .pbackend = NA, nrun = 1,
                  maxIter = max_it, eta = eta_v, beta = beta_v,
                  bi_conv = c(0L, 10L * max_it),
                  eps_conv = 1e-30)
.write_mat(basis(fit_l), file.path(out_dir, "snmfL_W.tsv"))
.write_mat(coef(fit_l),  file.path(out_dir, "snmfL_H.tsv"))

cat("[reference_snmf.R] done. eta=", eta_v, ", beta=", beta_v, "\n")
