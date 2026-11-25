# Shared R interface for the simulation and the GAMA case study.
# Matrices on the pipes are little-endian doubles in R's column-major order.
args <- commandArgs(trailingOnly = TRUE)
n <- as.integer(args[1])
m <- as.integer(args[2])
d <- as.integer(args[3])
seed <- as.integer(args[4])
threads <- as.integer(args[5])
mode <- args[6]
trees <- if (length(args) >= 7L) as.integer(args[7]) else 1500L
leaf <- if (length(args) >= 8L) as.integer(args[8]) else 10L
suppressPackageStartupMessages(library(drf))
stopifnot(as.character(packageVersion("drf")) == "1.3.0")
input <- file("/dev/stdin", "rb", raw = TRUE)
output <- file("/dev/stdout", "wb", raw = TRUE)
read_matrix <- function(rows, cols) {
    matrix(readBin(input, "double", n = rows * cols, size = 8, endian = "little"), rows, cols)
}
send <- function(x) {
    writeBin(as.double(x), output, size = 8, endian = "little")
    flush(output)
}
x <- read_matrix(n, d)
y <- read_matrix(n, 1L)
q <- read_matrix(m, d)
# drf's R-side bandwidth subsample uses the R RNG as well as the native forest seed.
set.seed(seed)
fit <- drf(x, y, num.trees = trees, min.node.size = leaf, ci.group.size = 100L,
           num.threads = threads, seed = seed)
w <- predict(fit, newdata = q, num.threads = threads)$weights
mass <- Matrix::rowSums(w)
stopifnot(nrow(w) == m, ncol(w) == n, max(abs(mass - 1)) < 1e-8, min(w) >= 0)
if (mode == "weights") {
    send(as.matrix(w))
} else {
    ord <- order(y[, 1])
    send(cumsum(as.numeric(Matrix::crossprod(w, 1 / (mass * m)))[ord]))
    if (mode == "primary") {
        maximum <- numeric(n)
        cumulative <- numeric(m)
        for (first in seq.int(1L, n, by = 128L)) {
            last <- min(first + 127L, n)
            block <- as.matrix(w[, ord[seq.int(first, last)], drop = FALSE])
            for (j in seq_len(ncol(block))) {
                cumulative <- cumulative + block[, j]
                maximum[first + j - 1L] <- mean((cumulative / mass)^10)
            }
        }
        send(maximum)
    }
    repeat {
        count <- readBin(input, "integer", n = 1L, size = 4, endian = "little")
        if (!length(count) || count == 0L) break
        ids <- readBin(input, "integer", n = count, size = 4, endian = "little")
        stopifnot(length(ids) == count, all(diff(ids) > 0L), min(ids) >= 0L, max(ids) <= n)
        cumulative <- numeric(m)
        last <- 0L
        values <- matrix(0, m, count)
        for (j in seq_along(ids)) {
            k <- ids[j]
            if (k > last) cumulative <- cumulative + Matrix::rowSums(w[, ord[seq.int(last + 1L, k)], drop = FALSE])
            last <- k
            values[, j] <- cumulative / mass
        }
        send(values)
    }
}
close(input)
close(output)
