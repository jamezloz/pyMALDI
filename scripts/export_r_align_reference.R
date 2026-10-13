args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript scripts/export_r_align_reference.R <fid_root> <out_json> [transform] [smooth] [smooth_half] [baseline] [baseline_half] [baseline_iter] [calibrate] [noise] [detect_half] [snr] [align_tolerance] [warping_method]")
}

fid_root <- args[[1]]
out_json <- args[[2]]

transform_method <- ifelse(length(args) >= 3, args[[3]], "none")
smooth_method <- ifelse(length(args) >= 4, args[[4]], "SavitzkyGolay")
smooth_half <- ifelse(length(args) >= 5, as.integer(args[[5]]), 10L)
baseline_method <- ifelse(length(args) >= 6, args[[6]], "SNIP")
baseline_half <- ifelse(length(args) >= 7, as.integer(args[[7]]), 20L)
baseline_iter <- ifelse(length(args) >= 8, as.integer(args[[8]]), 40L)
calibrate_method <- ifelse(length(args) >= 9, args[[9]], "TIC")
noise_method <- ifelse(length(args) >= 10, args[[10]], "MAD")
detect_half <- ifelse(length(args) >= 11, as.integer(args[[11]]), 20L)
snr <- ifelse(length(args) >= 12, as.numeric(args[[12]]), 2.0)
align_tol <- ifelse(length(args) >= 13, as.numeric(args[[13]]), 0.002)
warping_method <- ifelse(length(args) >= 14, args[[14]], "lowess")

suppressPackageStartupMessages({
  library(MALDIquant)
  library(MALDIquantForeign)
  library(jsonlite)
})

find_fid_folders <- function(root) {
  fid_files <- list.files(root, pattern = "^fid$", recursive = TRUE, full.names = TRUE)
  sort(unique(dirname(fid_files)))
}

sample_id_from_path <- function(path) {
  parts <- strsplit(normalizePath(path, winslash = "/"), "/")[[1]]
  n <- length(parts)
  if (n >= 4) {
    return(paste(parts[(n-3):n], collapse = "__"))
  }
  basename(path)
}

folders <- find_fid_folders(fid_root)
if (length(folders) == 0) {
  stop("No fid folders found")
}

spec <- list()
sids <- c()
for (folder in folders) {
  sid <- sample_id_from_path(folder)
  s <- importBrukerFlex(path = folder, verbose = FALSE)
  if (tolower(transform_method) != "none") {
    s <- transformIntensity(s, method = transform_method)
  }
  if (tolower(smooth_method) != "none") {
    s <- smoothIntensity(s, method = smooth_method, halfWindowSize = smooth_half)
  }
  if (tolower(baseline_method) != "none") {
    bl <- tolower(baseline_method)
    if (bl == "snip") {
      s <- removeBaseline(s, method = baseline_method, iterations = baseline_iter)
    } else if (bl == "tophat" || bl == "median") {
      s <- removeBaseline(s, method = baseline_method, halfWindowSize = baseline_half)
    } else {
      s <- removeBaseline(s, method = baseline_method)
    }
  }
  if (tolower(calibrate_method) != "none") {
    s <- calibrateIntensity(s, method = calibrate_method)
  }
  spec[[length(spec) + 1]] <- s[[1]]
  sids <- c(sids, sid)
}

peaks <- detectPeaks(spec, halfWindowSize = detect_half, method = noise_method, SNR = snr)
aligned <- alignSpectra(spec, halfWindowSize = detect_half, noiseMethod = noise_method, SNR = snr, tolerance = align_tol, warpingMethod = warping_method)
aligned_peaks <- detectPeaks(aligned, halfWindowSize = detect_half, method = noise_method, SNR = snr)

items <- list()
for (i in seq_along(sids)) {
  items[[length(items) + 1]] <- list(
    sample_id = sids[[i]],
    peaks_before = as.numeric(mass(peaks[[i]])),
    peaks_after = as.numeric(mass(aligned_peaks[[i]]))
  )
}

payload <- list(
  options = list(
    transform_method = transform_method,
    smooth_method = smooth_method,
    smooth_half = smooth_half,
    baseline_method = baseline_method,
    baseline_half = baseline_half,
    baseline_iter = baseline_iter,
    calibrate_method = calibrate_method,
    noise_method = noise_method,
    detect_half = detect_half,
    snr = snr,
    align_tol = align_tol,
    warping_method = warping_method
  ),
  items = items
)

write_json(payload, path = out_json, auto_unbox = TRUE, digits = NA)
cat(sprintf("OK\t%s\n", out_json))
