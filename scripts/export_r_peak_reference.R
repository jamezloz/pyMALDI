args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript scripts/export_r_peak_reference.R <fid_root> <out_dir> [transform] [smooth] [smooth_half] [baseline] [baseline_half] [baseline_iter] [calibrate] [noise] [detect_half] [snr]")
}

fid_root <- args[[1]]
out_dir <- args[[2]]

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

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(MALDIquant)
  library(MALDIquantForeign)
  library(jsonlite)
})

find_fid_folders <- function(root) {
  fid_files <- list.files(root, pattern = "^fid$", recursive = TRUE, full.names = TRUE)
  unique(dirname(fid_files))
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

for (folder in folders) {
  sid <- sample_id_from_path(folder)
  out_json <- file.path(out_dir, paste0(sid, ".json"))

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

  p <- detectPeaks(s, halfWindowSize = detect_half, method = noise_method, SNR = snr)

  payload <- list(
    sample_id = sid,
    source = folder,
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
      snr = snr
    ),
    mz = as.numeric(mass(p[[1]])),
    intensity = as.numeric(intensity(p[[1]]))
  )

  write_json(payload, path = out_json, auto_unbox = TRUE, digits = NA)
  cat(sprintf("OK\t%s\n", out_json))
}
