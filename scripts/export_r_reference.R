args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript scripts/export_r_reference.R <fid_root> <out_dir>")
}

fid_root <- args[[1]]
out_dir <- args[[2]]

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(MALDIquant)
  library(MALDIquantForeign)
  library(jsonlite)
})

find_fid_folders <- function(root) {
  fid_files <- list.files(root, pattern = "^fid$", recursive = TRUE, full.names = TRUE)
  dirs <- unique(dirname(fid_files))
  dirs
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

  spectra <- importBrukerFlex(path = folder, verbose = FALSE)
  if (length(spectra) == 0) {
    next
  }

  sp <- spectra[[1]]
  payload <- list(
    sample_id = sid,
    source = folder,
    mz = as.numeric(mass(sp)),
    intensity = as.numeric(intensity(sp))
  )

  write_json(payload, path = out_json, auto_unbox = TRUE, digits = NA)
  cat(sprintf("OK\t%s\n", out_json))
}
