# Deterministic test fixture, not a real-data or human-review receipt.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
dir <- normalizePath(args[[1]], mustWork = TRUE)
receipt <- normalizePath(args[[2]], mustWork = TRUE)
reader <- if (requireNamespace("arrow", quietly = TRUE)) "arrow" else "nanoparquet"
stopifnot(requireNamespace(reader, quietly = TRUE),
          requireNamespace("SingleCellExperiment", quietly = TRUE))
cat("R:", R.version.string, "\n")
for (package in c(reader, "SingleCellExperiment", "S4Vectors", "SummarizedExperiment")) {
  cat(package, as.character(utils::packageVersion(package)), "\n")
}
loader <- file.path(dir, "load_spacr_export.R")
source(loader)
stopifnot(identical(.spacr_export_dir, dir))
tables <- read_spacr_tables()
sce <- load_spacr_export()
stopifnot(inherits(sce, "SingleCellExperiment"), methods::validObject(sce))
objects <- tables$objects
features <- as.character(tables$features$feature)
keys <- as.character(objects$object_key)
stopifnot(length(keys) == 20L, !anyNA(keys), !anyDuplicated(keys),
          length(features) > 0L, !anyDuplicated(features))
expected <- t(as.matrix(objects[, features, drop = FALSE]))
storage.mode(expected) <- "double"
dimnames(expected) <- list(features, keys)
values <- SummarizedExperiment::assay(sce, "measurements")
stopifnot(identical(dimnames(values), dimnames(expected)),
          identical(is.na(values), is.na(expected)),
          isTRUE(all.equal(values, expected, tolerance = 0)),
          sum(values["cell_area", ]) == 2040,
          sum(is.na(values["pathogen_area", ])) == 4L)
column_values <- function(column) {
  if (is.factor(column)) as.character(column) else
  if (is.numeric(column)) as.numeric(column) else column
}
same_frame <- function(actual, expected) {
  actual <- as.data.frame(actual, check.names = FALSE)
  expected <- as.data.frame(expected, check.names = FALSE)
  stopifnot(identical(names(actual), names(expected)), nrow(actual) == nrow(expected))
  for (column in names(expected)) {
    stopifnot(isTRUE(all.equal(column_values(actual[[column]]),
                              column_values(expected[[column]]),
                              tolerance = 0, check.attributes = FALSE)))
  }
}
stopifnot(identical(colnames(sce), keys), identical(rownames(sce), features))
same_frame(SummarizedExperiment::colData(sce),
           objects[, setdiff(names(objects), features), drop = FALSE])
same_frame(SummarizedExperiment::rowData(sce), tables$features)
stopifnot("infected" %in% names(SummarizedExperiment::colData(sce)),
          identical(SingleCellExperiment::reducedDimNames(sce), "X_umap"))
embedding <- SingleCellExperiment::reducedDim(sce, "X_umap")
aligned <- tables$embeddings[match(keys, as.character(tables$embeddings$object_key)),
                              c("X_umap_1", "X_umap_2"), drop = FALSE]
stopifnot(identical(rownames(embedding), keys),
          isTRUE(all.equal(unname(embedding), unname(as.matrix(aligned)), tolerance = 0)),
          identical(as.numeric(embedding[, 1]), as.numeric(0:19)),
          identical(as.numeric(embedding[, 2]), -as.numeric(0:19)))
metadata <- S4Vectors::metadata(sce)
same_frame(metadata$wells, tables$wells)
well_keys <- intersect(c("plateID", "rowID", "columnID", "timeID"), names(tables$wells))
for (i in seq_len(nrow(tables$wells))) {
  matching <- rep(TRUE, nrow(objects))
  for (key in well_keys) {
    matching <- matching & as.character(objects[[key]]) == as.character(tables$wells[[key]][i])
  }
  stopifnot(sum(matching) == tables$wells$n_objects[i])
  for (feature in features) {
    stopifnot(isTRUE(all.equal(as.numeric(tables$wells[[feature]][i]),
                              mean(objects[[feature]][matching], na.rm = TRUE),
                              tolerance = 0)))
  }
}
parse_value <- if (requireNamespace("jsonlite", quietly = TRUE)) {
  function(value) jsonlite::fromJSON(value, simplifyVector = TRUE)
} else identity
for (section in unique(as.character(tables$provenance$section))) {
  rows <- tables$provenance[as.character(tables$provenance$section) == section, , drop = FALSE]
  expected_provenance <- stats::setNames(lapply(as.character(rows$value), parse_value),
                                        as.character(rows$key))
  stopifnot(identical(metadata$spacr[[section]], expected_provenance))
}
rds_path <- file.path(receipt, "native-saveRDS.rds")
saveRDS(sce, rds_path)
stopifnot(identical(sce, readRDS(rds_path)))
cli_path <- file.path(receipt, "cli-sce.rds")
cli_log <- file.path(receipt, "cli.log")
cli_status <- system2(file.path(R.home("bin"), "Rscript"),
                      c("--vanilla", shQuote(loader), shQuote(dir), shQuote(cli_path)),
                      stdout = cli_log, stderr = cli_log)
stopifnot(cli_status == 0L, isTRUE(all.equal(sce, readRDS(cli_path), tolerance = 0)))
available_rds <- character()
rds_dir <- Sys.getenv("RDS_EXPORT_DIR", unset = "")
require_table_rds <- nzchar(rds_dir)
if (!require_table_rds) rds_dir <- dir
rds_tables <- read_spacr_tables(rds_dir)
for (table in names(rds_tables)) {
  path <- file.path(rds_dir, paste0(table, ".rds"))
  if (require_table_rds) stopifnot(file.exists(path))
  if (file.exists(path)) {
    same_frame(readRDS(path), rds_tables[[table]])
    available_rds <- c(available_rds, table)
  }
}
cat("Native data-frame RDS tables checked:", paste(available_rds, collapse = ", "), "\n")
cat("PASS: all assay values/NA positions, keys, colData, rowData, embedding coordinates,\n",
    "well means/counts, provenance, source default, saveRDS/readRDS and CLI SCE RDS.\n")
