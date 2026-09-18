
# ==================== Perf recorder (same standard as perf_utils.py) ====================
# Usage: cd data/ && Rscript step0_seurat_process.R
# Writes to the same file ../enhanced_results/perf_report.jsonl (fields identical to the Python version)
# Note: peak_rss_mb on the R side is sampled at exit (ps rss); the Python side reports the exact peak (ru_maxrss)
perf_task  <- "step0_seurat_process_R"
perf_out   <- "../enhanced_results/perf_report.jsonl"
perf_t0    <- proc.time()[["elapsed"]]
perf_start <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
perf_write <- function(status = "ok") {
  rss_kb <- as.numeric(system2("ps", c("-o", "rss=", "-p", Sys.getpid()), stdout = TRUE))
  rec <- sprintf(
    paste0('{"task": "%s", "status": "%s", "wall_s": %.2f, "peak_rss_mb": %.1f, ',
           '"start": "%s", "end": "%s", "argv": "Rscript %s", "python": "%s", ',
           '"host": "%s", "note": "peak_rss_mb sampled at end (R)"}'),
    perf_task, status, proc.time()[["elapsed"]] - perf_t0, rss_kb / 1024,
    perf_start, format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
    paste(commandArgs(trailingOnly = TRUE), collapse = " "),
    R.version.string, Sys.info()[["nodename"]]
  )
  dir.create(dirname(perf_out), showWarnings = FALSE, recursive = TRUE)
  cat(rec, "\n", file = perf_out, append = TRUE, sep = "")
  cat(sprintf("[perf] %s -> %s\n", perf_task, perf_out))
}

# Load required packages
library(Seurat)
library(SeuratData)
library(ggplot2)
library(glmGamPoi)
library(dplyr)
library(tidyr)

# ==================== Step 0: Load data ====================
sobj <- UpdateSeuratObject(ifnb)

donor_map <- read.csv("ifnb_donor_map.csv", stringsAsFactors = FALSE, na.strings = c("", "NA"))
donor_vec <- setNames(donor_map$donor_id, donor_map$cell_barcode)
sobj$donor_id <- unname(donor_vec[colnames(sobj)])
cat("Donor match rate:", mean(!is.na(sobj$donor_id)), "\n")

# Filter out cells without donor assignment
cells_before <- ncol(sobj)
sobj <- subset(sobj, cells = colnames(sobj)[!is.na(sobj$donor_id)])
cat("Cells:", cells_before, "->", ncol(sobj), "after donor filter\n")

# Define donor as batch. Prefix "D" avoids numeric-looking batch labels being
sobj$batch <- factor(paste0("D", sobj$donor_id))
print(table(sobj$batch))
print(table(sobj$batch, sobj$stim))  # donor x condition must be crossed (every donor has both)

# ==================== Step 1: Split layers (critical step) ====================
sobj[["RNA"]] <- split(sobj[["RNA"]], f = sobj$batch)

# ==================== Step 2: SCTransform normalization ====================
sobj <- SCTransform(sobj, vst.flavor = "v2", verbose = TRUE)

# ==================== Step 3: PCA ====================
sobj <- RunPCA(sobj, npcs = 30, verbose = FALSE)

# ==================== Step 4: Integrate using IntegrateLayers ====================
sobj <- IntegrateLayers(
  object = sobj,
  method = CCAIntegration,
  orig.reduction = "pca",
  new.reduction = "integrated.cca",
  normalization.method = "SCT",
  dims = 1:30,
  verbose = TRUE
)

# ==================== Step 5: Downstream analysis ====================
sobj <- FindNeighbors(sobj, reduction = "integrated.cca", dims = 1:30, verbose = FALSE)
sobj <- FindClusters(sobj, resolution = 0.5, verbose = FALSE)
sobj <- RunUMAP(sobj, reduction = "integrated.cca", dims = 1:30, reduction.name = "umap.integrated", verbose = FALSE)

# ==================== Step 6: Visualization ====================
p1 <- DimPlot(sobj, reduction = "umap.integrated", label = TRUE) +
  ggtitle("UMAP after batch correction (clusters)")
print(p1)

p2 <- DimPlot(sobj, reduction = "umap.integrated", group.by = "stim") +
  ggtitle("UMAP after batch correction (by stim)")
print(p2)

p3 <- DimPlot(sobj, reduction = "umap.integrated", group.by = "batch") +
  ggtitle("UMAP after batch correction (by batch)")
print(p3)

p4 <- DimPlot(sobj, reduction = "umap.integrated", group.by = "seurat_annotations") +
  ggtitle("UMAP after batch correction (by seurat_annotations)")
print(p4)
# ==================== Step 7: Save results ====================
cluster_data <- data.frame(
  index = rownames(sobj[[]]),
  seurat_label = sobj$seurat_clusters
)
write.csv(cluster_data, "sobj_cluster_labels_v5.csv", row.names = FALSE)

umap_matrix <- Embeddings(sobj, reduction = "umap.integrated")
write.csv(data.frame(
  index = rownames(umap_matrix),
  UMAP_1 = umap_matrix[, 1],
  UMAP_2 = umap_matrix[, 2]
), "sobj_umap_matrix_v5.csv", row.names = FALSE)

# Pre-integration PCA (kept for diagnostics/backward compatibility only;
# the native integrated representation is integrated.cca, exported below)
pca_matrix <- Embeddings(sobj, reduction = "pca")
write.csv(pca_matrix, "sobj_pca_matrix_v5.csv")

# Integrated CCA embedding: the native integrated cell representation produced
# by CCAIntegration (all downstream steps above - neighbors, clusters, UMAP -
# are computed on it). This is the embedding on which Seurat is benchmarked.
cca_matrix <- Embeddings(sobj, reduction = "integrated.cca")
write.csv(cca_matrix, "sobj_cca_matrix_v5.csv")

saveRDS(sobj, "sobj_seurat_processed.rds")

metadata <- sobj@meta.data
metadata[] <- lapply(metadata, function(x) if(is.factor(x)) as.character(x) else x)
write.csv(metadata, "sobj_seurat_processed_metadata.csv", row.names = TRUE)

# ==================== Step 8: Convert to h5ad-compatible format ====================
library(Matrix)
sobj <- readRDS("sobj_seurat_processed.rds")
#counts <- GetAssayData(sobj, assay = "RNA", layer = "counts")
sobj[["RNA"]] <- JoinLayers(sobj[["RNA"]])
counts <- LayerData(sobj, assay = "RNA", layer = "counts")
writeMM(counts, "counts.mtx")
writeLines(colnames(counts), "cells.txt")
writeLines(rownames(counts), "genes.txt")
cat("Counts matrix:", dim(counts), "\n")

# ==================== Perf: write record on normal completion ====================
perf_write("ok")