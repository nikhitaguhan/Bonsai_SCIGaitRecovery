if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse, readxl, janitor, fs,
  umap, cluster, factoextra,
  patchwork, ggrepel, RColorBrewer,
  randomForest, caret, MASS
)
library(dplyr, warn.conflicts = FALSE)

# ============================================================================
# PARAMETERS
# ============================================================================

NORMALIZE_TO_BASELINE <- TRUE
COLLAPSE_TIMEPOINTS <- TRUE
UMAP_N_NEIGHBORS <- 15
UMAP_MIN_DIST <- 0.1

# Feature selection strategy
USE_RFE <- TRUE                    # Use Recursive Feature Elimination
RFE_N_FEATURES <- 30               # Reduce to 30 features (more conservative)

# Cross-validation
USE_LOMO_CV <- TRUE                # Leave-One-Mouse-Out for realistic estimates

cat("\n=== PRESENTATION PIPELINE ===\n")
cat("✓ Engineered features (bilateral asymmetry + front/hind coordination)\n")
cat("✓ Conservative feature selection (30 features)\n")
cat("✓ Leave-One-Mouse-Out cross-validation\n")

# ============================================================================
# DATA LOADING
# ============================================================================

script_dir <- tryCatch({
  dirname(rstudioapi::getActiveDocumentContext()$path)
}, error = function(e) {
  tryCatch({
    dirname(sys.frame(1)$ofile)
  }, error = function(e) {
    getwd()
  })
})

possible_data_dirs <- c(
  file.path(script_dir, "Motorater Output Files"),
  file.path(getwd(), "Motorater Output Files"),
  "Motorater Output Files"
)

data_dir <- NULL
for (dir_path in possible_data_dirs) {
  if (dir.exists(dir_path)) {
    data_dir <- dir_path
    break
  }
}

if (is.null(data_dir)) {
  all_files <- list.files(script_dir, pattern = "_out\\.(xlsx|csv)$", 
                          recursive = TRUE, full.names = TRUE)
  if (length(all_files) > 0) {
    data_dir <- dirname(all_files[1])
  } else {
    stop("ERROR: Cannot find data files")
  }
}

output_dir <- dirname(data_dir)
setwd(output_dir)
cat("Data directory:", data_dir, "\n\n")

read_motorater_file <- function(file_path) {
  file_name <- basename(file_path)
  parts <- str_split(file_name, "_", simplify = TRUE)
  if (ncol(parts) < 4) return(NULL)
  
  date <- parts[1]
  mouse_id <- parts[2]
  timepoint <- parts[3]
  trial_num <- str_extract(parts[4], "\\d+")
  
  data <- NULL
  
  if (str_detect(file_path, "\\.xlsx$")) {
    tryCatch({
      sheets <- excel_sheets(file_path)
      if ("Calculated Data" %in% sheets) {
        data <- read_excel(file_path, sheet = "Calculated Data", 
                           col_types = "guess", .name_repair = "minimal")
      } else if (length(sheets) > 0) {
        data <- read_excel(file_path, sheet = sheets[1], 
                           col_types = "guess", .name_repair = "minimal")
      }
    }, error = function(e) NULL)
  } else if (str_detect(file_path, "\\.csv$")) {
    tryCatch({
      data <- read_csv(file_path, show_col_types = FALSE, 
                       guess_max = 1000, name_repair = "minimal")
    }, error = function(e) NULL)
  }
  
  if (is.null(data) || nrow(data) == 0) return(NULL)
  
  names(data) <- make.names(names(data), unique = TRUE)
  numeric_cols <- setdiff(names(data), c("Time", "time", "TIME"))
  
  for (col in numeric_cols) {
    if (col %in% names(data)) {
      data[[col]] <- suppressWarnings(as.numeric(as.character(data[[col]])))
    }
  }
  
  time_col <- names(data)[tolower(names(data)) %in% c("time", "Time", "TIME")]
  if (length(time_col) == 0) return(NULL)
  if (time_col[1] != "Time") names(data)[names(data) == time_col[1]] <- "Time"
  
  data$Time <- suppressWarnings(as.numeric(data$Time))
  data <- data %>% dplyr::filter(!is.na(Time))
  if (nrow(data) < 10) return(NULL)
  
  data <- data %>%
    dplyr::mutate(
      date = date,
      mouseid = mouse_id,
      timepoint = timepoint,
      trial = trial_num,
      filename = file_name
    )
  
  return(data)
}

cat("Loading files...\n")
files <- dir_ls(data_dir, regexp = "\\.(xlsx|csv)$")
all_data_list <- list()

for (i in seq_along(files)) {
  result <- tryCatch(read_motorater_file(files[i]), error = function(e) NULL)
  if (!is.null(result)) {
    all_data_list[[i]] <- result
  }
}

all_data_list <- compact(all_data_list)
all_data <- bind_rows(all_data_list)

cat("Loaded:", nrow(all_data), "observations\n")

# ============================================================================
# FEATURE EXTRACTION WITH ENGINEERING
# ============================================================================

cat("\n=== EXTRACTING FEATURES ===\n")

extract_raw_features <- function(trial_data) {
  meta_cols <- c("Time", "date", "mouseid", "timepoint", "trial", "filename")
  gait_cols <- setdiff(names(trial_data), meta_cols)
  gait_cols <- gait_cols[sapply(trial_data[gait_cols], is.numeric)]
  
  features <- list()
  
  for (col in gait_cols) {
    signal <- trial_data[[col]]
    signal <- signal[!is.na(signal)]
    
    if (length(signal) < 5) next
    
    features[[paste0(col, "_mean")]] <- mean(signal)
    features[[paste0(col, "_sd")]] <- sd(signal)
    features[[paste0(col, "_median")]] <- median(signal)
    features[[paste0(col, "_min")]] <- min(signal)
    features[[paste0(col, "_max")]] <- max(signal)
    features[[paste0(col, "_range")]] <- max(signal) - min(signal)
    features[[paste0(col, "_cv")]] <- sd(signal) / (abs(mean(signal)) + 1e-10)
    features[[paste0(col, "_iqr")]] <- IQR(signal)
  }
  
  if (length(features) == 0) return(tibble())
  return(as_tibble(features))
}

cat("Processing trials...\n")
trial_features_list <- list()
trial_groups <- all_data %>%
  dplyr::group_by(mouseid, timepoint, trial) %>%
  dplyr::group_split()

for (i in seq_along(trial_groups)) {
  cat(".")
  features <- extract_raw_features(trial_groups[[i]])
  if (nrow(features) > 0) {
    features$mouseid <- trial_groups[[i]]$mouseid[1]
    features$timepoint <- trial_groups[[i]]$timepoint[1]
    features$trial <- trial_groups[[i]]$trial[1]
    trial_features_list[[i]] <- features
  }
}

trial_features <- bind_rows(trial_features_list)
cat("\n")

# Engineer asymmetry and coordination features
cat("Engineering asymmetry & coordination features...\n")

all_features <- names(trial_features)[sapply(trial_features, is.numeric)]
engineered_features_list <- list()

for (i in 1:nrow(trial_features)) {
  row <- trial_features[i, ]
  eng_feats <- list()
  
  # Left-right asymmetry
  left_features <- all_features[grepl("Left", all_features)]
  
  for (left_feat in left_features) {
    right_feat <- gsub("Left", "Right", left_feat)
    
    if (left_feat %in% names(row) && right_feat %in% names(row)) {
      left_val <- row[[left_feat]]
      right_val <- row[[right_feat]]
      
      base_name <- gsub("_Left", "", left_feat)
      
      # Absolute asymmetry
      eng_feats[[paste0(base_name, "_LR_abs_diff")]] <- abs(left_val - right_val)
      
      # Percent asymmetry
      avg_val <- (abs(left_val) + abs(right_val)) / 2
      if (avg_val > 1e-10) {
        eng_feats[[paste0(base_name, "_LR_pct_diff")]] <- 
          100 * abs(left_val - right_val) / avg_val
      }
    }
  }
  
  # Front-hind coordination
  front_features <- all_features[grepl("Front|Fore", all_features, ignore.case = TRUE)]
  
  for (front_feat in front_features) {
    base_name <- gsub("Front|Fore", "", front_feat, ignore.case = TRUE)
    base_name <- gsub("paw", "", base_name, ignore.case = TRUE)
    
    possible_hinds <- all_features[grepl(paste0("Hind.*", base_name), all_features, ignore.case = TRUE)]
    
    for (hind_feat in possible_hinds) {
      if (front_feat %in% names(row) && hind_feat %in% names(row)) {
        front_val <- row[[front_feat]]
        hind_val <- row[[hind_feat]]
        
        eng_feats[[paste0("FH_diff_", gsub("_mean|_sd|_median", "", front_feat))]] <- 
          front_val - hind_val
      }
    }
  }
  
  engineered_features_list[[i]] <- as_tibble(eng_feats)
}

engineered_features <- bind_rows(engineered_features_list)
trial_features_combined <- bind_cols(trial_features, engineered_features)

cat("Total features:", sum(sapply(trial_features_combined, is.numeric)), "\n")

# ============================================================================
# AGGREGATE AND PREPROCESS
# ============================================================================

cat("\n=== PREPROCESSING ===\n")

numeric_cols <- names(trial_features_combined)[sapply(trial_features_combined, is.numeric)]

aggregated_features <- trial_features_combined %>%
  dplyr::group_by(mouseid, timepoint) %>%
  dplyr::summarise(
    across(all_of(numeric_cols), ~mean(., na.rm = TRUE)),
    .groups = "drop"
  )

# Collapse timepoints
aggregated_features <- aggregated_features %>%
  dplyr::mutate(
    timepoint_original = timepoint,
    timepoint_group = dplyr::case_when(
      timepoint %in% c("PreSCI") ~ "Baseline",
      timepoint %in% c("7DPI", "14DPI") ~ "Early",
      timepoint %in% c("21DPI", "28DPI") ~ "Mid",
      timepoint %in% c("35DPI", "43DPI") ~ "Late",
      TRUE ~ "Unknown"
    ),
    timepoint_group = factor(timepoint_group, 
                             levels = c("Baseline", "Early", "Mid", "Late"))
  )

# Baseline normalization
if (NORMALIZE_TO_BASELINE) {
  mice_with_baseline <- aggregated_features %>%
    dplyr::filter(timepoint == "PreSCI") %>%
    dplyr::pull(mouseid) %>%
    unique()
  
  if (length(mice_with_baseline) > 0) {
    aggregated_features <- aggregated_features %>%
      dplyr::filter(mouseid %in% mice_with_baseline)
    
    baseline_values <- aggregated_features %>%
      dplyr::filter(timepoint == "PreSCI") %>%
      dplyr::select(mouseid, all_of(numeric_cols))
    
    aggregated_features <- aggregated_features %>%
      dplyr::left_join(baseline_values, by = "mouseid", suffix = c("", "_baseline"))
    
    for (col in numeric_cols) {
      baseline_col <- paste0(col, "_baseline")
      if (baseline_col %in% names(aggregated_features)) {
        aggregated_features[[col]] <- 
          ((aggregated_features[[col]] - aggregated_features[[baseline_col]]) / 
             (abs(aggregated_features[[baseline_col]]) + 1e-10)) * 100
        aggregated_features[[col]] <- pmax(pmin(aggregated_features[[col]], 500), -500)
      }
    }
    
    aggregated_features <- aggregated_features %>%
      dplyr::select(mouseid, timepoint, timepoint_original, timepoint_group, all_of(numeric_cols))
  }
}

# Prepare feature matrix
feature_matrix <- as.matrix(aggregated_features[, numeric_cols])
feature_matrix[is.na(feature_matrix)] <- 0
feature_matrix[is.infinite(feature_matrix)] <- 0

col_vars <- apply(feature_matrix, 2, var)
feature_matrix <- feature_matrix[, col_vars > 1e-10]

feature_matrix_scaled <- scale(feature_matrix)
feature_matrix_scaled[is.na(feature_matrix_scaled)] <- 0
feature_matrix_scaled[is.infinite(feature_matrix_scaled)] <- 0

metadata <- aggregated_features %>% 
  dplyr::select(mouseid, timepoint_group, timepoint_original)

cat("Feature matrix:", nrow(feature_matrix_scaled), "samples x", 
    ncol(feature_matrix_scaled), "features\n")

# ============================================================================
# FEATURE SELECTION (if enabled)
# ============================================================================

if (USE_RFE) {
  cat("\n=== FEATURE SELECTION ===\n")
  
  set.seed(42)
  
  # Train RF for feature ranking
  rf_initial <- randomForest(
    x = as.data.frame(feature_matrix_scaled),
    y = as.factor(metadata$timepoint_group),
    ntree = 500,
    importance = TRUE
  )
  
  feat_imp <- importance(rf_initial)
  feat_ranking <- data.frame(
    feature = rownames(feat_imp),
    importance = feat_imp[, "MeanDecreaseGini"]
  ) %>%
    dplyr::arrange(desc(importance))
  
  # Select top features
  n_features_rfe <- min(RFE_N_FEATURES, nrow(feat_ranking))
  selected_features <- feat_ranking$feature[1:n_features_rfe]
  
  cat("Selected", n_features_rfe, "features\n")
  
  # Update feature matrix
  feature_matrix_scaled <- feature_matrix_scaled[, selected_features]
} else {
  # Use all features
  feat_ranking <- data.frame(
    feature = colnames(feature_matrix_scaled),
    importance = NA
  )
}

# ============================================================================
# CROSS-VALIDATED ANALYSIS
# ============================================================================

cat("\n=== LEAVE-ONE-MOUSE-OUT CROSS-VALIDATION ===\n")

mice <- unique(metadata$mouseid)
n_mice <- length(mice)

# Storage for LOMO results
lomo_rf_acc <- numeric(n_mice)
lomo_lda_acc <- numeric(n_mice)
lomo_predictions_rf <- list()
lomo_predictions_lda <- list()

for (i in seq_along(mice)) {
  test_mouse <- mice[i]
  cat(sprintf("Fold %d/%d: Holding out mouse %s...\n", i, n_mice, test_mouse))
  
  train_idx <- metadata$mouseid != test_mouse
  test_idx <- metadata$mouseid == test_mouse
  
  # Random Forest
  rf_fold <- tryCatch({
    randomForest(
      x = as.data.frame(feature_matrix_scaled[train_idx, ]),
      y = as.factor(metadata$timepoint_group[train_idx]),
      ntree = 500
    )
  }, error = function(e) NULL)
  
  if (!is.null(rf_fold)) {
    rf_pred <- predict(rf_fold, as.data.frame(feature_matrix_scaled[test_idx, ]))
    lomo_rf_acc[i] <- mean(rf_pred == metadata$timepoint_group[test_idx])
    lomo_predictions_rf[[i]] <- data.frame(
      mouse = test_mouse,
      actual = as.character(metadata$timepoint_group[test_idx]),
      predicted = as.character(rf_pred)
    )
  }
  
  # LDA
  lda_fold <- tryCatch({
    MASS::lda(feature_matrix_scaled[train_idx, ], 
              grouping = metadata$timepoint_group[train_idx])
  }, error = function(e) NULL)
  
  if (!is.null(lda_fold)) {
    lda_pred <- predict(lda_fold, feature_matrix_scaled[test_idx, ])$class
    lomo_lda_acc[i] <- mean(lda_pred == metadata$timepoint_group[test_idx])
    lomo_predictions_lda[[i]] <- data.frame(
      mouse = test_mouse,
      actual = as.character(metadata$timepoint_group[test_idx]),
      predicted = as.character(lda_pred)
    )
  }
}

cat("\n")
cat("LOMO Random Forest Accuracy:", round(mean(lomo_rf_acc) * 100, 1), "%\n")
cat("  (Range:", round(min(lomo_rf_acc) * 100, 1), "-", 
    round(max(lomo_rf_acc) * 100, 1), "%)\n")

cat("LOMO LDA Accuracy:", round(mean(lomo_lda_acc) * 100, 1), "%\n")
cat("  (Range:", round(min(lomo_lda_acc) * 100, 1), "-", 
    round(max(lomo_lda_acc) * 100, 1), "%)\n")

# ============================================================================
# FINAL MODEL FOR VISUALIZATION
# ============================================================================

cat("\n=== TRAINING FINAL MODEL ===\n")

# Train on all data for visualization purposes only
rf_final <- randomForest(
  x = as.data.frame(feature_matrix_scaled),
  y = as.factor(metadata$timepoint_group),
  ntree = 500,
  importance = TRUE
)

# UMAP
n_neighbors_actual <- max(2, min(UMAP_N_NEIGHBORS, nrow(feature_matrix_scaled) - 1))

umap_result <- umap(feature_matrix_scaled,
                    n_neighbors = n_neighbors_actual,
                    min_dist = UMAP_MIN_DIST,
                    n_components = 2)

umap_scores <- as.data.frame(umap_result$layout)
colnames(umap_scores) <- c("UMAP1", "UMAP2")

# K-means
n_clusters <- n_distinct(metadata$timepoint_group)
kmeans_result <- kmeans(umap_scores, centers = n_clusters, nstart = 50)
sil <- silhouette(kmeans_result$cluster, dist(umap_scores))

# LDA for visualization
lda_final <- tryCatch({
  MASS::lda(feature_matrix_scaled, grouping = metadata$timepoint_group)
}, error = function(e) NULL)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

cat("\n=== CREATING VISUALIZATIONS ===\n")

plot_data <- bind_cols(
  metadata,
  umap_scores,
  cluster = as.factor(kmeans_result$cluster)
)

if (!is.null(lda_final)) {
  lda_scores <- as.data.frame(predict(lda_final)$x)
  plot_data <- bind_cols(plot_data, lda_scores[, 1:min(2, ncol(lda_scores))])
}

timepoint_colors <- brewer.pal(4, "Set1")
cluster_colors <- brewer.pal(n_clusters, "Set2")

# Plot 1: UMAP by recovery stage
p1 <- ggplot(plot_data, aes(x = UMAP1, y = UMAP2, color = timepoint_group, label = mouseid)) +
  geom_point(size = 4, alpha = 0.8) +
  geom_text_repel(size = 2.5, max.overlaps = 10) +
  scale_color_manual(values = timepoint_colors, name = "Recovery Stage") +
  labs(title = "UMAP - Recovery Stages",
       subtitle = sprintf("Engineered features (n=%d)", ncol(feature_matrix_scaled))) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "right",
        plot.title = element_text(face = "bold"))

# Plot 2: LDA with LOMO accuracy
if (!is.null(lda_final) && ncol(lda_scores) >= 2) {
  p2 <- ggplot(plot_data, aes(x = LD1, y = LD2, color = timepoint_group, label = mouseid)) +
    geom_point(size = 4, alpha = 0.8) +
    geom_text_repel(size = 2.5, max.overlaps = 10) +
    scale_color_manual(values = timepoint_colors, name = "Recovery Stage") +
    labs(title = "Linear Discriminant Analysis",
         subtitle = sprintf("LOMO CV Accuracy: %.1f%%", mean(lomo_lda_acc) * 100)) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "right",
          plot.title = element_text(face = "bold"))
} else {
  p2 <- ggplot() + theme_void()
}

# Plot 3: Confusion matrix
confusion_df <- plot_data %>%
  dplyr::group_by(timepoint_group, cluster) %>%
  dplyr::summarise(count = n(), .groups = "drop")

p3 <- ggplot(confusion_df, aes(x = cluster, y = timepoint_group, fill = count)) +
  geom_tile(color = "white", linewidth = 1.5) +
  geom_text(aes(label = count), color = "white", size = 7, fontface = "bold") +
  scale_fill_gradient(low = "lightblue", high = "darkblue", name = "Count") +
  labs(title = "Unsupervised Clusters vs Recovery Stages",
       subtitle = sprintf("Silhouette: %.3f", mean(sil[, 3]))) +
  theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank(),
        plot.title = element_text(face = "bold"))

# Plot 4: Top features with type annotation
feat_imp_final <- importance(rf_final)
top_features_df <- data.frame(
  feature = rownames(feat_imp_final),
  importance = feat_imp_final[, "MeanDecreaseGini"]
) %>%
  dplyr::arrange(desc(importance)) %>%
  dplyr::slice_head(n = 15) %>%
  dplyr::mutate(
    feature_type = dplyr::case_when(
      grepl("LR_", feature) ~ "Left-Right Asymmetry",
      grepl("FH_", feature) ~ "Front-Hind Coordination",
      grepl("cv|stability", feature, ignore.case = TRUE) ~ "Variability/Stability",
      TRUE ~ "Raw Gait Metric"
    )
  )

p4 <- ggplot(top_features_df, aes(x = reorder(feature, importance), y = importance, fill = feature_type)) +
  geom_bar(stat = "identity", width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("Left-Right Asymmetry" = "#E41A1C",
                               "Front-Hind Coordination" = "#377EB8",
                               "Variability/Stability" = "#4DAF4A",
                               "Raw Gait Metric" = "#984EA3"),
                    name = "Feature Type") +
  labs(title = "Top 15 Most Important Features",
       subtitle = sprintf("RF LOMO Accuracy: %.1f%%", mean(lomo_rf_acc) * 100),
       x = "", y = "Importance (Gini)") +
  theme_minimal(base_size = 10) +
  theme(axis.text.y = element_text(size = 8),
        legend.position = "bottom",
        plot.title = element_text(face = "bold"))

combined <- (p1 | p2) / (p3 | p4)

ggsave("results.pdf", combined, width = 18, height = 14)

# ============================================================================
# SAVE RESULTS
# ============================================================================

cat("\nSaving results...\n")

write_csv(plot_data, "results.csv")
write_csv(feat_ranking, "feature_rankings.csv")

# LOMO predictions
lomo_predictions_rf_df <- bind_rows(lomo_predictions_rf)
lomo_predictions_lda_df <- bind_rows(lomo_predictions_lda)

write_csv(lomo_predictions_rf_df, "lomo_predictions_rf.csv")
write_csv(lomo_predictions_lda_df, "lomo_predictions_lda.csv")

# Feature type summary
feature_summary <- data.frame(
  feature = rownames(feat_imp_final),
  importance = feat_imp_final[, "MeanDecreaseGini"]
) %>%
  dplyr::arrange(desc(importance)) %>%
  dplyr::mutate(
    rank = row_number(),
    feature_type = dplyr::case_when(
      grepl("LR_", feature) ~ "Left-Right Asymmetry",
      grepl("FH_", feature) ~ "Front-Hind Coordination",
      grepl("cv|stability", feature, ignore.case = TRUE) ~ "Variability/Stability",
      TRUE ~ "Raw Gait Metric"
    )
  ) %>%
  dplyr::group_by(feature_type) %>%
  dplyr::summarise(
    total_count = n(),
    in_top_10 = sum(rank <= 10),
    in_top_20 = sum(rank <= 20),
    mean_importance = mean(importance),
    .groups = "drop"
  ) %>%
  dplyr::arrange(desc(mean_importance))

write_csv(feature_summary, "feature_type_summary.csv")

# Performance summary
performance <- tibble(
  Method = c("Random Forest (LOMO CV)", "LDA (LOMO CV)", "Unsupervised K-means"),
  Accuracy = c(sprintf("%.1f%% (%.1f-%.1f%%)", 
                       mean(lomo_rf_acc) * 100, 
                       min(lomo_rf_acc) * 100, 
                       max(lomo_rf_acc) * 100),
               sprintf("%.1f%% (%.1f-%.1f%%)", 
                       mean(lomo_lda_acc) * 100, 
                       min(lomo_lda_acc) * 100, 
                       max(lomo_lda_acc) * 100),
               sprintf("Silhouette: %.3f", mean(sil[, 3]))),
  Notes = c("Out-of-sample", "Out-of-sample", "Cluster quality")
)

write_csv(performance, "performance_summary.csv")

cat("\n", rep("=", 75), "\n", sep="")
cat("RESULTS COMPLETE\n")
cat(rep("=", 75), "\n", sep="")

cat("\nFEATURE TYPE DISTRIBUTION:\n")
print(feature_summary)

cat("\nPERFORMANCE SUMMARY:\n")
print(performance)


cat("\n", rep("=", 75), "\n", sep="")
