# --- Libraries ---
library(ggplot2)
library(dplyr)
library(readr)
library(scales)
library(bestNormalize)
library(patchwork)  # for side-by-side plots
library(tidyr)
library(ggrepel)
library(effectsize)
# ============================================================
# --- Load ---
path <- "combined_avg_features_per_run.csv"

df <- read.csv(path) %>%
  mutate(condition = factor(condition,
                            levels = c("PreSCI", "7DPI", "14DPI", "21DPI",
                                       "28DPI", "35DPI", "43DPI"))) %>%
  # mean-impute within each animal_id × condition group
  group_by(animal_id, condition) %>%
  mutate(across(where(is.numeric),
                ~ ifelse(is.na(.x), mean(.x, na.rm = TRUE), .x))) %>%
  ungroup() %>%
  # drop any remaining NAs (if group mean couldn't be computed)
  drop_na() %>%
  dplyr::select(-run_number, -run_time)
# ============================================================
# # --- Function: select features (method A, see line 97) ---
# https://www.eneuro.org/content/8/2/ENEURO.0497-20.2021
feat_select <- function(df, injury_p = 1, injury_d = 0.6, recovery_p = 0, recovery_d = 0.8) {
  
  # Found results better without using significance thresholds injury_p and recovery_p,
  # so just made them 1 and 0 so they would have no effect
  
  all_feats <- df %>%
    dplyr::select(where(is.numeric)) %>%
    dplyr::select(-animal_id) %>% 
    colnames()
  
  results <- lapply(all_feats, function(feat) {
    presci <- df %>% dplyr::filter(condition == "PreSCI") %>% pull(feat)
    dpi7   <- df %>% dplyr::filter(condition == "7DPI") %>% pull(feat)
    dpi43  <- df %>% dplyr::filter(condition == "43DPI") %>% pull(feat)
    
    # PreSCI vs 7DPI — looking for large change
    t_7 <- t.test(presci, dpi7, var.equal = FALSE)
    d_7 <- cohens_d(presci, dpi7, pooled_sd = FALSE)$Cohens_d
    
    # PreSCI vs 43DPI — looking for recovery (smaller change than PreSCI vs 7DPI)
    t_43 <- t.test(presci, dpi43, var.equal = FALSE)
    d_43 <- cohens_d(presci, dpi43, pooled_sd = FALSE)$Cohens_d
    
    tibble(
      feature = feat,
      p_7dpi = t_7$p.value,
      d_7dpi = d_7,
      p_43dpi = t_43$p.value,
      d_43dpi = d_43
    )
  }) %>% bind_rows()
  
  # Select features:
  # (1) PreSCI vs 7DPI are significantly different (injury effect)
  # (2) PreSCI vs 43DPI are not as different (recovery)
  selected <- results %>%
    dplyr::filter(
      p_7dpi < injury_p, abs(d_7dpi) > injury_d,
      p_43dpi > recovery_p, abs(d_43dpi) < recovery_d
    ) %>%
    arrange(p_7dpi)
  
  return(list(
    select_feats = selected$feature,
    stats = selected
  ))
}
# ============================================================
# Function: Run PCA with preprocessing and plot scree + cumulative variance
run_PCA <- function(df, var_threshold = 1e-4, group_by_animal = TRUE) {
  
  if (group_by_animal) {
    df <- df %>%
      group_by(animal_id, condition) %>%
      summarise(across(where(is.numeric), mean), .groups = "drop") %>%
      arrange(condition)
  }
  
  # Method A - using feat_select function
  # select_res <- feat_select(df, 1, 0.6, 0, 0.8)
  # selected_features <- select_res$select_feats
  
  # Select features
  # Method B - variance based, best results
  selected_features <- df %>%
    dplyr::select(where(is.numeric)) %>%
    dplyr::select(-animal_id) %>%
    dplyr::select(where(~ var(.) > var_threshold)) %>%
    colnames()

  # Method C - variance across conditions
  # feature_variance <- df %>%
  #   group_by(condition) %>%
  #   summarise(across(all_of(names(feature_cols)), mean)) %>%
  #   summarise(across(where(is.numeric), var))
  # selected_features <- names(feature_cols)[unlist(feature_variance) > var_threshold]
  
  print(selected_features)
  
  cat("Selected features:\n", paste(selected_features, collapse = ", "), "\n\n")
  
  # Transform and scale features
  X <- df %>%
    dplyr::select(all_of(selected_features)) %>% 
    mutate(across(everything(), ~ yeojohnson(.)$x.t)) %>%
    scale()
  
  # PCA
  pca_res <- prcomp(X, center = TRUE, scale. = TRUE)
  explained_var <- pca_res$sdev^2
  explained_var_ratio <- explained_var / sum(explained_var)
  
  # Scree + Cumulative variance plot
  scree_plot <- data.frame(Component = 1:length(explained_var), Eigenvalue = explained_var) %>%
    ggplot(aes(Component, Eigenvalue)) +
    geom_line() + geom_point() +
    labs(title = "Scree Plot", x = "Component", y = "Eigenvalue") +
    theme_minimal()
  
  cumvar_plot <- data.frame(Component = 1:length(explained_var_ratio), CumulativeVariance = cumsum(explained_var_ratio)) %>%
    ggplot(aes(Component, CumulativeVariance)) +
    geom_line() + geom_point() +
    scale_y_continuous(labels = percent_format()) +
    labs(title = "Cumulative Explained Variance", x = "Component", y = "Cumulative Explained Variance") +
    theme_minimal()
  
  print(scree_plot | cumvar_plot)
  
  # Return pca result + input df
  return(list(pca_res = pca_res, df = df))
}
# ============================================================
# Function: Inspect one principal component across recovery stage
inspect_PC <- function(pca_result, pc_num = 1, animal = NULL, grouped_animal=FALSE) {
  df <- pca_result$df
  pca_res <- pca_result$pca_res
  
  explained_var_ratio <- pca_res$sdev^2 / sum(pca_res$sdev^2)
  var_exp <- explained_var_ratio[pc_num]
  
  pc_df <- data.frame(
    animal_id = df$animal_id,
    condition = df$condition,
    score = pca_res$x[, pc_num]
  )
  
  # Filter if target_animal is specified
  if (!is.null(animal)) {
    pc_df <- pc_df %>% dplyr::filter(animal_id == animal)
  }
  
  pc_str = paste("PC", pc_num)
  
  # Return explained variance and loadings
  pc_loadings <- pca_res$rotation[, pc_num, drop = FALSE]
  pc_loadings <- pc_loadings[order(-abs(pc_loadings[, 1])), , drop = FALSE]
  
  p1 <- ggplot(pc_df, aes(x=condition, y=score, color=condition)) +
    geom_jitter(width=0.2, alpha=0.7, size=2.5) +
    stat_summary(fun=mean, geom="point", shape=18, size=4, color="black") +
    stat_summary(fun=mean, geom="line", aes(group=1), color="black", linewidth=1.1) +
    labs(title=paste(pc_str, "across Recovery Stages (all runs)"),
         x="Condition", y=paste(pc_str, "value")) +
    theme_minimal(base_size=14) +
    theme(legend.position="none")
  
  if(!grouped_animal) {
    return(list(
      plot = p1,
      explained_variance = var_exp,
      top_loadings = head(pc_loadings, 30)
    ))
  }
  
  pc_condition <- pc_df %>%
    group_by(condition) %>%
    summarise(across(where(is.numeric), mean), .groups = "drop") %>% 
    mutate(condition = factor(condition, levels = c("PreSCI", "7DPI", "14DPI", "21DPI", "28DPI", "35DPI", "43DPI")))
  
  # Plot
  p2 <- ggplot() +
    # per-animal lines
    geom_line(data = pc_df, 
              aes(x = condition, y = score, group = animal_id, color = as.factor(animal_id)), size = 1, alpha = 0.4) +
    # geom_point(data = lda_animal, 
    #            aes(x = condition, y = LD1, color = as.factor(animal_id)), size = 2, alpha = 0.5) +
    # mean line with automatic legend entry
    geom_line(data = pc_condition,
              aes(x = condition, y = score, linetype = "Condition Average", group = 1), color = "black", linewidth=1.1) +
    geom_point(data = pc_condition,
               aes(x = condition, y = score, group = 1), color = "black", shape=18, size = 4) +
    labs(
      title = paste(pc_str, "across Recovery Stages (avg. by animal)"),
      x = "Condition",
      y = paste(pc_str, "value"),
      color = "Animal",
      linetype = "All Animals"
    ) +
    theme_minimal(base_size = 14) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major.x = element_blank())
  
  
  return(list(
    plot = p2,
    explained_variance = var_exp,
    top_loadings = head(pc_loadings, 30)
  ))
}
# ============================================================
# Function: Plot 2D PCA scatter for two components
plot_PCA_scatter <- function(pca_result, pc_x = 1, pc_y = 2, label_animals = TRUE) {
  df <- pca_result$df
  pca_res <- pca_result$pca_res
  
  pca_df <- data.frame(
    PCX = pca_res$x[, pc_x],
    PCY = pca_res$x[, pc_y],
    condition = df$condition,
    animal_id = df$animal_id
  )
  
  p <- ggplot(pca_df, aes(PCX, PCY, color = condition)) +
    geom_point(alpha = 0.7, size = 2.5) +
    { if (label_animals) geom_text_repel(aes(label = animal_id), color = "black", size = 2.5) } +
    labs(
      title = paste0("PCA of Animal Gait Features (PC", pc_x, " vs PC", pc_y, ")"),
      x = paste0("PC", pc_x),
      y = paste0("PC", pc_y),
      color = "Condition"
    ) +
    theme_minimal()
  
  print(p)
}
# ============================================================
group_animal = FALSE
# 1. Run PCA
pca_result <- run_PCA(df, var_threshold = 1e-4, group_by_animal = group_animal)

# 2. Inspect one PC (variance + loadings)
pc_num = 1
pc_info <- inspect_PC(pca_result, pc_num = pc_num, grouped_animal = group_animal)
pc_info$plot
pc_info$explained_variance
# head(pc_info$top_loadings)
data.frame(pc_info$top_loadings) %>% arrange(paste0("PC", pc_num))

# Visual for loadings of Principal Component in step 2
# load_plot <- data.frame(
#   feature = rownames(pc_info$top_loadings),
#   loading = pc_info$top_loadings[, 1]
# ) %>%
#   mutate(abs_loading = abs(loading)) %>%
#   arrange(desc(abs_loading)) %>%
#   head(15) %>%
#   ggplot(aes(x = reorder(feature, loading), y = loading, fill = loading > 0)) +
#   geom_col(show.legend = FALSE) +
#   coord_flip() +
#   labs(
#     title = paste0("Top Loadings for PC", pc_num),
#     x = "Feature",
#     y = "Loading"
#   ) +
#   theme_minimal(base_size = 13)
# print(load_plot)

# 3. 2D scatter between two PCs
plot_PCA_scatter(pca_result, pc_x = 2, pc_y = 3, label_animals = FALSE)