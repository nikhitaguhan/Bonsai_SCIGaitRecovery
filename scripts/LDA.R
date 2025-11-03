library(readr)
library(dplyr)
library(bestNormalize)
library(MASS)
library(ggplot2)
library(tidyr)
library(patchwork)
library(effectsize)
# ============================================================
# --- Load ---
path <- "combined_avg_features_per_run.csv"
df <- read.csv(path) %>%
  mutate(condition = factor(condition,
                            levels = c("PreSCI", "7DPI", "14DPI", "21DPI",
                                       "28DPI", "35DPI", "43DPI"))) %>%
  # mean-impute within each animal_id × condition group
  # group_by(animal_id, condition) %>%
  # mutate(across(where(is.numeric),
  #               ~ ifelse(is.na(.x), mean(.x, na.rm = TRUE), .x))) %>%
  # ungroup() %>%
  # drop any remaining NAs (if group mean couldn't be computed)
  drop_na() %>%
  dplyr::select(-run_number, -run_time, -matches("regularity_score"))
# ============================================================
# --- Feature selection method A ---
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
# --- Function: run LDA ---
compute_lda <- function(data, select_feats, pca_var, group_animal = FALSE) {
  
  if(group_animal) {
    data <- data %>%
      group_by(animal_id, condition) %>%
      summarise(across(where(is.numeric), mean), .groups = "drop")
  }
  
  # Filter numeric features
  # select_feats <- data %>%
  #   dplyr::select(where(is.numeric)) %>%
  #   dplyr::select(-animal_id) %>%
  #   dplyr::select(where(~ var(.) > feat_var)) %>%
  #   colnames()
  
  # select_res <- feat_select(df, injury_p = select_p, injury_d = select_d)
  # select_feats <- select_res$select_feats
  
  X <- data[, select_feats, drop = FALSE]
  
  yeo_params <- lapply(X, function(x) yeojohnson(x, standardize = FALSE))
  X <- as.data.frame(Map(predict, yeo_params, X))
  
  scale_center <- sapply(X, mean)
  scale_sd <- sapply(X, sd)
  X <- scale(X, center = scale_center, scale = scale_sd)
  
  # PCA (optional) + LDA
  if (!is.null(pca_var)) {
    # PCA
    pca_res <- prcomp(X, center = FALSE, scale. = FALSE)
    cum_var <- cumsum(pca_res$sdev^2 / sum(pca_res$sdev^2))
    k <- which(cum_var >= pca_var)[1]
    if (is.na(k)) k <- length(cum_var)
    # print(k)
    # View(pca_res$x[, 1:k, drop = FALSE])
    pca_scores_k <- as.data.frame(pca_res$x[, 1:k, drop = FALSE])
    lda_res <- lda(data$condition ~ ., data = pca_scores_k)
  } else {
    # No PCA
    lda_res <- lda(data$condition ~ ., data = as.data.frame(X))
  }
  
  lda_pred <- predict(lda_res)
  lda_scores <- as.data.frame(lda_pred$x)
  lda_scores$condition <- data$condition
  lda_scores$animal_id <- data$animal_id
  
  # LDA finds directions/separation so flip for interpretation if needed
  ref_mean <- mean(lda_scores$LD1[lda_scores$condition == "PreSCI"])
  dpi7_mean <- mean(lda_scores$LD1[lda_scores$condition == "7DPI"])
  
  if(ref_mean < dpi7_mean) {
    lda_scores$LD1 <- -lda_scores$LD1
    lda_res$scaling[, "LD1"] <- -lda_res$scaling[, "LD1"]
  }
  
  # if(group_animal) {
  #   lda_scores <- lda_scores %>%
  #     group_by(animal_id, condition) %>%
  #     summarise(across(where(is.numeric), mean), .groups = "drop")
  # }
  
  pca_rotation = if (!is.null(pca_var)) {
    pca_res$rotation[, 1:k, drop = FALSE]
  } else {
    id <- diag(ncol(X))
    colnames(id) <- colnames(X)
    rownames(id) <- colnames(X)
    id
  }
  
  return(list(
    numeric_cols = select_feats, 
    yeo_params = yeo_params,          # will use to generalize a formula to predict LD1 (line 221)
    scale_center = scale_center,      # will use to generalize a formula to predict LD1 (line 222)
    scale_scale = scale_sd,           # will use to generalize a formula to predict LD1 (line 222)
    pca_rotation = pca_rotation,
    scaling = lda_res$scaling,
    lda_scores = lda_scores
  ))
}
# ============================================================
# --- Function: score LDA config (for testing what parameters give trend we want) ---
score_config <- function(condition_means) {
  LD1 <- condition_means$LD1
  cond <- condition_means$condition
  
  pre <- LD1[cond == "PreSCI"]
  dpi7 <- LD1[cond == "7DPI"]
  dpi14 <- LD1[cond == "14DPI"]
  dpi21 <- LD1[cond == "21DPI"]
  dpi28 <- LD1[cond == "28DPI"]
  dpi35 <- LD1[cond == "35DPI"]
  dpi43 <- LD1[cond == "43DPI"]
  
  ld1_vector <- c(pre, dpi7, dpi14, dpi21, dpi28, dpi35, dpi43)
  
  # 1. Reward early deviation
  score <- abs(dpi7 - pre) / abs(pre) * 100
  print(pre)
  print(dpi7)
  
  # 2. Penalize drops in recovery
  recovery <- c(dpi14, dpi21, dpi28, dpi35, dpi43)
  print(recovery)
  monotonic_penalty <- sum(pmax(0, recovery[-length(recovery)] - recovery[-1]))
  
  # 3. Penalize going over baseline at 43DPI
  overshoot_penalty <- max(0, dpi43 - pre)
  
  # Final "score" - higher indicates more of trend we want, but not perfect
  score_final <- score - monotonic_penalty - overshoot_penalty
  
  return(list(
    score = score_final,
    ld1_vector = ld1_vector
  ))
}
# ============================================================
# -- Function: run & score LDA configs for feature selection Method A parameters
# (optimized for p, d, and PCA tuning) ---
tune_lda_configs <- function(df, group_animal = FALSE,
                             injury_d_vals = seq(0.3, 1.3, 0.1),
                             recovery_d_vals = seq(0.3, 1.3, 0.1),
                             pca_var_vals = c(list(NULL), as.list(seq(0.9, 1, 0.01)))) {

  results <- list()

  for (d1 in injury_d_vals) {
    for (d2 in recovery_d_vals) {

      # Select features once per (p, d)
      select_res <- feat_select(df, injury_d = d1, recovery_d = d2)

      for (pv in pca_var_vals) {
        df_use <- df

        if (group_animal) {
          df_use <- df_use %>%
            group_by(animal_id, condition) %>%
            summarise(across(where(is.numeric), mean), .groups = "drop")
        }

        lda_result <- tryCatch(
          compute_lda(df_use, select_feats = select_res$select_feats, pca_var = pv),
          error = function(e) {
            message("Skipping config: injury_d=", d1,
                    ", recovery_d=", d2, ", pca_var=", pv,
                    " due to error: ", e$message)
            return(NULL)
          }
        )
        # Skip if lda_result failed -
        # happens sometimes when using all principal components (pca_var = 1) when
        # least sig. PC components have no variance btwn rows
        if (is.null(lda_result)) next

        lda_scores <- lda_result$lda_scores

        # Mean per condition
        condition_means <- lda_scores %>%
          group_by(condition) %>%
          summarise(across(where(is.numeric), mean))

        print(paste("Checking", d1, d2, pv))
        # Compute score
        s_res <- score_config(condition_means)
        results[[length(results) + 1]] <- tibble(
          injury_d = d1,
          recovery_d = d2,
          pca_var = pv,
          score = s_res$score,
          ld1_vector = list(s_res$ld1_vector),  # list-column
          n_feats = length(select_res$select_feats)
        )
      }
    }
  }

  # Combine and rank
  results_df <- bind_rows(results)
  results_df <- results_df %>% arrange(desc(score))
  results_df <- results_df %>%
    mutate(row_id = 1:n())

  return(results_df)
}
# ============================================================
# -- Function: run & score LDA configs for feature selection Method B (diff. variances + PCA) ---
tune_lda_configs <- function(df, group_animal = FALSE,
                             feat_vars = 10^seq(-3, -9, by = -1),
                             pca_var_vals = c(list(NULL), as.list(seq(0.5, 0.9, 0.05)), as.list(seq(0.9, 1, 0.01)))) {

  results <- list()

  for (var in feat_vars) {

    feats <- df %>%
      dplyr::select(where(is.numeric)) %>%
      dplyr::select(-animal_id) %>%
      dplyr::select(where(~ var(.) > var)) %>%
      colnames()

    for (pv in pca_var_vals) {
      df_use <- df

      if (group_animal) {
        df_use <- df_use %>%
          group_by(animal_id, condition) %>%
          summarise(across(where(is.numeric), mean), .groups = "drop")
      }

      lda_result <- tryCatch(
        compute_lda(df_use, select_feats = feats, pca_var = pv),
        error = function(e) {
          message("Skipping config: feat_var", var, ", pca_var=", pv,
                  " due to error: ", e$message)
          return(NULL)
        }
      )
      # Skip if lda_result failed -
      # happens sometimes when using all principal components (pca_var = 1) when
      # least sig. PC components have no variance btwn rows
      if (is.null(lda_result)) next

      lda_scores <- lda_result$lda_scores

      # Mean per condition
      condition_means <- lda_scores %>%
        group_by(condition) %>%
        summarise(across(where(is.numeric), mean))

      print(paste("Checking", var, pv))
      # Compute score
      s_res <- score_config(condition_means)
      results[[length(results) + 1]] <- tibble(
        feat_var = var,
        pca_var = pv,
        score = s_res$score,
        ld1_vector = list(s_res$ld1_vector),  # list-column
        n_feats = length(select_res$select_feats)
        )
      }
    }

    # Combine and rank
    results_df <- bind_rows(results)
    results_df <- results_df %>% arrange(desc(score))
    results_df <- results_df %>%
      mutate(row_id = 1:n())

    return(results_df)
}
# ============================================================
# -- Uncomment out to test diff parameter configs to optimize LDA ---
# top_configs <- tune_lda_configs(df)
# 
# top_configs_expanded <- top_configs %>%
#   unnest_wider(ld1_vector, names_sep = "_")
# 
# # Valid configurations
# top_configs_expanded <- top_configs_expanded %>% dplyr::filter(
#   ld1_vector_1 > ld1_vector_7,
#   ld1_vector_7 > ld1_vector_6,
#   ld1_vector_6 > ld1_vector_5,
#   ld1_vector_5 > ld1_vector_4,
#   ld1_vector_4 > ld1_vector_3,
#   ld1_vector_3 > ld1_vector_2,
#   ld1_vector_2 < ld1_vector_1)
# View(top_configs_expanded)
# ============================================================
# --- Function: take 1 obs (run) and compute LD1 score based on training model coefficients  ---
predict_ld1_score <- function(obs, lda_result, ld1_coeffs) {
  # Extract numeric feature columns used in the model
  feats <- lda_result$numeric_cols 
  x <- obs[, feats, drop = FALSE]
  
  # Apply stored transforms
  x_yeo <- as.data.frame(Map(predict, lda_result$yeo_params, x))
  x_scaled <- scale(x_yeo,
                    center = lda_result$scale_center,
                    scale = lda_result$scale_scale)
  
  # sig <- c(top, bottom)
  # x_scaled <- x_scaled[, names(ld1_coeffs)]
  
  # Apply LD1 "formula" and return score
  print(sum(x_scaled * ld1_coeffs))
}
# ============================================================
# --- Function: plot LD1
plot_lda <- function(lda_df, group_animal = FALSE, predicted = "") {
  
  p1 <- ggplot(lda_df, aes(x=condition, y=LD1, color=condition)) +
    geom_jitter(width=0.2, alpha=0.7, size=2.5) +
    stat_summary(fun=mean, geom="point", shape=18, size=4, color="black") +
    stat_summary(fun=mean, geom="line", aes(group=1), color="black", linewidth=1.1) +
    labs(title=paste(predicted, "LD1 across Recovery Stages (all runs)"),
         x="Condition", y="LD1 Score") +
    theme_minimal(base_size=14) +
    theme(legend.position="none")
  
  if(!group_animal) {
    return(p1)
  }
  
  lda_animal <- lda_df %>%
    group_by(animal_id, condition) %>%
    summarise(across(where(is.numeric), mean), .groups = "drop")
  
  lda_condition <- lda_df %>%
    group_by(condition) %>%
    summarise(across(where(is.numeric), mean), .groups = "drop") %>% 
    mutate(condition = factor(condition, levels = c("PreSCI", "7DPI", "14DPI", "21DPI", "28DPI", "35DPI", "43DPI")))
  
  p2 <- ggplot() +
    # per-animal lines
    geom_line(data = lda_animal, 
              aes(x = condition, y = LD1, group = animal_id, color = as.factor(animal_id)), size = 1, alpha = 0.4) +
    # geom_point(data = lda_animal, 
    #            aes(x = condition, y = LD1, color = as.factor(animal_id)), size = 2, alpha = 0.5) +
    # mean line with automatic legend entry
    geom_line(data = lda_condition,
              aes(x = condition, y = LD1, linetype = "Condition Average", group = 1), color = "black", linewidth=1.1) +
    geom_point(data = lda_condition,
               aes(x = condition, y = LD1, group = 1), color = "black", shape=18, size = 4) +
    labs(
      title = paste(predicted, "LD1 Across Recovery Stages (avg. by animal)"),
      x = "Condition",
      y = "LD1 Score",
      color = "Animal",
      linetype = "All Animals"
    ) +
    theme_minimal(base_size = 14) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major.x = element_blank())
  
  return(list(p1 = p1, p2 = p2))
  
}
# ============================================================
# # --- Select important features
# Method A
# select_res <- feat_select(df, 1, 0.3, 0, 0.6)

# Method B
feat_var <- 1e-4
feats <- df %>%
  dplyr::select(where(is.numeric)) %>%
  dplyr::select(-animal_id) %>%
  dplyr::select(where(~ var(.) > feat_var)) %>%
  colnames()
# ============================================================
# --- Compute + plot full LDA --- 
lda_full <- compute_lda(df, select_feats = feats, pca_var = 0.96)
lda_full_plots <- plot_lda(lda_full$lda_scores, group_animal = TRUE)
lda_full_plots$p1 | lda_full_plots$p2

lda_loadings <- as.matrix(lda_full$pca_rotation) %*% as.matrix(lda_full$scaling)
ld1_coeffs <- lda_loadings[, 1]

top_pos_feats <- head(sort(ld1_coeffs, decreasing=TRUE), 15)
top_neg_feats <- tail(sort(ld1_coeffs, decreasing=TRUE), 15)
top_pos_feats
top_neg_feats

# Plot LD1 loadings (top positive + negative contributing features)
lda_loadings_df <- data.frame(
  feature = names(ld1_coeffs),
  coefficient = ld1_coeffs
) %>%
  mutate(direction = ifelse(coefficient > 0, "Positive", "Negative"))

top_feats <- lda_loadings_df %>%
  group_by(direction) %>%
  slice_max(abs(coefficient), n = 10) %>%
  ungroup() %>%
  group_by(direction) %>%
  mutate(feature = factor(feature, levels = feature[order(coefficient)])) %>%
  ungroup() %>%
  # Keep only relevant coefficients for each direction
  dplyr::filter((direction == "Positive" & coefficient > 0) |
           (direction == "Negative" & coefficient < 0)) %>%
  mutate(coefficient_plot = ifelse(direction == "Negative", -coefficient, coefficient))

ggplot(top_feats, aes(x = feature, y = coefficient_plot, fill = direction)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~ direction, scales = "free") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(title = "Top Features Contributing to LD1",
       x = "Feature",
       y = "LD1 Coefficient") +
  scale_fill_manual(values = c("Positive" = "steelblue", "Negative" = "tomato")) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")
# ============================================================
# --- Compute + plot LD1 predictions (leave one run out and get predictions)
ld1_predictions <- df %>%
  mutate(LD1 = NA_real_)

# Loop over each row in df to get leave one-out LD1 score (compute LD1 model without test row, get test row score)
for (i in 1:nrow(df)) {
  # Leave one row (run) out
  train_data <- df[-i, , drop = FALSE]
  test_row  <- df[i, , drop = FALSE]
  
  # Compute LDA on training data using function
  lda_res <- compute_lda(train_data, select_feats = feats, pca_var = 0.96, group_animal = FALSE)
  
  # Compute LD1 coefficients from the LDA training data results
  # Multiplying PCA weight x LDA weight to get coefficients for orig. features
  lda_loadings <- as.matrix(lda_res$pca_rotation) %*% as.matrix(lda_res$scaling)
  ld1_coeffs <- lda_loadings[, 1]
  
  # Predict LD1 for the left-out row
  ld1_predictions$LD1[i] <- predict_ld1_score(test_row, lda_res, ld1_coeffs)
  
  # Most significant features (positive + negative impact)
  top_pos_feats <- head(sort(ld1_coeffs, decreasing=TRUE), 10)
  top_neg_feats <- tail(sort(ld1_coeffs, decreasing=TRUE), 10)
  
  top_pos_feats
  top_neg_feats
}

lda_plots <- plot_lda(ld1_predictions, group_animal = TRUE, predicted = "Predicted")
lda_plots$p1 | lda_plots$p2