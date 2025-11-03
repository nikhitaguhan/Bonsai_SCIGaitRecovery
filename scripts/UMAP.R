library(readxl)
library(dplyr)
library(recipes)
library(embed)
library(ggplot2)
# ============================================================
# --- Load data ---
path <- "combined_avg_features_per_run.csv"

df <- read.csv(path) %>%
  mutate(condition = factor(condition,
                            levels = c("PreSCI", "7DPI", "14DPI", "21DPI",
                                       "28DPI", "35DPI", "43DPI"))) %>%
  # mean-impute within each animal_id × condition group
  # group_by(condition, animal_id) %>%
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

# Method A - cohen's d 
# select_res <- feat_select(df, 1, 0.8, 0.1, 1.5)
# selected_features <- select_res$select_feats
# ============================================================
# --- Feature selection methods B and C: variance based filtering ---
# Method B - variance based, best results
# var_threshold <- 1e-8
# selected_features <- df %>%
#   dplyr::select(where(is.numeric)) %>%
#   dplyr::select(-animal_id) %>%
#   dplyr::select(where(~ var(.) > var_threshold)) %>%
#   colnames()

# Method C - variance across conditions
feature_cols <- df %>%
  dplyr::select(where(is.numeric), -animal_id)

var_threshold <- 1e-8
feature_variance <- df %>%
  group_by(condition) %>%
  summarise(across(all_of(names(feature_cols)), mean)) %>%
  summarise(across(where(is.numeric), var))
selected_features <- names(feature_cols)[unlist(feature_variance) > var_threshold]

selected_features
# ============================================================
# --- Recipe for preprocessing + UMAP ---
gait_rec <- recipe(~ ., data = df %>% dplyr::select(all_of(selected_features), animal_id, condition)) %>%
  # Treat IDs / metadata as non-predictors
  update_role(animal_id, condition, new_role = "id") %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  # step_log(all_numeric_predictors(), offset=1) %>%
  # Scale/center numeric features
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.4) %>%
  # Apply UMAP on numeric features
  step_umap(all_numeric_predictors(), neighbors = 10, min_dist = 0.05, metric = "cosine")
# ============================================================
# --- Prep the recipe + plot UMAP colored by condition ---
gait_prep <- prep(gait_rec)

juice(gait_prep) %>%
  ggplot(aes(UMAP1, UMAP2, label = animal_id)) +
  geom_point(aes(color = factor(condition)), alpha = 0.7, size = 3) +
  # geom_text(hjust = "inward", size=3) + # If want to label by animal ID
  labs(color = "Condition")
