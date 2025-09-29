library(readxl)
library(dplyr)
library(recipes)
library(embed)
library(ggplot2)

library(dplyr)
library(recipes)
library(embed)    # for step_umap()
library(ggplot2)

# --- Load data ---
# feats_by_animal <- read.csv("avg_features_per_animal.csv") %>%
#   mutate(condition = factor(condition, 
#                             levels = c("PreSCI", "7DPI", "14DPI", "21DPI", 
#                                        "28DPI", "35DPI", "43DPI"))) %>%
#   arrange(condition)

# feats_by_run <- read.csv("avg_features_per_run.csv") %>%
#   mutate(condition = factor(condition,
#                             levels = c("PreSCI", "7DPI", "14DPI", "21DPI",
#                                        "28DPI", "35DPI", "43DPI"))) %>%
#   arrange(condition)

cycle_feats_by_run <- read.csv("avg_cycle_features_per_run.csv") %>%
  separate(id, into = c("animal_id", "condition", "run_number"), sep = "_") %>%
  drop_na %>%
  mutate(condition = factor(condition,
                            levels = c("PreSCI", "7DPI", "14DPI", "21DPI",
                                       "28DPI", "35DPI", "43DPI"))) %>%
  arrange(condition)

# --- Separate metadata ---
meta <- cycle_feats_by_run %>% select(animal_id, condition)

# --- Variance filtering across conditions ---
feature_cols <- cycle_feats_by_run %>% select(-animal_id, -condition, -run_number)

# compute condition means per feature
condition_means <- cycle_feats_by_run %>%
  group_by(condition) %>%
  summarise(across(all_of(names(feature_cols)), mean)) %>%
  select(-condition)

# variance of each feature across condition means
feature_variance <- condition_means %>%
  summarise(across(everything(), var))

# keep only features with variance across conditions above threshold
threshold <- 1e-55
selected_features <- names(feature_variance)[feature_variance > threshold]

# --- Recipe for preprocessing + UMAP ---
gait_rec <- recipe(~ ., data = cycle_feats_by_run %>% select(all_of(selected_features), animal_id, condition)) %>%
  # Treat IDs / metadata as non-predictors
  update_role(animal_id, condition, new_role = "id") %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  # step_log(all_numeric_predictors(), offset=1) %>%
  # Scale/center numeric features
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.25) %>%
  # Apply UMAP on numeric features
  step_umap(all_numeric_predictors(), neighbors = 17, min_dist = 0.01, metric = "cosine")

# Baseline algorithm (euclidean), not as strong clustering but same color pattern
# gait_rec <- recipe(~ ., data = cycle_feats_by_run %>% select(all_of(selected_features), animal_id, condition)) %>%
#   # Treat IDs / metadata as non-predictors
#   # Treat IDs / metadata as non-predictors
#   update_role(animal_id, condition, new_role = "id") %>%
#   step_YeoJohnson(all_numeric_predictors()) %>%
#   # step_log(all_numeric_predictors(), offset=1) %>%
#   # Scale/center numeric features
#   step_normalize(all_numeric_predictors()) %>%
#   step_pca(all_predictors(), threshold = 0.25) %>%
#   # Apply UMAP on numeric features
#   step_umap(all_numeric_predictors(), neighbors = 10, min_dist = 0.01, metric = "euclidean")


# --- Prep the recipe + plot UMAP colored by condition ---
gait_prep <- prep(gait_rec)

juice(gait_prep) %>% 
  ggplot(aes(UMAP1, UMAP2, label = animal_id)) +
  geom_point(aes(color = condition), alpha = 0.7, size = 2.6) +
  # geom_text(check_overlap = TRUE, hjust = "inward", size=2) + # If want to label by animal ID
  labs(color = NULL)