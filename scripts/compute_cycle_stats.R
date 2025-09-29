library(readxl)
library(dplyr)
library(stringr)
library(zoo)
library(pracma)
library(purrr)
library(tidyr)

# Folder containing your Excel files
data_folder <- "data"
files <- list.files(data_folder, pattern = "\\.xlsx?$", full.names = TRUE)

# # This is just the plot for visualizing the cycle nature of one feature
# df <- read_excel(files[2], sheet = 2)
# acf(df$`Hip Angle (Right)`, lag.max = 200)

# --- Function to compute cycle stats for one feature ---
cycle_stats <- function(x, minpeakdist = 20) {
  x <- as.numeric(x)
  x <- x[!is.na(x)]  # drop NAs
  run_mean = mean(x)
  
  # Maxima
  peaks <- findpeaks(x, minpeakdistance = minpeakdist)
  if (is.null(peaks) || nrow(peaks) < 2) {
    return(data.frame(avg_cycle_length = NA, avg_cycle_max = NA, avg_cycle_min = NA))
  }
  
  peak_indices <- sort(peaks[,2])
  cycle_lengths <- diff(peak_indices)
  avg_cycle_length <- mean(cycle_lengths) # average cycle length
  avg_cycle_max <- mean(peaks[,1])  # average peak height (cycle max)
  
  # Minima
  troughs <- findpeaks(-x, minpeakdistance = minpeakdist)
  avg_cycle_min <- if (!is.null(troughs)) -mean(troughs[,1]) else NA
  
  data.frame(
    avg_cycle_length = avg_cycle_length,
    avg_cycle_max = avg_cycle_max,
    avg_cycle_min = avg_cycle_min
  )
}

# --- Loop through all files ---
all_cycle_stats <- map_dfr(files, function(f) {
  # Extract ID from filename: [date]_[animalid]_[condition]_[run]_out
  fname <- tools::file_path_sans_ext(basename(f))
  parts <- str_split(fname, "_")[[1]]
  row_id <- paste(parts[2], parts[3], parts[4], sep = "_")
  
  # Read file (2nd sheet assumed)
  df <- read_excel(f, sheet = 2)
  
  # Apply cycle_stats across numeric columns
  stats <- df %>%
    select(where(is.numeric), -matches("time", ignore.case = TRUE)) %>%
    map_dfr(~ cycle_stats(.x), .id = "feature")
  
  # Reshape: one row per row_id, columns = feature_avg_cycle_length, feature_avg_cycle_max, feature_avg_cycle_min
  stats_wide <- stats %>%
    pivot_longer(-feature, names_to = "stat", values_to = "value") %>%
    unite("var", feature, stat) %>%
    pivot_wider(names_from = var, values_from = value)
  
  stats_wide %>%
    mutate(id = row_id, .before = 1)
})

# Final results
View(all_cycle_stats)

write.csv(all_cycle_stats, "avg_cycle_features_per_run.csv", row.names = FALSE)