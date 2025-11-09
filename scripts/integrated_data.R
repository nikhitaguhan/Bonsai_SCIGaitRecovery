library(readxl)
library(dplyr)
library(stringr)
library(purrr)
library(tidyr)
library(pracma)
library(signal)
library(tuneR)
library(ggplot2)
# ============================================================
# --- Data ---
data_folder <- "mock_data"
files <- list.files(data_folder, pattern = "\\.xlsx?$", full.names = TRUE)
conditions <- c("PreSCI", "7DPI", "14DPI", "21DPI", "28DPI", "35DPI", "43DPI")
# ============================================================
# --- Function to take column and trim "flatline" sections at beginning and end ---
trim_flat_edges <- function(x) {
  x <- as.numeric(x)
  
  diffs <- diff(x)
  nonconst_idx <- which(diffs != 0)
  
  if (length(nonconst_idx) == 0) {
    # entire feature is constant
    return(list(x = x, idx = seq_along(x)))
  } else {
    start_idx <- nonconst_idx[1]           # first change
    end_idx <- nonconst_idx[length(nonconst_idx)] + 1  # last change
    kept_idx <- start_idx:end_idx
    # print(kept_idx)
    return(list(x = x[kept_idx], idx = kept_idx))
  }
}
# ============================================================
# --- Function to take file and return stats for each feature and data to plot cycles ---
compute_cycle_stats <- function(file, return_plot_data = TRUE) {
  
  df <- read_excel(file, sheet = 2)
  time_col <- df %>% dplyr::select(matches("time", ignore.case = TRUE)) %>% pull(1)
  features <- df %>% dplyr::select(where(is.numeric), -matches("time", ignore.case = TRUE))
  
  stats_list <- list()
  plot_list <- list()
  
  for (feat in names(features)) {
    # print(feat)
    # cat("Processing file:", basename(file), "feature:", feat, "\n")
    x <- as.numeric(features[[feat]])
    trimmed <- trim_flat_edges(round(x, 10))
    x_trim <- trimmed$x
    t_trim <- time_col[trimmed$idx]
    
    # Replace x_trim if it is near-constant (all zeros)
    if (all(x_trim == 0)) {
      x_trim <- as.numeric(features[[feat]])
      t_trim <- time_col
    }
    # Skip if x_trim is constant
    if (length(unique(x_trim)) == 1) {
      cat("File:", basename(file), "\nFeature:", feat, "– constant\n")
      next
    }
    
    # Using Fast Fourier Transform for signal processing
    n <- length(x_trim)
    xf <- fft(x_trim)
    mag <- Mod(xf)[1:floor(n/2)]
    dominant_idx <- which.max(mag[2:length(mag)]) + 1
    dominant_freq <- (dominant_idx - 1) / n * 100
    samples <- 100 / dominant_freq  # expected samples per cycle
    # print(samples)
    
    # Using Autocorrelation Function to compute peaks and cycles
    acf_result <- acf(x_trim, lag.max = length(x_trim)/2, plot = FALSE)
    first_peak_lag <- which.max(acf_result$acf[11:length(acf_result$acf)]) + 10
    # version 1
    mpd <- min(round(samples * 0.9, 0), round(first_peak_lag * 0.9, 0)) # Adaptive minimum distance
    if (mpd < 15) mpd <- 20
    # if (mpd > length(x_trim)/2) cat("File:", basename(file), "Feature:", feat)
    # version 2
    # mpd <- min(samples * 0.9, length(x_trim)/2)
    # print(mpd)
    
    peaks <- findpeaks(x_trim, minpeakdistance = mpd)
    troughs <- findpeaks(-x_trim, minpeakdistance = mpd)
    
    # Save originals for fallback after next step
    orig_peaks <- peaks
    orig_troughs <- troughs
    
    # Filter out peaks/troughs that are too similar in height (noise)
    if (!is.null(peaks) && !is.null(troughs) && nrow(peaks) >= 5 || nrow(troughs) >= 5) {
      # Use max peak and min trough to set threshold
      max_peak_height <- max(peaks[, 1])
      min_trough_height <- -max(troughs[, 1])  # deepest trough
      max_amplitude <- max_peak_height - min_trough_height
      threshold <- 0.1 * max_amplitude

      # Calculate average peak and trough heights
      avg_peak_height <- mean(peaks[, 1])
      avg_trough_height <- mean(-troughs[, 1])

      # cat("Feature:", feat, "Max amplitude:", max_amplitude, "Threshold:", threshold, "\n")

      # Filter out peak-trough pairs that are too close in height
      keep_peaks <- rep(TRUE, nrow(peaks))
      keep_troughs <- rep(TRUE, nrow(troughs))

      for (i in 1:nrow(peaks)) {
        peak_pos <- peaks[i, 2]
        peak_height <- peaks[i, 1]

        # Find nearest trough (before or after this peak)
        trough_distances <- abs(troughs[, 2] - peak_pos)
        nearest_trough_idx <- which.min(trough_distances)
        trough_height <- -troughs[nearest_trough_idx, 1]

        # Check if peak and trough are too close in height, filter out if so
        amplitude <- peak_height - trough_height
        # cat("Peak", i, "pos=", peak_pos, "-> Trough pos=", troughs[nearest_trough_idx, 2],
        #     "Amplitude=", amplitude, "\n")

        if (amplitude < threshold) {
          if (peak_height < avg_peak_height) {
            keep_peaks[i] <- FALSE
          }
          if (trough_height > avg_trough_height) {
            keep_troughs[nearest_trough_idx] <- FALSE
          }

        }
      }

      peaks_filt <- peaks[keep_peaks, , drop = FALSE]
      troughs_filt <- troughs[keep_troughs, , drop = FALSE]

      # If filtering peaks filtered out too much, fall back on original
      if (nrow(peaks_filt) < 2) {
        peaks <- orig_peaks
      } else {
        peaks <- peaks_filt
      }

      # If filtering troughs filtered out too much, fall back on original
      if (nrow(troughs_filt) < 2) {
        troughs <- orig_troughs
      } else {
        troughs <- troughs_filt
      }
    }
    
    # Cycle stats
    peak_indices <- sort(peaks[,2])
    cycle_lengths <- diff(peak_indices)
    avg_cycle_length <- mean(cycle_lengths)
    avg_cycle_max <- mean(peaks[,1])
    avg_cycle_min <- -mean(troughs[,1])
    # FFT-based regularity
    dominant_power <- mag[which.max(mag)]^2
    total_power <- sum(mag^2)
    regularity_fft <- dominant_power / total_power
    
    # All stats per feature to return
    stats_list[[feat]] <- data.frame(
      feature = feat,
      avg_cycle_length = avg_cycle_length,
      avg_cycle_max = avg_cycle_max,
      avg_cycle_min = avg_cycle_min,
      run_mean = mean(x_trim, na.rm = TRUE),
      regularity_score = regularity_fft
    )
    
    # All data to plot feature cycles to return
    if (return_plot_data) {
      plot_list[[feat]] <- data.frame(
        time = t_trim,
        value = x_trim,
        feature = feat,
        type = "Signal"
      )
      
      plot_list[[feat]] <- rbind(plot_list[[feat]],
                                 data.frame(
                                   time = t_trim[peaks[,2]],
                                   value = peaks[,1],
                                   feature = feat,
                                   type = "Peak"
                                 ))
      plot_list[[feat]] <- rbind(plot_list[[feat]],
                                 data.frame(
                                   time = t_trim[troughs[,2]],
                                   value = -troughs[,1],
                                   feature = feat,
                                   type = "Trough"
                                 ))
    }
  }
  
  stats_df <- bind_rows(stats_list)
  plot_df <- if(return_plot_data) bind_rows(plot_list) else NULL
  
  return(list(stats = stats_df, plot_data = plot_df))
}
# ============================================================
# --- Function to take all files and use compute stats function to get integrated, full dataset ---
all_runs <- map_dfr(files, function(f) {
  # Parse metadata
  fname <- tools::file_path_sans_ext(basename(f))
  parts <- str_split(fname, "_")[[1]]
  animal_id <- parts[2]
  condition <- parts[3]
  run_number <- parts[4]

  # Compute cycle stats and run mean
  cs <- compute_cycle_stats(f, return_plot_data = FALSE)
  stats_df <- cs$stats

  # Skip file if all features are constant
  if (nrow(stats_df) == 0) {
    cat("Skipping file (all features constant):", basename(f), "\n")
    return(NULL)
  }

  # Extract run-level means
  run_means <- stats_df %>%
    dplyr::select(feature, run_mean) %>%
    pivot_wider(names_from = feature, values_from = run_mean)

  # Extract other stats
  cycle_stats <- stats_df %>%
    dplyr::select(feature, avg_cycle_length, avg_cycle_max, avg_cycle_min, regularity_score) %>%
    pivot_longer(-feature, names_to = "stat", values_to = "value") %>%
    unite("varname", feature, stat, sep = "_") %>%
    pivot_wider(names_from = varname, values_from = value)

  # Combine all stats into a single row
  cbind(
    animal_id = animal_id,
    condition = condition,
    run_number = run_number,
    run_time = max(read_excel(f, sheet = 2)$Time),
    run_means,
    cycle_stats
  )
})
# ============================================================
# --- Function to take input file(s) and plot the signals for all features together ---
# Helps verify cyclical nature and correct peak detection + feature extraction
# for (i in 3) {
#   file <- files[i]
#   cycle_df <- compute_cycle_stats(file)
#   plot_df <- cycle_df$plot_data
# 
#   n_feats <- length(unique(plot_df$feature))
#   half <- ceiling(n_feats / 2)
#   feats1 <- unique(plot_df$feature)[1:half]
#   feats2 <- unique(plot_df$feature)[(half+1):n_feats]
# 
#   p1 <- ggplot(subset(plot_df, feature %in% feats1), aes(x = time, y = value)) +
#     geom_line(data = subset(plot_df, type=="Signal" & feature %in% feats1), color = "steelblue") +
#     geom_point(data = subset(plot_df, type!="Signal" & feature %in% feats1), aes(color = type), size = 1.2) +
#     scale_color_manual(values = c("Peak" = "red", "Trough" = "blue")) +
#     facet_wrap(~ feature, scales = "free_y", ncol = 4) +
#     theme_minimal(base_size = 10) +
#     theme(strip.text = element_text(face = "bold", size = 9)) +
#     labs(title = paste("Cycle Visualization (Features 1) —", basename(file)), x = "Time", y = "Value")
# 
#   p2 <- ggplot(subset(plot_df, feature %in% feats2), aes(x = time, y = value)) +
#     geom_line(data = subset(plot_df, type=="Signal" & feature %in% feats2), color = "steelblue") +
#     geom_point(data = subset(plot_df, type!="Signal" & feature %in% feats2), aes(color = type), size = 1.2) +
#     scale_color_manual(values = c("Peak" = "red", "Trough" = "blue")) +
#     facet_wrap(~ feature, scales = "free_y", ncol = 4) +
#     theme_minimal(base_size = 10) +
#     theme(strip.text = element_text(face = "bold", size = 9)) +
#     labs(title = paste("Cycle Visualization (Features 2) —", basename(file)), x = "Time", y = "Value")
# 
#   print(p1)
#   print(p2)
# }

# If want to randomly sampling 3 runs in animals with > 3
# set.seed(123)
# all_runs <- all_runs %>%
#   drop_na() %>%
#   group_by(animal_id, condition) %>%
#   slice_sample(n = 3, replace = FALSE) %>%
#   ungroup()
# ============================================================
# --- Create all CSVs ---
# 1. Run-level averages (just mean)
run_level_avg <- all_runs %>%
  dplyr::select(animal_id, condition, run_number, run_time, matches("^[^_]+$"))
write.csv(run_level_avg, "avg_features_per_run.csv", row.names = FALSE)

# 2. Combined run-level (means + cycle stats)
write.csv(all_runs, "combined_avg_features_per_run.csv", row.names = FALSE)

# 3. Animal-level summaries
animal_level_avg <- all_runs %>%
  group_by(animal_id, condition) %>%
  summarise(
    number_runs = n(),
    total_run_time = sum(run_time),
    across(where(is.numeric) & !matches("run_time|run_number|avg_cycle"), mean, na.rm = TRUE)
  ) %>%
  mutate(condition = factor(condition, levels = conditions)) %>%
  arrange(condition)
write.csv(animal_level_avg, "avg_features_per_animal.csv", row.names = FALSE)