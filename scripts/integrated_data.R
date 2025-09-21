library(readxl)
library(dplyr)
library(stringr)

# Folder containing your Excel files
data_folder <- "data"

# List all Excel files in the folder
files <- list.files(data_folder, pattern = "\\.xlsx?$", full.names = TRUE)

# Function to process a single file
process_file <- function(filepath) {
  # Extract filename (without extension)
  fname <- tools::file_path_sans_ext(basename(filepath))

  # Split according to your naming convention
  parts <- str_split(fname, "_", simplify = TRUE)
  date_code <- parts[1]
  animal_id <- parts[2]
  condition <- parts[3]
  run_number <- parts[4]

  # Read the **second sheet** and remove first column (just the time)
  df <- read_excel(filepath, sheet = 2)

  # Get run time and then remove that column
  run_time = df$Time[nrow(df)]
  df <- df[,-1]

  # Compute column means (ignore non-numeric columns)
  feature_means <- df %>%
    summarise(across(where(is.numeric), mean, na.rm = TRUE))

  # Add metadata columns
  final_row <- cbind(
    animal_id = animal_id,
    condition = condition,
    run_number = run_number,
    run_time = run_time,
    feature_means
  )

  return(final_row)
}

# Apply to all files
all_data <- lapply(files, process_file) %>% bind_rows()

# 
conditions <- c("PreSCI", "7DPI", "14DPI", "21DPI", "28DPI", "35DPI", "43DPI")
avg_features_by_animal = all_data %>% 
  group_by(animal_id, condition) %>% 
  summarize(
    number_runs = n(),
    total_run_time = sum(run_time),
    across(where(is.numeric) & !c(run_time|run_number), mean)
  ) %>% 
  mutate(condition = factor(condition, levels = conditions)) %>% 
  arrange(condition)

# View(avg_features_by_animal)


# Save run-level averages
write.csv(all_data, "features_per_run.csv", row.names = FALSE)

# Save animal-level summaries
write.csv(avg_features_by_animal, "features_per_animal.csv", row.names = FALSE)
  