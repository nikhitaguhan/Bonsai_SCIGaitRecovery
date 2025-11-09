clear;
warning('off', 'all');

%Make sure the first column is the animal names, and add a second column
%with the header "Condition" and enter the experimental condition for each
%animal

FiletoAnalyse='data_processed_all_with_metadata.xlsx';
MAXRELthreshold = 0.3; %set to zero to disable, or pick a value (cohen d) to ignore features that don't change between the two conditions
uninjuredname = {'Vehicle0','HiDose0','LoDose0'}; %baseline or uninjured
injuredname = {'Vehicle7'};  %injured
FirstParameter='StepWidth_Hind__average'; %make sure you put here the first (leftmore) parameter that appears in your data matrix. The script will extract this column + all the data right of this column
eliminateOutliers = 1; % Set to false to keep all rows for CKS calculation, otherwise this will remove outlier runs per animal per session
exclude_ids = [];  %animal IDs to exclude, e.g. [25101, 25102]
exclude_treatment_groups = {''};  %Treatment Groups to exclude, e.g. {'Vehicle', 'HiDose'}
exclude_timepoints = [];  % Timepoints to exclude, e.g, [7, 14] 

% Parameters for outlier detection
GMMthreshold= 30; %Percentile threshold for Gaussian model outlier detection (e.g. a value of 10 flags the outer 10 percentile points as outliers)

%%%%% BEGIN CODE

%check if there are intermediate output files from previous runs and remove them
filename = 'PC1and2.xlsx';
if exist(filename, 'file')
    delete(filename);
end

filename = 'pca_results.xlsx';
if exist(filename, 'file')
    delete(filename);
end

% Read data from the Excel file
data = readtable(FiletoAnalyse);

%Remove animals specified for exclusion, if any
if exist('exclude_ids', 'var') && ~isempty(exclude_ids)
    % Convert both to strings for reliable comparison
    animal_ids = string(data.AnimalID);
    exclude_ids = string(exclude_ids);

    % Trim whitespace and ensure consistent formatting
    animal_ids = strtrim(animal_ids);
    exclude_ids = strtrim(exclude_ids);

    % Identify rows to keep
    keep_rows = ~ismember(animal_ids, exclude_ids);

    % Apply filter to the table
    data = data(keep_rows, :);
end

%Remove Treatment Groups specified for exclusion, if any
if exist('exclude_treatment_groups', 'var') && ~isempty(exclude_treatment_groups)
    % Validate types
    if ~iscell(exclude_treatment_groups) || ~all(cellfun(@ischar, exclude_treatment_groups))
        error('exclude_treatment_groups must be a cell array of character vectors.');
    end
    if ~iscell(data.TreatmentGroups) || ~all(cellfun(@ischar, data.TreatmentGroups))
        error('data.TreatmentGroups must be a cell array of character vectors.');
    end

    % Trim whitespace for consistency
    treatment_groups = strtrim(data.TreatmentGroups);
    exclude_treatment_groups = strtrim(exclude_treatment_groups);

    % Identify rows to keep
    keep_rows = ~ismember(treatment_groups, exclude_treatment_groups);

    % Apply filter to the table
    data = data(keep_rows, :);
end

%Remove Timepoints specified for exclusion, if any
if ~isempty(exclude_timepoints)
    % Remove rows where Timepoint_days_ matches any value in exclude_timepoints
    data = data(~ismember(data.Timepoint_days_, exclude_timepoints), :);
end

% Get the RunIdentifiers
RunIdentifierColumnIndex = find(strcmp(data.Properties.VariableNames, 'FileName'), 1);
RunIdentifiers = data{:, RunIdentifierColumnIndex};

% Get the conditions
% Find the index of the relevant columns
conditionColumnIndex = find(strcmp(data.Properties.VariableNames, 'Condition'), 1);
Condition = data{:,conditionColumnIndex};

% Check if the required columns are found
if isempty(conditionColumnIndex)
    error('Column "Condition" not found in the data.');
end

% Validate that both conditions exist in the data
if ~any(ismember(Condition, uninjuredname))
    error('None of the specified uninjured conditions found in the data.');
end

if ~any(ismember(Condition, injuredname))
    error('None of the specified injured conditions found in the data.');
end

% Extract the values for PCA (excluding the first column)
FirstParameterColumnIndex = find(strcmp(data.Properties.VariableNames, FirstParameter), 1);
pcaData = data{:, FirstParameterColumnIndex:end};

% Find rows with condition A and condition B
uninjuredname_idx = ismember(Condition, uninjuredname);
injuredname_idx = ismember(Condition, injuredname);

% Extract data for conditions A and B
dataA = pcaData(uninjuredname_idx, :);
dataB = pcaData(injuredname_idx, :);

% Calculate the absolute value of Cohen's d for each column
cohen_d_abs = zeros(1, size(pcaData, 2));
for i = 1:size(pcaData, 2)
    meanA = mean(dataA(:, i));
    meanB = mean(dataB(:, i));
    pooled_std = sqrt(((size(dataA, 1) - 1) * var(dataA(:, i)) + (size(dataB, 1) - 1) * var(dataB(:, i))) / (size(dataA, 1) + size(dataB, 1) - 2));
    cohen_d_abs(i) = abs((meanA - meanB) / pooled_std);
end

% Eliminate columns with cohen_d_abs less than 0.5
pcaData_filtered = pcaData(:, cohen_d_abs >= MAXRELthreshold);
fprintf('Number of parameters used in calculation: %.4f\n', width(pcaData_filtered));

% Standardize the data
pcaData_standardized = zscore(pcaData_filtered);

% Perform PCA analysis
[coeff, score, latent, tsquared, explained] = pca(pcaData_standardized);

% Create a table for the PCA results
pcaTable = array2table(score, 'VariableNames', ...
    strcat('PC', string(1:size(score, 2))));

% Handle duplicate row names
[uniqueRunIdentifiers, ~, idx] = unique(RunIdentifiers, 'stable');
duplicateCounts = histcounts(idx, 1:(numel(uniqueRunIdentifiers) + 1));
for i = 1:numel(uniqueRunIdentifiers)
    if duplicateCounts(i) > 1
        dupIdx = find(idx == i);
        for j = 2:duplicateCounts(i)
            RunIdentifiers{dupIdx(j)} = sprintf('%s_%d', RunIdentifiers{dupIdx(j)}, j);
        end
    end
end

pcaTable.Properties.RowNames = RunIdentifiers;

% Display the PCA results table
%disp(pcaTable);

% Save the PCA results table to a new Excel file
writetable(pcaTable, 'pca_results.xlsx', 'WriteRowNames', true);

%reset RunIdentifiers
RunIdentifiers = data{:, 1};

% Extract metadata columns from the original data (up to FirstParameterColumnIndex - 1)
metadata_columns = data(:, 1:(FirstParameterColumnIndex - 1));

% Create a new table with metadata and PC1, PC2
PC1and2Table = [metadata_columns, ...
    array2table(score(:, 1:2), 'VariableNames', {'PC1', 'PC2'})];

%%%%%%%%%%%%%%


%GMM-based outlier detection using average variance across animal-timepoint groups
T = PC1and2Table;

% Initialize outlier flag
T.OutlierStatus = zeros(height(T), 1);

% Get unique animals and timepoints
animals = unique(T.AnimalID);
days = unique(T.Timepoint_days_);

% Step 1: Compute average covariance matrix across all eligible groups
cov_matrices = [];
for i = 1:numel(animals)
    for j = 1:numel(days)
        idx = T.AnimalID == animals(i) & T.Timepoint_days_ == days(j);
        group_data = [T.PC1(idx), T.PC2(idx)];
        if size(group_data, 1) >= 3
            cov_matrices(:, :, end+1) = cov(group_data);
        end
    end
end

% Average covariance matrix
avg_cov = mean(cov_matrices, 3);

% Step 2: Apply outlier detection per group using shared covariance
for i = 1:numel(animals)
    for j = 1:numel(days)
        idx = T.AnimalID == animals(i) & T.Timepoint_days_ == days(j);
        group_data = [T.PC1(idx), T.PC2(idx)];
        if size(group_data, 1) >= 3
            % Fit Gaussian model with group mean and shared covariance
            mu = mean(group_data);
            gmModel_local = gmdistribution(mu, avg_cov);

            % Compute log-likelihoods for group points
            logL_group = log(pdf(gmModel_local, group_data));

            % Determine threshold based on local distribution
            local_threshold = prctile(logL_group, GMMthreshold);

            % Flag outliers within the group
            T.OutlierStatus(idx) = double(logL_group < local_threshold);
        end
    end
end

% --- End Revised Outlier Detection Block ---

% Convert logical to numeric
T.OutlierStatus = double(T.OutlierStatus);

% Write the updated table
writetable(T, 'PC1and2.xlsx');

%%%%%%%%%%%%%%

% Plot PC1 vs PC2 for Baseline and Injury groups
figure;
hold on;

% Indices for each condition
idx_uninjured = ismember(Condition, uninjuredname);
idx_injured = ismember(Condition, injuredname);

% Plot Baseline group
scatter(score(idx_uninjured, 1), score(idx_uninjured, 2), 'o', 'filled', 'DisplayName', 'Baseline Reference');

% Plot Injury group
scatter(score(idx_injured, 1), score(idx_injured, 2), 's', 'filled', 'DisplayName', 'Injury Reference');

xlabel('PC1');
ylabel('PC2');
title('PCA: PC1 vs PC2');
legend('Location', 'best');
grid on;
hold off;

%%Let's now run the script for calculating CKS


% Read PC1 and PC2 data
data = readtable('PC1and2.xlsx');

% Optionally eliminate outliers
if eliminateOutliers
    data = data(data.OutlierStatus ~= 1, :);
end

% Find the index of the relevant columns
conditionColumnIndex = find(strcmp(data.Properties.VariableNames, 'Condition'));
PC1ColumnIndex = find(strcmp(data.Properties.VariableNames, 'PC1'));
PC2ColumnIndex = find(strcmp(data.Properties.VariableNames, 'PC2'));

% Check if the required columns are found
if isempty(conditionColumnIndex)
    error('Column "Condition" not found in the data.');
end
if isempty(PC1ColumnIndex)
    error('Column "PC1" not found in the data.');
end
if isempty(PC2ColumnIndex)
    error('Column "PC2" not found in the data.');
end

% Get the values for uninjured control
uninjuredrows = data(ismember(data{:, conditionColumnIndex}, uninjuredname), :);

% Get the values for injured control
injuredrows = data(ismember(data{:, conditionColumnIndex}, injuredname), :);

% Check if the required columns are found
if isempty(uninjuredrows)
    error('Uninjured condition not found in the data.');
end
if isempty(injuredrows)
    error('Injured condition not found in the data.');
end

% Get the centroid for uninjured
Xuninjured = mean(uninjuredrows{:, PC1ColumnIndex});
Yuninjured = mean(uninjuredrows{:, PC2ColumnIndex});

% Get the centroid for injured
Xinjured = mean(injuredrows{:, PC1ColumnIndex});
Yinjured = mean(injuredrows{:, PC2ColumnIndex});

% Find the midpoint
Xmidpoint = (Xinjured + Xuninjured) / 2;
Ymidpoint = (Yinjured + Yuninjured) / 2;

% Find the line intersecting the two centroids
% Calculate the slope
m = (Yinjured - Yuninjured) / (Xinjured - Xuninjured);

% Calculate the y-intercept
b = Yinjured - m * Xinjured;

% Extract x and y coordinates from columns 2 and 3
x_coords = data{:, PC1ColumnIndex};
y_coords = data{:, PC2ColumnIndex};

% Initialize arrays to store the projection coordinates
x_proj = zeros(size(x_coords));
y_proj = zeros(size(y_coords));

% Calculate the orthogonal projection for each point to the line connecting
% the centroids
for i = 1:length(x_coords)
    x0 = x_coords(i);
    y0 = y_coords(i);

    % Calculate the x-coordinate of the projection
    x_p = (x0 + m * (y0 - b)) / (1 + m^2);

    % Calculate the y-coordinate of the projection
    y_p = m * x_p + b;

    % Store the projection coordinates
    x_proj(i) = x_p;
    y_proj(i) = y_p;
end

% Add the projection coordinates as new columns to the table
data.x_proj = x_proj;
data.y_proj = y_proj;

% Calculate the direction vector of the line
x_coords = x_proj;
y_coords = y_proj;

direction_vector = [x_coords(end) - x_coords(1), y_coords(end) - y_coords(1)];

% Normalize the direction vector
direction_vector = direction_vector / norm(direction_vector);

% Initialize array to store the transformed 1D coordinates
transformed_coordinates = zeros(size(x_coords));

% Calculate the transformed 1D coordinates along the line
for i = 1:length(x_coords)
    point_vector = [x_coords(i) - Xmidpoint, y_coords(i) - Ymidpoint];
    transformed_coordinates(i) = -dot(point_vector, direction_vector); % Negate to invert direction
end

% Calculate the mean of the projected coordinates for each group
mean_uninjured = mean(transformed_coordinates(ismember(data{:, conditionColumnIndex}, uninjuredname)));
mean_injured = mean(transformed_coordinates(ismember(data{:, conditionColumnIndex}, injuredname)));

% Check if the mean of the uninjured group is not larger than the injured group
if mean_uninjured <= mean_injured
    % Invert the sign of the transformed coordinates
    transformed_coordinates = -transformed_coordinates;
end

% Add the transformed coordinates as new columns to the table
data.CKS = transformed_coordinates;

% --- Begin CKS% Calculation Block ---

% Use previously calculated centroids and column indices
origin = [Xinjured, Yinjured];  % Injured centroid as 0%
target = [Xuninjured, Yuninjured];  % Uninjured centroid as 100%

% Direction vector from injured to uninjured centroid
direction_vector = target - origin;
direction_vector = direction_vector / norm(direction_vector);  % Normalize

% Maximum distance between centroids
max_distance = norm(target - origin);

% Project each data point onto the direction vector
num_points = height(data);
CKS_percent = zeros(num_points, 1);

for i = 1:num_points
    point = [data{ i, PC1ColumnIndex }, data{ i, PC2ColumnIndex }];
    relative_vector = point - origin;
    scalar_projection = dot(relative_vector, direction_vector);

    % Rescale to percentage
    CKS_percent(i) = (scalar_projection / max_distance) * 100;
end

% Add CKS% to the table
data.CKS_percent = CKS_percent;

% --- End CKS% Calculation Block ---

% Get the current timestamp
timestamp = datestr(now, 'yyyymmdd_HHMMSS');

% Create a new filename with the timestamp
new_filename = ['CKS_' timestamp '.xlsx'];

% --- Begin CKS% Visualization Block ---

% Get unique treatment groups and timepoints
treatment_groups = unique(data.TreatmentGroups);
timepoints = unique(data.Timepoint_days_);

% Prepare figure
figure;
hold on;
colors = lines(numel(treatment_groups)); % Distinct colors for each group

for g = 1:numel(treatment_groups)
    group = treatment_groups{g};
    group_data = data(strcmp(data.TreatmentGroups, group), :);

    % For each animal and timepoint, average CKS_percent across trials
    [G, animal_ids, tps] = findgroups(group_data.AnimalID, group_data.Timepoint_days_);
    animal_tp_means = splitapply(@mean, group_data.CKS_percent, G);

    % Create a table for easier aggregation
    animal_tp_table = table(animal_ids, tps, animal_tp_means, ...
        'VariableNames', {'AnimalID', 'Timepoint', 'CKS_percent'});

    % For each timepoint, calculate mean and SEM across animals
    mean_vals = nan(size(timepoints));
    sem_vals = nan(size(timepoints));
    for t = 1:numel(timepoints)
        tp = timepoints(t);
        idx = animal_tp_table.Timepoint == tp;
        vals = animal_tp_table.CKS_percent(idx);
        mean_vals(t) = mean(vals, 'omitnan');
        sem_vals(t) = std(vals, 'omitnan') / sqrt(sum(~isnan(vals)));
    end

    % Plot mean ± SEM
    errorbar(timepoints, mean_vals, sem_vals, '-o', ...
        'Color', colors(g,:), 'MarkerFaceColor', colors(g,:), ...
        'DisplayName', group, 'CapSize', 6, 'LineWidth', 1.5);
end

xlabel('Timepoint (days)');
ylabel('CKS%');
title('CKS% Over Time by Treatment Group');
legend('Location', 'best'); % No legend title for compatibility
grid on;
hold off;

% --- End CKS% Visualization Block ---

% Save the table data to the new Excel file with the timestamp in the filename
writetable(data, new_filename);

% Calculate Cohen's d between CKS values of Baseline and Injury groups
cks_baseline = data.CKS(ismember(data.Condition, uninjuredname));
cks_injury = data.CKS(ismember(data.Condition, injuredname));

mean_baseline = mean(cks_baseline);
mean_injury = mean(cks_injury);
std_baseline = std(cks_baseline);
std_injury = std(cks_injury);

n_baseline = numel(cks_baseline);
n_injury = numel(cks_injury);

% Pooled standard deviation
pooled_std = sqrt(((n_baseline - 1) * std_baseline^2 + (n_injury - 1) * std_injury^2) / (n_baseline + n_injury - 2));

% Cohen's d
cohen_d_cks = (mean_baseline - mean_injury) / pooled_std;

fprintf('Cohen''s d between CKS values of Baseline and Injury groups: %.4f\n', cohen_d_cks);

% --- Begin Settings Logging Block ---

% Define log filename with timestamp
log_filename = ['script_settings_log_' timestamp '.txt'];

% Open file for writing
fid = fopen(log_filename, 'w');

% Write header
fprintf(fid, 'Script Settings Log\n');

% Log script name automatically
[~, script_name, ext] = fileparts(mfilename('fullpath'));
fprintf(fid, 'Script Name: %s%s\n', script_name, ext);
fprintf(fid, 'Timestamp: %s\n\n', datestr(now));

% Log input file
fprintf(fid, 'Input File: %s\n', FiletoAnalyse);

% Log thresholds and parameters
fprintf(fid, 'MaxRel cohen d Threshold: %.4f\n', MAXRELthreshold);
fprintf(fid, 'First Parameter: %s\n', FirstParameter);
fprintf(fid, 'Eliminate Outliers: %d\n', eliminateOutliers);

% Log condition groups
fprintf(fid, 'Baseline Reference: %s\n', strjoin(uninjuredname, ', '));
fprintf(fid, 'Injured Reference: %s\n', strjoin(injuredname, ', '));

% Log exclusions
fprintf(fid, 'Excluded Animal IDs: %s\n', mat2str(exclude_ids));
fprintf(fid, 'Excluded Treatment Groups: %s\n', strjoin(exclude_treatment_groups, ', '));
fprintf(fid, 'Excluded Timepoints: %s\n', mat2str(exclude_timepoints));

% Log outlier detection parameters
fprintf(fid, 'GMM Outlier percentile threshold: %.4f\n', GMMthreshold);

% Log assay window
fprintf(fid,'Cohen''s d between CKS values of Baseline and Injury groups: %.4f\n', cohen_d_cks);
fprintf(fid,'Number of parameters used in calculation: %.4f\n', width(pcaData_filtered));

% Log output files
fprintf(fid, 'PCA Results File: pca_results.xlsx\n');
fprintf(fid, 'PC1 and PC2 File: PC1and2.xlsx\n');
fprintf(fid, 'CKS Results File: %s\n', new_filename);

% Close file
fclose(fid);

fprintf('Settings log saved to: %s\n', log_filename);

% --- End Settings Logging Block ---

