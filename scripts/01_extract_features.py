"""
01_extract_features.py

Phase 1: Feature Extraction with Method A: Baseline summary stats, Method B: TSFresh, or Method C: Catch22
Currently uses Method A: basic stats & cycle based.

Inputs: Motorater Excel files in data/
Outputs: 
    If run: python 01_extract_features.py --visualize [num_files], outputs cycle_plots folder
    Otherwise: ouptuts extracted_features_method{A/B/C}.csv

Expects folder called "outputs" to save output cycle_plots folder or csv
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path
from scipy import signal, stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def trim_flat_edges(x: np.ndarray) -> tuple:
    """
    Remove flatline sections at the beginning and end of a time series.
    Finds first and last points where the signal changes, keeps only that range.
    
    Returns: (trimmed_array, kept_indices)
    """
    x = np.asarray(x, dtype=float)
    diffs = np.diff(x)
    nonconst_idx = np.where(diffs != 0)[0]
    
    if len(nonconst_idx) == 0:
        return x, np.arange(len(x))
    
    start_idx = nonconst_idx[0]
    end_idx = nonconst_idx[-1] + 1
    kept_idx = np.arange(start_idx, end_idx + 1)
    
    return x[kept_idx], kept_idx

def visualize_file_cycles(file_path: Path, output_dir: Path = None):
    """
    Visualize signals with detected peaks and troughs for a single file and extract its features.
    
    Parameters:
    -----------
    file_path : Path to Excel file
    output_dir : Directory to save plots (if None, just displays)
    """
    # Load and process file
    df = pd.read_excel(file_path, sheet_name=1)
    df.columns = (df.columns.str.lower()
                 .str.replace(' ', '_')
                 .str.replace('(', '')
                 .str.replace(')', ''))
    
    time_col = df['time'].values if 'time' in df.columns else np.arange(len(df))
    feature_cols = [col for col in df.columns if col != 'time']
    
    # Extract features with peak/trough info
    plot_data = {}
    for col in feature_cols:
        x = df[col].values
        x_trimmed, kept_idx = trim_flat_edges(np.round(x, 10))
        
        if np.all(x_trimmed == 0):
            x_trimmed = x
            kept_idx = np.arange(len(x))
        
        t_trimmed = time_col[kept_idx]
        
        feature_stats = extract_features_methodA(x_trimmed, col, return_peak_info=True)

        if 'peak_info' in feature_stats:
            plot_data[col] = {
                'time': t_trimmed,
                'signal': feature_stats['peak_info']['signal'],
                'peaks': feature_stats['peak_info']['peaks'],
                'troughs': feature_stats['peak_info']['troughs']
            }
    
    # Create plots
    n_features = len(plot_data)
    if n_features == 0:
        print(f"No features to plot for {file_path.name}")
        return
    
    # Split into two figures since many features
    features_per_plot = 24  # 6 cols x 4 rows
    n_plots = int(np.ceil(n_features / features_per_plot))
    
    feature_list = list(plot_data.keys())
    
    for plot_idx in range(n_plots):
        start_idx = plot_idx * features_per_plot
        end_idx = min(start_idx + features_per_plot, n_features)
        features_subset = feature_list[start_idx:end_idx]
        
        n_subset = len(features_subset)
        ncols = min(6, n_subset)
        nrows = int(np.ceil(n_subset / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3*nrows))
        fig.suptitle(f"Cycle Visualization — {file_path.name} (Part {plot_idx+1}/{n_plots})", 
                     fontsize=14, fontweight='bold')
        
        if n_subset == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, feat in enumerate(features_subset):
            ax = axes[idx]
            data = plot_data[feat]
            
            # Plot signal
            ax.plot(data['time'], data['signal'], color='steelblue', linewidth=0.8, label='Signal')
            
            # Plot peaks
            if len(data['peaks']) > 0:
                ax.scatter(data['time'][data['peaks']], 
                          data['signal'][data['peaks']], 
                          color='red', s=20, zorder=5, label='Peaks')
            
            # Plot troughs
            if len(data['troughs']) > 0:
                ax.scatter(data['time'][data['troughs']], 
                          data['signal'][data['troughs']], 
                          color='blue', s=20, zorder=5, label='Troughs')
            
            ax.set_title(feat.replace('_', ' ').title(), fontsize=9, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Value', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(fontsize=7, loc='upper right')
        
        # Hide unused subplots
        for idx in range(n_subset, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save or show if output_dir is not specified
        if output_dir:
            output_dir.mkdir(exist_ok=True, parents=True)
            output_file = output_dir / f"{file_path.stem}_cycles_part{plot_idx+1}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_file}")
        else:
            plt.show()
        
        plt.close()


def extract_features_methodA(x: np.ndarray, feat_name: str, return_peak_info: bool = False) -> dict:
    """
    Extract Method A summary statistics from a single feature time series (trimmed).
    
    Includes:
    - Basic stats: mean, std, min, max, range, IQR, CV
    - Cycle-based: avg_cycle_length, avg_cycle_max, avg_cycle_min, regularity_score
    
    Parameters:
    -----------
    return_peak_info : If True, also return peak/trough indices for plotting
    
    Returns: dict of feature_name -> value pairs (+ optional peak_info)
    """
    stats_dict = {}
    prefix = feat_name
    
    # Skip if feature is completely constant
    if len(np.unique(np.round(x, 10))) == 1:
        # result = {f"{prefix}_constant": True}
        # if return_peak_info:
        #     result['peak_info'] = {'peaks': np.array([]), 'troughs': np.array([]), 'signal': x}
        return {}
    
    # Basic summary statistics
    stats_dict[f"{prefix}_mean"] = np.mean(x)
    stats_dict[f"{prefix}_std"] = np.std(x)
    stats_dict[f"{prefix}_min"] = np.min(x)
    stats_dict[f"{prefix}_max"] = np.max(x)
    stats_dict[f"{prefix}_range"] = np.ptp(x)
    stats_dict[f"{prefix}_iqr"] = stats.iqr(x)
    
    # Coefficient of variation (CV)
    mean_val = np.mean(x)
    if mean_val != 0:
        stats_dict[f"{prefix}_cv"] = np.std(x) / abs(mean_val)
    else:
        stats_dict[f"{prefix}_cv"] = 0
    
    # Cycle-based features using peak detection
    try:
        # Detect peaks/troughs
        min_distance = 10
        peaks, _ = signal.find_peaks(x, distance=min_distance, prominence = np.std(x) * 1)
        troughs, _ = signal.find_peaks(-x, distance=min_distance, prominence = np.std(x) * 1)

        # Retry with no prominence criteria if nothing found
        if len(peaks) == 0:
            peaks, _ = signal.find_peaks(x, distance=min_distance)
        if len(troughs) == 0:
            troughs, _ = signal.find_peaks(-x, distance=min_distance)

        # Filter outlier peaks and troughs - remove peaks too below peak average, troughs too above trough average
        peak_vals = x[peaks]
        trough_vals = x[troughs]

        if len(peaks) >= 4:
            peak_mean = np.mean(peak_vals)
            peak_std = np.std(peak_vals)
            keep_peaks = peaks[peak_vals >= (peak_mean - 1.5 * peak_std)]  # only remove unusually small peaks
            peaks = np.array(keep_peaks)

        if len(troughs) >= 4:
            trough_mean = np.mean(trough_vals)
            trough_std = np.std(trough_vals)
            keep_troughs = troughs[trough_vals <= (trough_mean + 1.5 * trough_std)]  # only remove unusually high troughs
            troughs = np.array(keep_troughs)

        # Compute cycle metrics
        if len(peaks) >= 2:
            cycle_lengths = np.diff(peaks)
            stats_dict[f"{prefix}_avg_cycle_length"] = np.mean(cycle_lengths)
        elif len(troughs) >= 2:
            cycle_lengths = np.diff(troughs)
            stats_dict[f"{prefix}_avg_cycle_length"] = np.mean(cycle_lengths)
        # Avg cycle max
        if len(peaks) >= 1:
            stats_dict[f"{prefix}_avg_cycle_max"] = np.mean(x[peaks])
        # Avg cycle min
        if len(troughs) >= 1:
            stats_dict[f"{prefix}_avg_cycle_min"] = np.mean(x[troughs])

        # Regularity = fraction of total power in dominant frequency ± tolerance
        fs = 100
        freqs, psd = signal.welch(x, fs=fs, nperseg=min(256, len(x)))
        dominant_freq = freqs[np.argmax(psd)] if len(freqs) > 0 else 0

        if len(psd) > 1 and np.sum(psd) > 0:
            band = (freqs > dominant_freq * 0.8) & (freqs < dominant_freq * 1.2)
            dominant_power = np.sum(psd[band])
            total_power = np.sum(psd)
            stats_dict[f"{prefix}_regularity_score"] = dominant_power / total_power
        else:
            stats_dict[f"{prefix}_regularity_score"] = np.nan

    except Exception as e:
        stats_dict[f"{prefix}_avg_cycle_length"] = np.nan
        stats_dict[f"{prefix}_avg_cycle_max"] = np.nan
        stats_dict[f"{prefix}_avg_cycle_min"] = np.nan
        stats_dict[f"{prefix}_regularity_score"] = np.nan
        peaks, troughs = np.array([]), np.array([])
    
    # Add peak info if requested
    if return_peak_info:
        stats_dict['peak_info'] = {
            'peaks': peaks,
            'troughs': troughs,
            'signal': x
        }
    
    return stats_dict

def extract_features_methodC(df, **kwargs):
    """
    Method B: Extract features with TSFresh (https://tsfresh.readthedocs.io/en/latest/text/introduction.html).
    TODO: Implement this method.
    """
    raise NotImplementedError("Method B not yet implemented")

def extract_features_methodC(df, **kwargs):
    """
    Method B: Extract features with Catch22 (hhttps://time-series-features.gitbook.io/catch22/information-about-catch22/feature-descriptions).
    TODO: Implement this method.
    """
    raise NotImplementedError("Method C not yet implemented")


def extract_file_features(df: pd.DataFrame, method="A") -> dict:
    """
    Extract features from all columns in a dataframe using specified method.
    Each feature column is trimmed to remove flatline beginning/end, then stats are computed.
    
    Returns: dict with all extracted features (flattened)
    """
    all_features = {}
    feature_cols = [col for col in df.columns if col != 'time']
    
    for col in feature_cols:
        x = df[col].values
        
        # Trim flat edges for this feature
        x_trimmed, _ = trim_flat_edges(np.round(x, 10))
        
        # If all-zero, just use original feature
        if np.all(x_trimmed == 0):
            x_trimmed = x
        
        # Extract stats for this feature
        if method == "A":
            feats = extract_features_methodA(x_trimmed, col)
        else:
            raise NotImplementedError(f"Method {method} not yet implemented")
        
        all_features.update(feats)
    
    return all_features


def extract_dir_features(data_dir: str, key_file: str, method: str = "A"):
    """
    Main pipeline to load/extract features from MotoRater files, creating one final dataset.
    
    Parameters:
    -----------
    method : Feature extraction method ("A", "B", or "C")
        A = Summary stats + cycle features
        B = TSFresh (comprehensive)
        C = Catch22 (interpretable)
    
    Returns: features_dataframe
    """
    data_path = Path(data_dir)
    key_path = Path(key_file)
    
    print("="*60)
    print("MOTORATER FEATURE EXTRACTION PIPELINE")
    print(f"Method: {method}")
    print("="*60)
    print()
    
    # Load animal key
    print(f"Loading animal data key from: {key_path}")
    key_df = pd.read_excel(key_path)
    print(f"  Loaded {len(key_df)} entries")
    print(f"  Unique animals: {key_df['Animal ID'].nunique()}")
    print(f"  Treatment groups: {key_df['Treatment Groups'].unique()}")
    print(f"  Timepoints: {sorted(key_df['Timepoint (days)'].unique())}")
    print()
    
    # Find all Excel files
    excel_files = list(data_path.glob("*.xlsx"))
    print(f"Found {len(excel_files)} Excel files in {data_dir}")
    print()
    
    # Process each file
    results = []
    extracted_features_list = []
    
    for i, file_path in enumerate(excel_files, 1):
        if i % 100 == 0:
            print(f"Processing file {i}/{len(excel_files)}...")
        
        try:
            # Parse filename: [DateCode]_[AnimalID]_[XDPI]_[RunNumber]_out.xlsx
            name = file_path.stem
            parts = name.split('_')
            animal_id = parts[1]
            
            # Load Sheet 2 (features) and standardize column names
            df = pd.read_excel(file_path, sheet_name=1)
            df.columns = (df.columns.str.lower()
                         .str.replace(' ', '_')
                         .str.replace('(', '')
                         .str.replace(')', ''))
            
            # Look up treatment group from key - match by filename
            filename_base = file_path.stem  # without extension
            key_match = key_df[key_df['FileName'].str.replace('.xlsx', '') == filename_base]
            
            if len(key_match) > 0:
                treatment_group = key_match.iloc[0]['Treatment Groups']
                timepoint_days = key_match.iloc[0]['Timepoint (days)']
            else:
                treatment_group = 'Unknown'
                timepoint_days = -1
                print(f"  WARNING: Animal ID {animal_id} not found in key")
            
            # Extract features based on method
            features = extract_file_features(df, method=method)
            
            # Store metadata + features
            row = {
                'animal_id': animal_id,
                'treatment_group': treatment_group,
                'timepoint_days': timepoint_days,
                'timepoint_label': parts[2],
                'run_number': parts[3],
                'date_code': parts[0],
                'filename': file_path.name,
                'condition': f"{treatment_group}{timepoint_days}"
            }
            row.update(features)
            extracted_features_list.append(row)
            
            # Store raw processing info
            results.append({
                **row,
                'n_rows_original': len(df)
            })
            
        except Exception as e:
            print(f"  ERROR processing {file_path.name}: {e}")
            continue
    
    print()
    print(f"Successfully processed {len(results)} files")
    print()
    
    # Create dataframes
    summary_df = pd.DataFrame(results)
    features_df = pd.DataFrame(extracted_features_list)
    
    print("="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files processed: {len(results)}")
    print(f"Treatment groups: {summary_df['treatment_group'].value_counts().to_dict()}")
    print(f"Timepoints: {sorted(summary_df['timepoint_days'].unique())}")
    print(f"Features extracted per file: {len(features_df.columns) - 8}")  # minus metadata cols
    print()
    
    return features_df


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    import sys
    
    DATA_DIR = "data"
    KEY_FILE = "animal_data_key.xlsx"
    OUTPUT_DIR = Path("outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    USE_METHOD = "A"
    
    # To visualize cycles & features for select files, run: python 01_extract_features.py --visualize [num_files]
    if len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        excel_files = list(Path(DATA_DIR).glob("*.xlsx"))

        # Plot random X files (or specify file index) and save their features
        n_files = 3 if len(sys.argv) <= 2 else int(sys.argv[2])
        # random.seed(42)
        selected_files = random.sample(excel_files, min(n_files, len(excel_files)))
        
        plot_output = OUTPUT_DIR / "cycle_plots"
        
        print(f"Visualizing {len(selected_files)} random files...")
        feature_rows = []  # store extracted features for each visualized file
        
        for i, file_path in enumerate(selected_files, 1):
            print(f"\nProcessing file {i}: {file_path.name}")
            visualize_file_cycles(file_path, output_dir=plot_output)
            
            # Extract same features (using existing logic)
            df = pd.read_excel(file_path, sheet_name=1)
            df.columns = (df.columns.str.lower()
                         .str.replace(' ', '_')
                         .str.replace('(', '')
                         .str.replace(')', ''))
            
            features = extract_file_features(df)
            row = {'filename': file_path.name}
            row.update(features)
            feature_rows.append(row)
        
        # Save visualization features CSV in cycle_plots
        features_viz_df = pd.DataFrame(feature_rows)
        csv_path = plot_output / "visualized_features_methodA.csv"
        features_viz_df.to_csv(csv_path, index=False)
        print(f"\nSaved visualization features to: {csv_path}")
        print(f"Plots saved to: {plot_output}")
    
    else:
        # Normal extraction mode (create csv with full feature set)
        features_df = extract_dir_features(DATA_DIR, KEY_FILE, method=USE_METHOD)
        
        # Save to CSV
        output_file = OUTPUT_DIR / f"extracted_features_method{USE_METHOD}.csv"
        features_df.to_csv(output_file, index=False)
        print(f"Saved features to: {output_file}")
        print(f"Shape: {features_df.shape}")
        print()
        
        # Output example
        print("Example - First file extracted features:")
        print(f"  File: {features_df.iloc[0]['filename']}")
        print(f"  Animal: {features_df.iloc[0]['animal_id']}")
        print(f"  Treatment: {features_df.iloc[0]['treatment_group']}")
        print(f"  Timepoint: {features_df.iloc[0]['timepoint_days']} days")
        print(f"  Sample features: {list(features_df.columns[8:13])}")