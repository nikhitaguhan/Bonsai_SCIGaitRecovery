"""
02_transform_select_features.py

Phase 2:
1. Tranform/clean data: mean impute missing values, normalize by each animal's baseline average, adjust for skew, scale
2. Feature Selection with Method A: RF Classifier, Method B: Cohen's d effect size, or Method C: Intersection
    - Currently uses Method B: identifies features most affected by injury (baseline vs 7dpi) based on effect size magnitude.

Inputs: extracted_features_methodA.csv
Outputs: 
    transformed_selected_features_method{A/B/C}.csv

    NOTE: for other methods, can create outptus similar to the below that save features w/ metric used for feature selection
    feature_info_methodB.csv: csv with Cohen's d baseline -> injury and injury -> recovery effect size for features
    methodB_effectsize_heatmap: heatmap for features with highest abs(Cohen's d baseline -> injury effect size)

Expects folder called "outputs" with input csv (run 01_extract_features.py) and to save output csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def mean_impute_by_group(df, group_cols=['animal_id', 'treatment_group', 'timepoint_days']):
    """
    Mean impute missing values using the average of other runs for same animal/condition.
    
    Parameters:
    -----------
    df : Data with potential missing values
    group_cols : Columns to group by for computing means
    
    Returns: DataFrame with imputed values
    """
    df_imputed = df.copy()
    
    # Get feature columns (exclude metadata)
    feature_cols = [col for col in df.columns if col not in 
                    ['animal_id', 'treatment_group', 'timepoint_days', 'timepoint_label', 
                     'run_number', 'date_code', 'filename', 'condition']]
    
    # Track imputation stats
    n_missing_before = df_imputed[feature_cols].isna().sum().sum()
    
    # Impute each feature
    for col in feature_cols:
        if df_imputed[col].isna().any():
            # Compute group means
            group_means = df_imputed.groupby(group_cols)[col].transform('mean')
            # Fill missing with group mean
            df_imputed[col].fillna(group_means, inplace=True)
    
    n_missing_after = df_imputed[feature_cols].isna().sum().sum()
    
    print(f"Imputation complete:")
    print(f"  Missing values before: {n_missing_before}")
    print(f"  Missing values after: {n_missing_after}")
    
    return df_imputed

def normalize_by_animal_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features by each animal's baseline average.
    """
    df_norm = df.copy()
    
    meta_cols = ['animal_id', 'treatment_group', 'timepoint_days', 
                 'timepoint_label', 'run_number', 'date_code', 
                 'filename', 'condition']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    # Compute baseline mean per animal
    baseline_means = (
        df[df['timepoint_days'] == 0]
        .groupby('animal_id')[feature_cols]
        .mean()
    )
    
    # Normalize each row by that animal's baseline
    for animal, base_vals in baseline_means.iterrows():
        idx = df['animal_id'] == animal
        df_norm.loc[idx, feature_cols] = df.loc[idx, feature_cols] / base_vals
    
    # Impute extreme values
    for col in feature_cols:
        extreme_mask = df_norm[col].abs() > 1000
        for row in df_norm[extreme_mask].index:
            animal = df_norm.at[row, 'animal_id']
            timepoint = df_norm.at[row, 'timepoint_days']
            mask = (df_norm['animal_id'] == animal) & (df_norm['timepoint_days'] == timepoint) & (df_norm.index != row)
            replacement = df_norm.loc[mask, col].mean()
            print(f"Imputing extreme value: Row {row}, Animal {animal}, Timepoint {timepoint}, Column {col}")
            df_norm.at[row, col] = replacement

    print("\nNormalized to each animal's baseline")
    
    return df_norm

def skew_and_scale(df: pd.DataFrame, skew_threshold: float = 1) -> tuple:
    """
    Apply Yeo-Johnson transformation if skewness > threshold, then standardize.
    """
    meta_cols = ['animal_id', 'treatment_group', 'timepoint_days', 
                 'timepoint_label', 'run_number', 'date_code', 
                 'filename', 'condition']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    df_transformed = df.copy()
    transform_params = {}

    # Step 1: Apply Yeo-Johnson if skewness exceeds threshold
    skewed_count = 0
    for col in feature_cols:
        skewness = df[col].skew()
        if abs(skewness) > skew_threshold:
            transformed, lmbda = stats.yeojohnson(df[col])
            df_transformed[col] = transformed
            transform_params[col] = {'lambda': lmbda, 'transformed': True}
            skewed_count += 1
        else:
            transform_params[col] = {'lambda': None, 'transformed': False}
    
    # Step 2: Standardize all features using StandardScaler
    scaler = StandardScaler()
    df_transformed[feature_cols] = scaler.fit_transform(df_transformed[feature_cols])
    
    for i, col in enumerate(feature_cols):
        transform_params[col]['mean'] = scaler.mean_[i]
        transform_params[col]['std'] = np.sqrt(scaler.var_[i])
    
    print(f"\nApplied skew adjustment to {skewed_count}/{len(feature_cols)} features and scaled all features (mean=0, std=1)")
    
    return df_transformed, transform_params

def select_features_methodB(df, threshold=0.6):
    """
    Method B: Feature Selection using Cohen's d Effect Size
    Identifies features most affected by injury (baseline vs 7dpi).

    Parameters:
    -----------
    df : Full imputed feature data
    threshold : Minimum absolute Cohen's d effect size for selection (e.g. 0.6)

    Returns: (selected_features_df, all_effect_sizes_df)
    """
    print(f"\nCohen's d Effect Size (threshold >= {threshold})")

    # Get feature columns ---
    feature_cols = [col for col in df.columns if col not in 
                    ['animal_id', 'treatment_group', 'timepoint_days', 'timepoint_label', 
                     'run_number', 'date_code', 'filename', 'condition']]

    # Subset data for key timepoints
    baseline_tp = 0
    injury_tp = 7
    recovery_tp = 42
    baseline_data = df[df['timepoint_days'] == baseline_tp]
    injury_data = df[df['timepoint_days'] == injury_tp]
    recovery_data = df[df['timepoint_days'] == recovery_tp]

    results = []
    for feat in feature_cols:

        base = baseline_data[feat].dropna().values
        inj = injury_data[feat].dropna().values
        rec = recovery_data[feat].dropna().values

        # Compute Cohen's d inline
        def d(g1, g2):
            n1, n2 = len(g1), len(g2)
            if n1 < 2 or n2 < 2:
                return 0.0
            m1, m2 = np.mean(g1), np.mean(g2)
            s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
            pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
            return 0.0 if pooled == 0 else (m1 - m2) / pooled

        d_injury = d(base, inj)
        d_recovery = d(inj, rec)

        results.append({
            'feature': feat,
            'd_injury': d_injury,
            'd_recovery': d_recovery,
            'abs_d_injury': abs(d_injury)
        })

    effect_df = pd.DataFrame(results)
    print(f"\nEffect sizes computed for {len(effect_df)} features")
    print(f"  Mean |d_injury|: {effect_df['abs_d_injury'].mean():.3f}")
    print(f"  Max |d_injury|: {effect_df['abs_d_injury'].max():.3f}")

    # Select features (baseline -> injury effect size > 0.6) and (injury -> recovery effect size > 0.2)
    # (abs(effect_df['d_recovery']) >= 0.05)
    selected = effect_df[
        (effect_df['abs_d_injury'] >= 0.8) & (abs(effect_df['d_recovery']) >= 0.2)
    ].copy()
    selected = selected.sort_values('abs_d_injury', ascending=False)
    print(f"\nFeature selection (Cohen's d >= {threshold}):")
    print(f"  Selected features: {len(selected)} / {len(effect_df)}")
    print(f"  Selection rate: {100 * len(selected) / len(effect_df):.1f}%")

    # Save plot (top 50 effect sizes)
    top = selected.sort_values('abs_d_injury', ascending=False).head(50)
    plt.figure(figsize=(10, max(8, len(top) * 0.25)))
    plt.barh(
        top['feature'].str.replace('_', ' ').str.title(),
        top['abs_d_injury'],
        color=plt.cm.Reds(np.linspace(0.9, 0.3, len(top))),
        edgecolor='darkred', linewidth=0.5
    )
    plt.gca().invert_yaxis()
    plt.xlabel("Cohen's d (Absolute)")
    plt.title("Top Features by Injury Effect Size (Baseline → 7dpi)")
    plt.tight_layout()

    output_path = Path("outputs")
    output_file = output_path / "methodB_effectsize_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved heatmap to: {output_file}")

    return selected, effect_df


def select_features_methodA(df, **kwargs):
    """
    Method A: Classification-Based Feature Selection (Random Forest).
    TODO: Implement this method.
    """
    raise NotImplementedError("Method A not yet implemented")


def select_features_methodC(df, **kwargs):
    """
    Method C: Simply an intersection of features deemed important by both Methods A and B.
    TODO: Implement this method after A.
    """
    raise NotImplementedError("Method C not yet implemented")


def preprocess_and_select(input_file, method="B", output_dir="outputs", **method_params):
    """
    Main pipeline to run data preprocessing + feature selection using specified method.
    
    Parameters:
    -----------
    input_file : Path to Phase 1 extracted features CSV
    method : Selection method ("A", "B", or "C")
    output_dir : Directory to save outputs
    **method_params : dict
        Method-specific parameters (e.g., threshold=0.6 for Method B)
    
    Returns: (selected_features_df, feature_info_df, transform_params)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*60)
    print(f"PHASE 2: FEATURE SELECTION (METHOD {method})")
    print("="*60)
    print()
    
    # Step 1: Load data with phase 1 extracted features
    print("\n" + "="*60)
    print("Step 1: Load Phase 1 data from: {input_file}")
    print("="*60)
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} samples")
    
    feature_cols = [col for col in df.columns if col not in 
                    ['animal_id', 'treatment_group', 'timepoint_days', 'timepoint_label', 
                     'run_number', 'date_code', 'filename', 'condition']]
    print(f"  Features: {len(feature_cols)}")
    print()
    
    # Check missing values
    missing_summary = df[feature_cols].isna().sum()
    missing_features = missing_summary[missing_summary > 0]
    
    print("Missing value summary:")
    print(f"  Features with missing values: {len(missing_features)}")
    if len(missing_features) > 0:
        print(f"  Max missing count: {missing_features.max()} ({100*missing_features.max()/len(df):.2f}%)")
    print()
    
    # Step 2: Mean impute by animal/timepoint for missing values
    print("\n" + "="*60)
    print("Step 2: Mean impute by animal/timepoint for missing values")
    print("="*60)
    df_imputed = mean_impute_by_group(df)
    print()

    # Step 3: Normalize, skew adjust, and scale features before feature selection
    print("\n" + "="*60)
    print("Step 3: Normalize, skew adjust, and scale features before feature selection")
    print("="*60)
    df_norm = normalize_by_animal_baseline(df_imputed)
    df_scaled, transform_params = skew_and_scale(df_norm)
    print("Preprocessing complete.\n")
    
    # Step 4: Run feature selection with chosen method
    print("\n" + "="*60)
    print(f"Step 4: Run feature selection with METHOD {method}")
    print("="*60)
    if method == "A":
        selected_features, feature_info = select_features_methodA(df_scaled, **method_params)
    elif method == "B":
        selected_features, feature_info = select_features_methodB(df_scaled, **method_params)
    elif method == "C":
        selected_features, feature_info = select_features_methodC(df_scaled, **method_params)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'A', 'B', or 'C'")
    
    print()
    
    # Save outputs
    selected_feature_names = selected_features['feature'].tolist()

    # Keep metadata columns + selected features only
    meta_cols = ['animal_id', 'treatment_group', 'timepoint_days', 'timepoint_label',
                'run_number', 'date_code', 'filename', 'condition']

    df_selected = df_imputed[meta_cols + selected_feature_names]

    # Save dataset in same format as input, but with selected features only
    selected_file = output_path / f"transformed_selected_features_method{method}.csv"
    df_selected.to_csv(selected_file, index=False)
    
    if feature_info is not None:
        full_file = output_path / f"feature_info_method{method}.csv"     # df with info for all features (e.g. effect size)
        feature_info.to_csv(full_file, index=False)
    
    print()
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total features analyzed: {len(feature_info) if feature_info is not None else 'N/A'}")
    print(f"Features selected: {len(selected_features)}")
    
    if method == "B" and len(selected_features) > 0:
        print(f"\nTop 5 features by injury effect size:")
        for i, row in selected_features.head(5).iterrows():
            print(f"  {row['feature']}: d = {row['d_injury']:.3f}")
    
    print(f"\nSaved selected dataset to: {selected_file}")
    print(f"Saved full results to: {full_file if feature_info is not None else 'N/A'}")
    print()
    
    return selected_features, feature_info, transform_params


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "outputs/extracted_features_methodA.csv"
    OUTPUT_DIR = "outputs"
    
    # Method selection
    SELECTION_METHOD = "B"
    
    # Method-specific parameters
    METHOD_PARAMS = {
        "B": {"threshold": 0.5},
        "A": {},  # Add params when Method A is implemented
        "C": {}   # Add params when Method C is implemented
    }
    
    # Run pipeline
    selected, feature_info, transform_params = preprocess_and_select(
        input_file=INPUT_FILE,
        method=SELECTION_METHOD,
        output_dir=OUTPUT_DIR,
        **METHOD_PARAMS[SELECTION_METHOD]
    )