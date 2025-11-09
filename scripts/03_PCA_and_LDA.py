"""
03_PCA_and_LDA.py

Phase 3: Performs PCA and LDA using transformed, cleaned, selected features from Phase 2
Saves scores and visualizations

Input: transformed_selected_features_method{A/B/C}.csv
Output: 
    PCA_results\ folder with PCA_scores and PC1 plots
    LDA_results\ folder with LD1_scores and LD1_plots

Expects folder called "outputs" with input csv (run 01_extract_features.py then 02_transform_select_features.py) 
and to save PCA_results\ and LDA_results\
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def run_pca_analysis(df: pd.DataFrame, n_components: int = None, output_folder: str = 'outputs') -> tuple:
    """
    Perform PCA on feature columns, save PCA scores and visualizations in 'PCA_results'.
    
    Returns: pca_model, pc_scores_df, explained_variance, feature_cols
    """
    meta_cols = ['animal_id', 'treatment_group', 'timepoint_days', 
                 'timepoint_label', 'run_number', 'date_code', 
                 'filename', 'condition']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    # Compute PCA
    X = df[feature_cols].values
    pca_model = PCA(n_components=n_components)
    pc_scores = pca_model.fit_transform(X)
    
    # Create DataFrame with PC scores + metadata
    pc_cols = [f'PC{i+1}' for i in range(pc_scores.shape[1])]
    pc_df = pd.DataFrame(pc_scores, columns=pc_cols, index=df.index)
    for col in meta_cols:
        pc_df[col] = df[col].values
    pc_df = pc_df[meta_cols + [c for c in pc_df.columns if c not in meta_cols]] # put metadata first
    
    explained_var = pca_model.explained_variance_ratio_
    
    # Save PCA results
    output_dir = Path(output_folder) / 'PCA_results'
    output_dir.mkdir(exist_ok=True, parents=True)
    pc_df.to_csv(output_dir / 'PC_scores.csv', index=False)
    
    # Plotting colors for treatment groups
    colors = {'Vehicle': '#1f77b4', 'LoDose': '#ff7f0e', 'HiDose': '#2ca02c'}
    
    # Plot 1: PC1 with individual runs as scatter points
    fig, ax = plt.subplots(figsize=(10, 6))
    pc_label = 'PC1'
    for treatment in pc_df['treatment_group'].unique():
        data = pc_df[pc_df['treatment_group'] == treatment]
        for tp in data['timepoint_days'].unique():
            tp_data = data[data['timepoint_days'] == tp]
            jitter = np.random.normal(0, 1, len(tp_data))
            ax.scatter(tp_data['timepoint_days'] + jitter, tp_data[pc_label],
                    alpha=0.3, s=30, color=colors.get(treatment, 'gray'))
        mean_trajectory = data.groupby('timepoint_days')[pc_label].mean().reset_index()
        ax.plot(mean_trajectory['timepoint_days'], mean_trajectory[pc_label],
                marker='o', linewidth=2, markersize=8, label=treatment,
                color=colors.get(treatment, 'gray'))
    ax.set_xlabel('Timepoint (days)', fontsize=12)
    ax.set_ylabel(f'{pc_label} ({explained_var[1]:.1%} variance)', fontsize=12)
    ax.set_title(f'{pc_label} Recovery Curves by Treatment Group', fontsize=14, fontweight='bold')
    ax.legend(title='Treatment Group')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{pc_label}_recovery_curves_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Same as Plot 1, but without individual runs as scatter points (just recovery lines)
    fig, ax = plt.subplots(figsize=(10, 6))
    pc_label = 'PC1'
    for treatment in pc_df['treatment_group'].unique():
        data = pc_df[pc_df['treatment_group'] == treatment]
        mean_trajectory = data.groupby('timepoint_days')[pc_label].mean().reset_index()
        ax.plot(mean_trajectory['timepoint_days'], mean_trajectory[pc_label],
                marker='o', linewidth=2, markersize=8, label=treatment,
                color=colors.get(treatment, 'gray'))
    ax.set_xlabel('Timepoint (days)', fontsize=12)
    ax.set_ylabel(f'{pc_label} ({explained_var[1]:.1%} variance)', fontsize=12)
    ax.set_title(f'{pc_label} Recovery Curves by Treatment Group', fontsize=14, fontweight='bold')
    ax.legend(title='Treatment Group')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{pc_label}_recovery_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: PC1 top loadings
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for i, ax in enumerate(axes):
        loadings = pd.Series(pca_model.components_[i], index=feature_cols)
        top_loadings = pd.concat([loadings.nlargest(10), loadings.nsmallest(10)]).sort_values()
        colors_load = ['#d62728' if x < 0 else '#2ca02c' for x in top_loadings.values]
        ax.barh(range(len(top_loadings)), top_loadings.values, color=colors_load)
        ax.set_yticks(range(len(top_loadings)))
        ax.set_yticklabels([label.replace('_', ' ') for label in top_loadings.index], fontsize=9)
        ax.set_xlabel('Loading Value', fontsize=11)
        ax.set_title(f'Top 20 Loadings for PC{i+1} ({explained_var[i]:.1%} variance)',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/PC_loadings.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"PC scores + visualizations saved to {output_dir}")
    return pca_model, pc_df, explained_var, feature_cols


def run_lda_analysis(df: pd.DataFrame, output_folder: str = "outputs", pca_variance_threshold: float = 0.8,
                     treatment_weight: float = 0.3, pca_model: PCA = None) -> tuple:
    """
    Perform LDA on PCA results, PCA scores and visualizations in 'LD1_results'.
    
    Parameters:
    - df: preprocessed DataFrame (normalized, scaled)
    - output_dir: folder to save LDA plots
    - pca_variance_threshold: PCA variance threshold for LDA input
    - treatment_weight: weight for treatment separation vs injury separation
    - pca_model: precomputed PCA object to avoid recomputation
    
    Returns:
    - lda_combined: combined LDA object
    - ld_df: DataFrame with LD1 scores + metadata
    - feature_cols: list of feature columns used
    - n_pca_components: number of PCA components used for LDA
    """
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np
    import pandas as pd
    
    meta_cols = ['animal_id', 'treatment_group', 'timepoint_days', 
                 'timepoint_label', 'run_number', 'date_code', 
                 'filename', 'condition']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols].values

    # ========================================================
    # Step 1: Get enough PCA components so cumulative explained variance > threshold
    # ========================================================
    if pca_model is None:
        print("\nNo PCA provided. Computing PCA for LDA...")
        pca_full = PCA()
        pca_full.fit(X)
    else:
        pca_full = pca_model

    # Determine number of components for threshold variance
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    if pca_variance_threshold is None:
        n_pca_components = X.shape[1]  # all features / components
    else:
        cumulative_var = np.cumsum(pca_full.explained_variance_ratio_) 
        n_pca_components = np.argmax(cumulative_var >= pca_variance_threshold) + 1
    X_pca = pca_full.transform(X)[:, :n_pca_components]

    print(f"PCA preprocessing for LDA:")
    print(f"  Components used: {n_pca_components} ({cumulative_var[n_pca_components-1]:.2%} variance)")

    # ========================================================
    # Step 2: LDA - Maximize separation injury axis (0 vs 7 days)
    # ========================================================
    train_mask_injury = df['timepoint_days'].isin([0, 7])
    X_train_injury = X_pca[train_mask_injury]
    y_train_injury = df.loc[train_mask_injury, 'timepoint_days'].values
    lda_injury = LinearDiscriminantAnalysis(n_components=1)
    lda_injury.fit(X_train_injury, y_train_injury)
    injury_direction = lda_injury.scalings_[:, 0]

    # ========================================================
    # Step 3: LDA - Maximize separation between treatment groups (treatment axis) at 42 days
    # ========================================================
    train_mask_treatment = df['timepoint_days'] == 42
    X_train_treatment = X_pca[train_mask_treatment]
    y_train_treatment = df.loc[train_mask_treatment, 'treatment_group'].values
    lda_treatment = LinearDiscriminantAnalysis(n_components=1)
    lda_treatment.fit(X_train_treatment, y_train_treatment)
    treatment_direction = lda_treatment.scalings_[:, 0]

    # ========================================================
    # Step 4: Combine directions - weigh both LDA coeffs from injury axis (0.7) and treatment group axis (0.3)
    # ========================================================
    combined_direction = (1 - treatment_weight) * injury_direction + treatment_weight * treatment_direction
    combined_direction /= np.linalg.norm(combined_direction)
    ld1_scores = X_pca @ combined_direction

    class CombinedLDA:
        def __init__(self, scalings, pca_model, n_pca_comp, lda_injury, lda_treatment):
            self.scalings_ = scalings.reshape(-1, 1)
            self.pca_model = pca_model
            self.n_pca_components = n_pca_comp
            self.lda_injury = lda_injury
            self.lda_treatment = lda_treatment

    lda_model = CombinedLDA(combined_direction, pca_full, n_pca_components, lda_injury, lda_treatment)

    # ========================================================
    # Step 5: Create and save LD1 DataFrame
    # ========================================================
    ld_df = pd.DataFrame({'LD1': ld1_scores.flatten()}, index=df.index)
    for col in meta_cols:
        ld_df[col] = df[col].values

    # Flip LD1 for interpretability
    if ld_df[ld_df['timepoint_days'] == 0]['LD1'].mean() < ld_df[ld_df['timepoint_days'] == 7]['LD1'].mean():
        ld_df['LD1'] = -ld_df['LD1']
        lda_model.scalings_[:, 0] = -lda_model.scalings_[:, 0]

    # Save LD1 results
    output_dir = Path(output_folder) / 'LDA_results'
    output_dir.mkdir(exist_ok=True, parents=True)
    ld_df.to_csv(output_dir / 'LD_scores.csv', index=False)
    
    # ========================================================
    # Step 6: Plot visualizations
    # ========================================================
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    colors = {'Vehicle': '#1f77b4', 'LoDose': '#ff7f0e', 'HiDose': '#2ca02c'}

    # LD1 scatter with all runs
    fig, ax = plt.subplots(figsize=(10, 6))
    for treatment in ld_df['treatment_group'].unique():
        data = ld_df[ld_df['treatment_group'] == treatment]
        for tp in data['timepoint_days'].unique():
            tp_data = data[data['timepoint_days'] == tp]
            jitter = np.random.normal(0, 1, len(tp_data))
            ax.scatter(tp_data['timepoint_days'] + jitter, tp_data['LD1'],
                       alpha=0.3, s=30, color=colors.get(treatment, 'gray'))
        mean_trajectory = data.groupby('timepoint_days')['LD1'].mean().reset_index()
        ax.plot(mean_trajectory['timepoint_days'], mean_trajectory['LD1'], marker='o', linewidth=2, markersize=8,
                label=treatment, color=colors.get(treatment, 'gray'))
    ax.set_xlabel('Timepoint (days)')
    ax.set_ylabel('LD1 Score')
    ax.set_title('LD1 Recovery Curves (All Runs)')
    ax.legend(title='Treatment Group')
    plt.tight_layout()
    plt.savefig(output_dir / 'LD1_scatter_allruns.png', dpi=300)
    plt.close()

    # LD1 animal averages
    fig, ax = plt.subplots(figsize=(12, 7))
    animal_avg = ld_df.groupby(['animal_id', 'treatment_group', 'timepoint_days'])['LD1'].mean().reset_index()
    for treatment in animal_avg['treatment_group'].unique():
        treatment_data = animal_avg[animal_avg['treatment_group'] == treatment]
        for animal in treatment_data['animal_id'].unique():
            animal_data = treatment_data[treatment_data['animal_id'] == animal]
            ax.plot(animal_data['timepoint_days'], animal_data['LD1'], alpha=0.3, linewidth=1, color=colors.get(treatment, 'gray'))
        mean_trajectory = treatment_data.groupby('timepoint_days')['LD1'].mean().reset_index()
        ax.plot(mean_trajectory['timepoint_days'], mean_trajectory['LD1'], marker='o', linewidth=3, markersize=10,
                label=treatment, color=colors.get(treatment, 'gray'), zorder=10)
    ax.set_xlabel('Timepoint (days)')
    ax.set_ylabel('LD1 Score')
    ax.set_title('LD1 Recovery Curves (Animal Averages)')
    ax.legend(title='Treatment Group')
    plt.tight_layout()
    plt.savefig(output_dir / 'LD1_animal_trajectories.png', dpi=300)
    plt.close()

    # LD1 loadings
    pca_rotation = pca_full.components_[:n_pca_components, :].T
    lda_loadings_original = pca_rotation @ lda_model.scalings_[:, 0]
    loadings_ld1 = pd.Series(lda_loadings_original, index=feature_cols)
    top_ld1 = pd.concat([loadings_ld1.nlargest(15), loadings_ld1.nsmallest(15)]).sort_values()
    fig, ax = plt.subplots(figsize=(10, 10))
    colors_ld1 = ['#d62728' if x < 0 else '#2ca02c' for x in top_ld1.values]
    ax.barh(range(len(top_ld1)), top_ld1.values, color=colors_ld1)
    ax.set_yticks(range(len(top_ld1)))
    ax.set_yticklabels([l.replace('_', ' ') for l in top_ld1.index])
    ax.set_xlabel('LD1 Coefficient')
    ax.set_title('Top 30 Feature Loadings for LD1')
    plt.tight_layout()
    plt.savefig(output_dir / 'LD1_loadings.png', dpi=300)
    plt.close()

    print(f"\nSaved LD1 scores + visualizations to {output_dir}")

    return lda_model, ld_df, feature_cols, n_pca_components


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    INPUT_FILE = Path("outputs/transformed_selected_features_methodB.csv")
    OUTPUT_DIR = Path("outputs")
    LDA_DIR = OUTPUT_DIR / "LDA_plots"
    
    print("="*60)
    print("MOTORATER PCA + LDA PIPELINE")
    print("="*60)
    print()
    
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    
    # Load data
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded feature dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    
    # Step 3: PCA
    print("\n" + "="*60)
    print("STEP 1: Principal Component Analysis")
    print("="*60)
    pca, pc_df, explained_var, feature_cols_pca = run_pca_analysis(
        df, n_components=None, output_folder=OUTPUT_DIR
    )
    
    # Step 4: LDA (consolidated with visualization)
    print("\n" + "="*60)
    print("STEP 2: Linear Discriminant Analysis (with plots)")
    print("="*60)
    lda, ld_df, feature_cols_lda, n_pca_comp = run_lda_analysis(
        df, output_folder=OUTPUT_DIR,
        pca_variance_threshold=0.98, treatment_weight=0.3, pca_model=pca
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nOutputs saved to:")
    # print(f"  - Normalized features: {df}")
    print(f"  - PC scores + plots: {OUTPUT_DIR}\\PCA_results\\")
    print(f"  - LD scores: {OUTPUT_DIR}\\LD1_results\\")