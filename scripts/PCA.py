import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

def PCA_analysis(path):
    df = pd.read_csv(path)

    # Separate features and labels
    features = df.columns[4:]
    X = df[features].values

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=min(X_scaled.shape[0], X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_
    print("Explained Variance:", explained_var)

    explained_var_ratio = pca.explained_variance_ratio_
    print("Explained Variance Ratio:", explained_var_ratio)

    # Scree plot
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(explained_var)+1), explained_var, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Eigenvalue")
    plt.title("Scree Plot")
    plt.grid(True)
    plt.savefig(f"scree_plot.png")
    plt.close()

    # Cumulative explained variance plot
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(explained_var_ratio)+1), explained_var_ratio.cumsum(), marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance")
    plt.grid(True)
    plt.savefig(f"cumulative_scree_plot.png")
    plt.close()

    # 2D PCA
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    df_pca = pd.DataFrame({
        "PC1": X_pca_2d[:,0],
        "PC2": X_pca_2d[:,1],
        "condition": df["condition"]
    })

    # 2D Plot
    plt.figure(figsize=(8,6))
    for cond in df_pca["condition"].unique():
        subset = df_pca[df_pca["condition"] == cond]
        plt.scatter(subset["PC1"], subset["PC2"], label=cond)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of Animal Gait Features")
    plt.legend()
    plt.savefig(f"pca_2d_plot.png")
    plt.close()

if __name__ == "__main__":
    path = f'avg_features_per_animal.csv'
    PCA_analysis(path)