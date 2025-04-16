import matplotlib.pyplot as plt
import numpy as np

def plot_clusters(pca_result, clusters, pca_model, feature_names):
    """
    Zeichnet einen 2D PCA-Scatterplot (matplotlib) mit Achsenbeschriftung inkl. Top-Feature.
    """
    comp1 = pca_model.components_[0]
    comp2 = pca_model.components_[1]
    idx1 = np.argmax(np.abs(comp1))
    idx2 = np.argmax(np.abs(comp2))
    xlabel = f"PCA 1 ({feature_names[idx1]})"
    ylabel = f"PCA 2 ({feature_names[idx2]})"

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        pca_result[:, 0], pca_result[:, 1],
        c=clusters, cmap='viridis', alpha=0.7
    )
    ax.set_title("PCA-Plot der Audio-Cluster")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster-ID")
    fig.tight_layout()
    return fig

def plot_feature_importance(pca_model, feature_names):
    """
    Erstellt ein Balkendiagramm der PC1-Loadings (matplotlib).
    """
    pc1_loadings = pca_model.components_[0]
    abs_pc1 = np.abs(pc1_loadings)
    sorted_idx = np.argsort(abs_pc1)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_loadings = pc1_loadings[sorted_idx]

    fig, ax = plt.subplots(figsize=(max(6, len(feature_names)*0.3), 4))
    ax.bar(range(len(sorted_features)), sorted_loadings)
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels(sorted_features, rotation=90)
    ax.set_ylabel("Loading")
    ax.set_title("Feature Importance (PC1)")
    fig.tight_layout()
    return fig
