# visualization.py
import matplotlib.pyplot as plt
import numpy as np

def plot_clusters(pca_result, clusters, pca_model, feature_names):
    """
    Zeichnet einen 2D PCA-Scatterplot mit sehr dunklem Hintergrund.
    """
    comp1 = pca_model.components_[0]
    comp2 = pca_model.components_[1]
    idx1 = np.argmax(np.abs(comp1))
    idx2 = np.argmax(np.abs(comp2))
    xlabel = f"PCA 1 ({feature_names[idx1]})"
    ylabel = f"PCA 2 ({feature_names[idx2]})"

    fig, ax = plt.subplots()
    # Hintergrund noch dunkler
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')

    ax.scatter(
        pca_result[:, 0], pca_result[:, 1],
        c=clusters, cmap='viridis', alpha=0.7
    )
    ax.set_title("PCA-Plot der Audio-Cluster", color='white')
    ax.set_xlabel(xlabel, color='white')
    ax.set_ylabel(ylabel, color='white')
    ax.tick_params(colors='white')

    fig.tight_layout()
    return fig


def plot_feature_importance(pca_model, feature_names, top_n=5):
    """
    Erstellt ein horizontales Balkendiagramm der Top-N wichtigsten Features (PC1)
    mit mittlerer Höhe, ohne Rahmenlinien und ohne x‑Achse.
    """
    # Ladewerte
    pc1 = np.abs(pca_model.components_[0])
    idx = np.argsort(pc1)[::-1][:top_n]
    features = [feature_names[i] for i in idx]
    values   = pc1[idx]

    # Mittlere Höhe: Breite 4", Höhe = 0.2*top_n + 0.5
    fig, ax = plt.subplots(figsize=(4, 0.2 * top_n + 0.5))
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')

    ax.barh(range(len(features)), values, color='teal')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, color='white', fontsize=4)
    ax.invert_yaxis()

    # Rahmen entfernen
    for spine in ax.spines.values():
        spine.set_visible(False)
    # x‑Achse verbergen
    ax.xaxis.set_visible(False)

    # Titel noch aufführen
    ax.set_title("Top Features (PCA Component 1)", color='white', fontsize=6)

    fig.tight_layout()
    return fig
