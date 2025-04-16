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
    # Noch dunkleres Grau
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')

    scatter = ax.scatter(
        pca_result[:, 0], pca_result[:, 1],
        c=clusters, cmap='viridis', alpha=0.7
    )
    ax.set_title("PCA-Plot der Audio-Cluster", color='white')
    ax.set_xlabel(xlabel, color='white')
    ax.set_ylabel(ylabel, color='white')
    # Achsenticks in Weiß
    ax.tick_params(colors='white')

    # Keine Farbskala mehr

    fig.tight_layout()
    return fig

def plot_feature_importance(pca_model, feature_names, top_n=10):
    """
    Erstellt ein horizontales Balkendiagramm der Top-N wichtigsten Features (PC1) mit dunklem Hintergrund.
    """
    pc1_loadings = np.abs(pca_model.components_[0])
    sorted_idx    = np.argsort(pc1_loadings)[::-1][:top_n]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_values   = pc1_loadings[sorted_idx]

    # ─── HIER FIGURESIZE ANPASSEN ───
    # Aktuell: Breite 3 Zoll, Höhe 0.2*top_n + 0.5 Zoll
    fig, ax = plt.subplots(figsize=(3, 0.2 * top_n + 0.5))

    # Hintergrund sehr dunkel
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')

    bars = ax.barh(
        range(len(sorted_features)),
        sorted_values,
        color='teal'
    )

    # ─── HIER TEXTGRÖSSEN ANPASSEN ───
    # Achsenbeschriftungen:
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features,
                       color='white',
                       fontsize=8)               # <-- hier fontsize ändern
    ax.invert_yaxis()  # höchste Wichtigkeit oben

    ax.set_xlabel(
        "Feature Importance (abs. PC1 Loading)",
        color='white',
        fontsize=10                      # <-- hier fontsize ändern
    )
    ax.set_title(
        "Top Features (PCA Component 1)",
        color='white',
        fontsize=12                      # <-- hier fontsize ändern
    )

    # X‑Achsen‑Ticks (Zahlen) in Weiß und eigene Größe
    ax.tick_params(
        axis='x',
        colors='white',
        labelsize=8                       # <-- hier fontsize ändern
    )

    fig.tight_layout()
    return fig