import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_clusters_interactive(pca_result, clusters, pca_model, feature_names, df_results):
    """
    Erzeugt einen interaktiven Plotly-Scatterplot der PCA-Ergebnisse.
    - Die X- und Y-Achse werden anhand der höchsten absoluten Loadings in PC1 und PC2 bezeichnet.
    - Im Hover wird der file_path angezeigt.
    """
    comp1 = pca_model.components_[0]
    comp2 = pca_model.components_[1]
    idx1 = np.argmax(np.abs(comp1))
    idx2 = np.argmax(np.abs(comp2))
    xlabel = f"PCA 1 ({feature_names[idx1]})"
    ylabel = f"PCA 2 ({feature_names[idx2]})"
    
    df_plot = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    df_plot["cluster"] = clusters
    df_plot["file_path"] = df_results["file_path"].values
    
    fig = px.scatter(
         df_plot, x="PC1", y="PC2", color="cluster",
         hover_data=["file_path"],
         labels={"PC1": xlabel, "PC2": ylabel, "cluster": "Cluster"}
    )
    return fig

def plot_feature_importance(pca_model, feature_names):
    """
    Erzeugt ein Balkendiagramm (Plotly) für die Loadings der Features in PC1.
    """
    pc1_loadings = pca_model.components_[0]
    df_pc1 = pd.DataFrame({"Feature": feature_names, "Loading": pc1_loadings, "Abs": np.abs(pc1_loadings)})
    df_pc1 = df_pc1.sort_values("Abs", ascending=False)
    
    fig = px.bar(df_pc1, x="Feature", y="Loading", title="Feature Importance (PC1)", text="Loading")
    fig.update_layout(xaxis_title="Feature", yaxis_title="Loading", xaxis_tickangle=-45)
    return fig
