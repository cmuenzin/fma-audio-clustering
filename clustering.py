import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def perform_kmeans(features_matrix, k, random_state=42):
    """
    Führt k-Means-Clustering mit k Clustern auf der gegebenen Feature-Matrix durch.
    Gibt die Cluster-Zuordnungen und das trainierte k-Means Modell zurück.
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    clusters = kmeans.fit_predict(features_matrix)
    return clusters, kmeans

def perform_pca(features_matrix, n_components=2):
    """
    Reduziert die Dimensionalität der Feature-Matrix mittels PCA auf n_components.
    Gibt die transformierten Daten und das PCA-Modell zurück.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_matrix)
    return pca_result, pca
