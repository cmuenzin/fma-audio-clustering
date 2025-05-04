import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Anzahl der Cluster anhand der Menge der Genres festlegen
# unique_genres = df['genre'].nunique()
# clusters, model = perform_kmeans(features_matrix, k=unique_genres)

def perform_kmeans(features_matrix, k, random_state=42):
    """
    Führe k-Means-Clustering durch und gib Labels und das Modell zurück.
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    clusters = kmeans.fit_predict(features_matrix)
    return clusters, kmeans

def perform_pca(features_matrix, n_components=2):
    """
    Reduziere Dimensionen mittels PCA und gib transformierte Daten sowie Modell zurück.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_matrix)
    return pca_result, pca
