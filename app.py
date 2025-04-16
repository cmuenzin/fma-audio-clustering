import os
import random
import streamlit as st
import numpy as np
import pandas as pd

from audio_processing import get_audio_files, create_feature_matrix
from clustering import perform_kmeans, perform_pca
from visualization import plot_clusters_interactive, plot_feature_importance

st.title("Unsupervised Audio-Clustering")

# Warten mit der Auswertung, bis der User auf "Start" drückt.
if "start_pressed" not in st.session_state:
    st.session_state.start_pressed = False

col_filters, col_start_button = st.columns([3, 1])
with col_filters:
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            limit = st.number_input("Songs", min_value=10, max_value=1000, value=50, step=10)
        with col2:
            offset = st.number_input("Offset (Sek.)", min_value=0.0, value=10.0, step=1.0)
        with col3:
            duration = st.number_input("Dauer (Sek.)", min_value=5.0, value=20.0, step=5.0)
        with col4:
            k = st.slider("Cluster", min_value=2, max_value=20, value=5)
with col_start_button:
    if st.button("Start"):
        st.session_state.start_pressed = True

if not st.session_state.start_pressed:
    st.stop()

# Dateien laden – effizient per glob & random sampling (siehe audio_processing.py)
audio_base_path = "fma-master/data/fma_medium"
audio_files = get_audio_files(audio_base_path, limit=limit)
st.write(f"Gefundene MP3-Dateien: {len(audio_files)}")

@st.cache_data
def get_features(files, duration, offset):
    return create_feature_matrix(files, duration=duration, offset=offset)
features_matrix = get_features(audio_files, duration, offset)
st.write("Shape der Feature-Matrix:", features_matrix.shape)

# Clustering durchführen
clusters, _ = perform_kmeans(features_matrix, k)
df_results = pd.DataFrame({"file_path": audio_files, "cluster": clusters})

# Cluster Plot und Tabelle nebeneinander
col_vis, col_table = st.columns([2, 1])
with col_vis:
    pca_result, pca_model = perform_pca(features_matrix)
    feature_names = [f"MFCC {i+1}" for i in range(13)] + \
                    [f"Chroma {i+1}" for i in range(12)] + ["ZCR", "Spectral Centroid"]
    st.subheader("PCA-Visualisierung")
    fig_cluster = plot_clusters_interactive(pca_result, clusters, pca_model, feature_names, df_results)
    st.plotly_chart(fig_cluster, use_container_width=True)
with col_table:
    st.subheader("Cluster-Zuordnungen")
    st.dataframe(df_results)

# (Der Block zum Track-Abspielen entfällt, da interaktive Punkte per Hover bereits Zuordnungen liefern)

# Genre-Metadaten laden (Pfad ggf. anpassen)
@st.cache_data
def load_track_metadata():
    df = pd.read_csv("fma-metadata/fma_metadata/tracks.csv", header=[0, 1])
    df_result = pd.DataFrame({
        "track_id": df.index,
        "genre": df[("track", "genre_top")]
    })
    return df_result
df_tracks = load_track_metadata()

st.subheader("Cluster-Zusammenfassung")
def parse_track_id_from_path(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(filename)
    except ValueError:
        return None

def get_cluster_summary(df_results, df_tracks, features_matrix):
    """
    Erzeugt eine Zusammenfassung pro Cluster mit:
      - Cluster
      - Song Count
      - Genres (sorted): Genre-Häufigkeit als String
      - Mean ZCR (Feld f25)
      - Mean Spectral Centroid (Feld f26)
    """
    df_genres = df_results.copy()
    df_genres["track_id"] = df_genres["file_path"].apply(parse_track_id_from_path)
    merged = pd.merge(df_genres, df_tracks, on="track_id", how="left")
    genre_group = merged.groupby(["cluster", "genre"]).size().reset_index(name="count")
    
    genre_summary = {}
    for c in sorted(genre_group["cluster"].unique()):
        temp = genre_group[genre_group["cluster"] == c].sort_values("count", ascending=False)
        genre_list = []
        for _, row in temp.iterrows():
            genre_name = str(row["genre"])
            count = row["count"]
            genre_list.append(f"{genre_name} ({count})")
        genre_summary[c] = ", ".join(genre_list)
    
    df_feat = pd.DataFrame(features_matrix, columns=[f"f{i}" for i in range(features_matrix.shape[1])])
    df_feat["cluster"] = df_results["cluster"].values
    stats = df_feat.groupby("cluster").agg(
        song_count=("cluster", "size"),
        mean_zcr=("f25", "mean"),
        mean_spec_centroid=("f26", "mean")
    ).reset_index()
    
    stats["genres"] = stats["cluster"].apply(lambda c: genre_summary.get(c, ""))
    stats = stats.rename(columns={
        "cluster": "Cluster",
        "song_count": "Song Count",
        "genres": "Genres (sorted)",
        "mean_zcr": "Mean ZCR",
        "mean_spec_centroid": "Mean Spectral Centroid"
    })
    
    return stats

df_cluster_summary = get_cluster_summary(df_results, df_tracks, features_matrix)
st.dataframe(df_cluster_summary)

st.subheader("Feature Importance List")
fig_importance = plot_feature_importance(pca_model, feature_names)
st.plotly_chart(fig_importance, use_container_width=True)
