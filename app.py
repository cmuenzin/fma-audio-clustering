import os
import random
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   # neu

from audio_processing import get_audio_files, create_feature_matrix
from clustering import perform_kmeans, perform_pca
from visualization import plot_clusters, plot_feature_importance

st.title("Unsupervised Audio-Clustering")

# Start-Button
if "start_pressed" not in st.session_state:
    st.session_state.start_pressed = False
col_filters, col_start = st.columns([3, 1])
with col_filters:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        limit    = st.number_input("Songs", min_value=10, max_value=1000, value=200, step=10)
    with col2:
        offset   = st.number_input("Offset (Sek.)", min_value=0.0, value=30.0, step=1.0)
    with col3:
        duration = st.number_input("Dauer (Sek.)", min_value=5.0, value=45.0, step=5.0)
    with col4:
        k        = st.slider("Cluster", min_value=2, max_value=20, value=3)
with col_start:
    if st.button("Start"):
        st.session_state.start_pressed = True
if not st.session_state.start_pressed:
    st.stop()

# Daten holen
audio_files = get_audio_files("fma-master/data/fma_medium", limit=limit)
st.write(f"Gefundene MP3-Dateien: {len(audio_files)}")

@st.cache_data
def get_features(files, duration, offset):
    return create_feature_matrix(files, duration=duration, offset=offset)
features_matrix, audio_files = get_features(audio_files, duration, offset)
st.write(f"Verfügbare Dateien nach Filter: {len(audio_files)}")
st.write("Feature-Matrix Shape:", features_matrix.shape)

# Clustering & PCA
clusters, _       = perform_kmeans(features_matrix, k)
pca_result, pca_model = perform_pca(features_matrix)
feature_names    = [f"MFCC {i+1}" for i in range(13)] + [f"Chroma {i+1}" for i in range(12)] + ["ZCR", "Spectral Centroid"]
df_results       = pd.DataFrame({"file_path": audio_files, "cluster": clusters})

# PCA-Plot
st.subheader("PCA-Visualisierung")
fig = plot_clusters(pca_result, clusters, pca_model, feature_names)
st.pyplot(fig)

# Songs zum Abspielen
st.subheader("Songs zum Abspielen")
cluster_ids = sorted(df_results["cluster"].unique())
cols = st.columns(len(cluster_ids))
for idx, c in enumerate(cluster_ids):
    with cols[idx]:
        songs = df_results[df_results.cluster == c].file_path.tolist()
        sample = random.sample(songs, min(3, len(songs)))
        choice = st.selectbox(f"Cluster {c}", sample, key=f"select_{c}")
        st.audio(choice)

# Feature-Importance
st.subheader("Feature Importance Ranking")
fig_imp = plot_feature_importance(pca_model, feature_names, top_n=10)
st.pyplot(fig_imp)

# ─── Ersatz für Tabelle: Pie-Charts ───
st.subheader("Cluster-Zusammenfassung (Genre-Verteilung)")
# Metadata laden
@st.cache_data
def load_track_metadata():
    df = pd.read_csv("fma-metadata/fma_metadata/tracks.csv", header=[0,1])
    return pd.DataFrame({
        "track_id": df.index,
        "genre": df[("track","genre_top")]
    })
df_tracks = load_track_metadata()

# Track-IDs zuordnen
def parse_track_id(path):
    try:
        return int(os.path.splitext(os.path.basename(path))[0])
    except:
        return None
df_results["track_id"] = df_results["file_path"].apply(parse_track_id)

# Merge & Gruppierung
merged = pd.merge(df_results, df_tracks, on="track_id", how="left")
summary = merged.groupby(["cluster","genre"]).size().reset_index(name="count")

# Pie-Charts pro Cluster
for c in cluster_ids:
    sub = summary[summary.cluster == c]
    labels = sub.genre.tolist()
    sizes  = sub["count"].tolist()
    fig, ax = plt.subplots(figsize=(3,3))
    # dunkler Hintergrund
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#121212')
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        textprops={'color':'white', 'fontsize':8}
    )
    ax.set_title(f"Cluster {c}", color='white', fontsize=12)
    st.pyplot(fig)
