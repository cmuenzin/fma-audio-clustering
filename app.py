import os
import streamlit as st
import numpy as np
import pandas as pd

from audio_processing import get_audio_files, create_feature_matrix
from clustering import perform_kmeans, perform_pca
from visualization import plot_clusters, plot_feature_importance
from visualization_pies import plot_genre_pies

st.title("Unsupervised Audio‑Clustering")

st.markdown("""
Diese App führt eine **unüberwachte Cluster‑Analyse** auf Audio‑Dateien durch.  
Dabei werden aus jedem Song akustische Merkmale extrahiert,  
per PCA reduziert und dann mit K‑Means gruppiert.
""")

# --- Start-Button + Filter ---
if "start_pressed" not in st.session_state:
    st.session_state.start_pressed = False

col_f, col_s = st.columns([3,1])
with col_f:
    c1, c2, c3 = st.columns(3)
    with c1:
        limit    = st.number_input("Songs", min_value=10, max_value=1000, value=200, step=10)
    with c2:
        offset   = st.number_input("Offset (Sek.)", min_value=0.0, value=30.0, step=1.0)
    with c3:
        duration = st.number_input("Dauer (Sek.)", min_value=5.0, value=45.0, step=5.0)
with col_s:
    if st.button("Start"):
        st.session_state.start_pressed = True

if not st.session_state.start_pressed:
    st.stop()

# --- Dateien & Features ---
@st.cache_data
def load_audio_files(path, limit):
    return get_audio_files(path, limit)

audio_files = load_audio_files("fma-master/data/fma_medium", limit)
st.write(f"Gefundene MP3‑Dateien: **{len(audio_files)}**")

@st.cache_data
def get_features(files, duration, offset):
    return create_feature_matrix(files, duration=duration, offset=offset)

features_matrix, audio_files = get_features(audio_files, duration, offset)
st.write(f"Verfügbare Dateien nach Filter: **{len(audio_files)}**")
st.write("Feature‑Matrix Shape (vor Filterung):", features_matrix.shape)

# --- Feature-Auswahl ---
all_features = [f"MFCC {i+1}" for i in range(13)] + \
               [f"Chroma {i+1}" for i in range(12)] + \
               ["ZCR", "Spectral Centroid"]

st.subheader("Feature-Auswahl")
selected_features = st.multiselect(
    "Welche Features sollen verwendet werden?",
    options=all_features,
    default=all_features
)

feature_names = all_features
feature_indices = [i for i, name in enumerate(feature_names) if name in selected_features]
features_matrix = features_matrix[:, feature_indices]
feature_names = selected_features
st.write("Feature‑Matrix Shape (nach Filterung):", features_matrix.shape)

# --- Metadaten laden ---
@st.cache_data
def load_metadata():
    metadata_path = "fma_metadata/tracks.csv"
    if not os.path.exists(metadata_path):
        st.warning(f"Metadaten nicht gefunden unter `{metadata_path}`. "
                   "Bitte stelle sicher, dass die Datei vorhanden ist.")
        return pd.DataFrame(columns=["track_id", "genre"])
    
    df = pd.read_csv(metadata_path, header=[0,1], low_memory=False)
    return pd.DataFrame({
        "track_id": df.index,
        "genre": df[("track","genre_top")]
    })

df_tracks = load_metadata()
n_genres = df_tracks["genre"].nunique() if not df_tracks.empty else 5

# --- Cluster-Wahl ---
st.subheader("Cluster-Einstellungen")
cluster_mode = st.radio("Cluster-Wahl", ["Manuell", "Genres als Clusteranzahl"])
if cluster_mode == "Manuell":
    k = st.slider("Cluster‑Anzahl", min_value=2, max_value=20, value=3)
else:
    k = n_genres
    st.markdown(f"**Cluster-Anzahl gesetzt auf Anzahl Genres: {k}**")

# --- Clustering & PCA ---
clusters, _           = perform_kmeans(features_matrix, k)
pca_result, pca_model = perform_pca(features_matrix)
df_results           = pd.DataFrame({"file_path": audio_files, "cluster": clusters})
cluster_ids          = sorted(df_results["cluster"].unique())

# --- PCA-Plot ---
st.subheader("PCA‑Visualisierung")
st.pyplot(plot_clusters(pca_result, clusters, pca_model, feature_names),
           use_container_width=True)

# --- Audio-Player ---
st.subheader("Songs zum Abspielen")
cols = st.columns(len(cluster_ids))
for i, c in enumerate(cluster_ids):
    with cols[i]:
        songs = df_results[df_results.cluster == c].file_path.tolist()
        choice = st.selectbox(f"Cluster {c}", songs,
                              format_func=lambda x: os.path.basename(x),
                              key=f"sel_{c}")
        st.audio(choice)

# --- Feature Importance ---
st.subheader("Feature Importance Ranking")
st.pyplot(plot_feature_importance(pca_model, feature_names, top_n=5),
           use_container_width=True)

# --- Genre-Zuordnung ---
st.subheader("Cluster‑Zusammenfassung (Genre‑Verteilung)")

df_results["track_id"] = df_results["file_path"].apply(
    lambda p: int(os.path.splitext(os.path.basename(p))[0]) if p else None
)

fig = plot_genre_pies(df_results, df_tracks, cluster_ids,
                      cols=3, pie_size=300)
st.plotly_chart(fig, use_container_width=True)
