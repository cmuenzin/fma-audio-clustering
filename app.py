# app.py
import os
import random
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from audio_processing import get_audio_files, create_feature_matrix
from clustering import perform_kmeans, perform_pca
from visualization import plot_clusters, plot_feature_importance

st.title("Unsupervised Audio‑Clustering")

# Kurze Einführung
st.markdown("""
Diese App führt eine **unüberwachte Cluster‑Analyse** auf Audio‑Dateien durch.  
Dabei werden aus jedem Song akustische Merkmale (MFCCs, Chroma, ZCR, Spectral Centroid) extrahiert,  
per PCA auf zwei Dimensionen reduziert und anschließend mit K‑Means in Cluster gruppiert.
""")

# Start‑Button
if "start_pressed" not in st.session_state:
    st.session_state.start_pressed = False
col_filters, col_start = st.columns([3, 1])
with col_filters:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        limit    = st.number_input("Songs", min_value=10, max_value=1000, value=200, step=10)
    with c2:
        offset   = st.number_input("Offset (Sek.)", min_value=0.0, value=30.0, step=1.0)
    with c3:
        duration = st.number_input("Dauer (Sek.)", min_value=5.0, value=45.0, step=5.0)
    with c4:
        k        = st.slider("Cluster‑Anzahl", min_value=2, max_value=20, value=3)
with col_start:
    if st.button("Start"):
        st.session_state.start_pressed = True
if not st.session_state.start_pressed:
    st.stop()

# Audio‑Dateien laden und Features extrahieren
audio_files = get_audio_files("fma-master/data/fma_medium", limit=limit)
st.write(f"Gefundene MP3‑Dateien: **{len(audio_files)}**")

@st.cache_data
def get_features(files, duration, offset):
    return create_feature_matrix(files, duration=duration, offset=offset)

features_matrix, audio_files = get_features(audio_files, duration, offset)
st.write(f"Verfügbare Dateien nach Filter: **{len(audio_files)}**")  
st.write("Feature‑Matrix Shape:", features_matrix.shape)

# Clustering & PCA
clusters, _          = perform_kmeans(features_matrix, k)
pca_result, pca_model= perform_pca(features_matrix)
feature_names       = [f"MFCC {i+1}" for i in range(13)] + \
                      [f"Chroma {i+1}" for i in range(12)] + \
                      ["ZCR", "Spectral Centroid"]
df_results          = pd.DataFrame({"file_path": audio_files, "cluster": clusters})
cluster_ids         = sorted(df_results["cluster"].unique())

# --- PCA‑Visualisierung ---
st.subheader("PCA‑Visualisierung")
st.markdown("""
- **Was?** Jeder Punkt ist ein Song, dargestellt auf zwei Hauptkomponenten (PCA).  
- **Wie?** Die Farben kodieren die von K‑Means gefundenen Cluster.  
- **Warum?** So sieht man auf einen Blick, welche Songs akustisch ähnlich sind.
""")
st.pyplot(plot_clusters(pca_result, clusters, pca_model, feature_names))

# --- Songs zum Abspielen ---
st.subheader("Songs zum Abspielen")
st.markdown("Wählen Sie einen beliebigen Song pro Cluster aus (Dateiname) und spielen Sie ihn ab.")
cols = st.columns(len(cluster_ids))
for i, c in enumerate(cluster_ids):
    with cols[i]:
        # alle Songs im Cluster als Optionen
        songs = df_results[df_results.cluster == c].file_path.tolist()
        choice = st.selectbox(
            f"Cluster {c}",
            options=songs,
            format_func=lambda x: os.path.basename(x),
            key=f"sel_{c}"
        )
        st.audio(choice)

# --- Feature Importance Ranking ---
st.subheader("Feature Importance Ranking")
st.markdown("""
Dieses Balkendiagramm zeigt die **Top‑5 Audio‑Features**,  
die am stärksten zur Trennung der Cluster (PC1) beitragen.  
Je länger der Balken, desto größer der Einfluss.
""")
st.pyplot(plot_feature_importance(pca_model, feature_names, top_n=5))

# --- Cluster‑Zusammenfassung als Gradient‑Pies ---
st.subheader("Cluster‑Zusammenfassung (Genre‑Verteilung)")
st.markdown("""
Die folgenden **Tortendiagramme** veranschaulichen für jedes Cluster,  
welche Musik‑Genres dort dominieren.  
Größeres Segment = häufigeres Genre, Farbverlauf von Cluster‑Farbe → Grau.
""")
@st.cache_data
def load_metadata():
    df = pd.read_csv("fma-metadata/fma_metadata/tracks.csv", header=[0,1])
    return pd.DataFrame({
        "track_id": df.index,
        "genre": df[("track","genre_top")]
    })
df_tracks = load_metadata()

def parse_id(path):
    try:
        return int(os.path.splitext(os.path.basename(path))[0])
    except:
        return None

df_results["track_id"] = df_results["file_path"].apply(parse_id)
summary = pd.merge(df_results, df_tracks, on="track_id", how="left") \
               .groupby(["cluster","genre"]) \
               .size() \
               .reset_index(name="count")

vir  = plt.cm.viridis
gray = np.array([0.5,0.5,0.5,1])

for i in range(0, len(cluster_ids), 3):
    row      = cluster_ids[i:i+3]
    pie_cols = st.columns(len(row))
    for j, c in enumerate(row):
        sub    = summary[summary.cluster == c].sort_values("count", ascending=False)
        labels = sub.genre.tolist()
        sizes  = sub["count"].tolist()
        n      = len(sizes)
        base   = np.array(vir(c / (len(cluster_ids) - 1)))
        colors = [tuple(base*(1-t) + gray*t) for t in np.linspace(0,1,n)]

        with pie_cols[j]:
            fig, ax = plt.subplots(figsize=(1.5, 1.5))
            fig.patch.set_facecolor('#121212')
            ax.set_facecolor('#121212')

            ax.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                textprops={'color':'white', 'fontsize':6},
                colors=colors
            )
            ax.axis('equal')
            # Feste Plot-Position, verhindert Einziehen bei vielen Labels
            ax.set_position([0.05, 0.05, 0.9, 0.9])

            ax.set_title(f"Cluster {c}", color='white', fontsize=8)
            # Rahmen entfernen
            for spine in ax.spines.values():
                spine.set_visible(False)

            st.pyplot(fig)
