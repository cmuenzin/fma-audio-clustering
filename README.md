# fma-audio-clustering  
_Clustering von Audiodaten im Free Music Archive (FMA)_

---

## Projektübersicht  
Dieses Projekt beschäftigt sich mit der automatischen Cluster-Analyse von Audiodaten aus dem Free Music Archive (FMA). Ziel ist es, Musiktitel anhand ihrer akustischen Merkmale (z. B. Spektral- und Rhythmus-Features) zu gruppieren und so versteckte Strukturen und Genre-Ähnlichkeiten zu entdecken.

---

## Inhalte  
- **data/**  
  - **raw/** – Original-Audiodateien (MP3)  
  - **features/** – Vorverarbeitete Feature-Dateien (CSV mit MFCC, Chroma)  
- **notebooks/**  
  Jupyter-Notebooks zur Exploration, Feature-Extraktion und Clustering  
- **src/**  
  - `extract_features.py` – Skript zur Extraktion akustischer Features  
  - `cluster.py` – Skript zum Durchführen und Visualisieren von Clustering-Algorithmen  
  - `utils.py` – Hilfsfunktionen (Daten-Loading, Preprocessing)  
- **results/**  
  Visualisierungen und Clustering-Ergebnisse (z. B. t-SNE Plots, Cluster-Statistiken)  
- **README.md**  
  Projektbeschreibung und Anleitung (dieses Dokument)  

---

## Voraussetzungen  
- Python 3.8+  
- Bibliotheken (installierbar via `pip install -r requirements.txt`):  
  - `numpy`, `pandas`  
  - `librosa` (Feature-Extraktion)  
  - `scikit-learn` (Clustering-Algorithmen)  
  - `matplotlib`, `seaborn` (Visualisierung)  
  - `jupyter` (Notebooks)  

---

## Installation  
1. Repository klonen  
   ```bash
   git clone https://github.com/dein-user/fma-audio-clustering.git
   cd fma-audio-clustering
