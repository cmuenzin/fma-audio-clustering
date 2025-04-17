import glob
import random
import librosa
import numpy as np

def get_audio_files(base_path, limit=50):
    """
    Sammelt rekursiv MP3-Dateien und gibt eine zufällige Auswahl der gewünschten Größe zurück.
    """
    files = glob.glob(f"{base_path}/**/*.mp3", recursive=True)
    if len(files) > limit:
        return random.sample(files, limit)
    return files

def extract_features(audio_path, sr=22050, duration=20, offset=10):
    """
    Lädt einen definierten Ausschnitt der Audiodatei und extrahiert:
      - 13 MFCCs
      - 12 Chroma-Features
      - Zero Crossing Rate
      - Spectral Centroid
    Anschließend L2-normalisieren.
    """
    y, sr = librosa.load(audio_path, sr=sr, duration=duration, offset=offset)
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    # Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_centroid_mean = np.mean(spec_centroid)

    feature_vector = np.concatenate((mfcc_mean, chroma_mean, [zcr_mean, spec_centroid_mean]))
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm
    return feature_vector

def create_feature_matrix(audio_files, duration=20, offset=10):
    """
    Baut Feature-Matrix und aussagekräftige Dateiliste auf;
    überspringt Dateien bei Lade-Fehlern.
    """
    features = []
    valid_files = []
    for path in audio_files:
        try:
            feat = extract_features(path, duration=duration, offset=offset)
            features.append(feat)
            valid_files.append(path)
        except Exception:
            # Überspringe fehlerhafte Dateien
            continue
    if features:
        return np.vstack(features), valid_files
    else:
        return np.empty((0, 27)), []
