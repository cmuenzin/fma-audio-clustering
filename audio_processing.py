import glob
import random
import librosa
import numpy as np

def get_audio_files(base_path, limit=50):
    files = glob.glob(f"{base_path}/**/*.mp3", recursive=True)
    if len(files) > limit:
        return random.sample(files, limit)
    return files

def extract_features(audio_path, sr=22050, duration=20, offset=10):
    y, sr = librosa.load(audio_path, sr=sr, duration=duration, offset=offset)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_centroid_mean = np.mean(spec_centroid)
    feature_vector = np.concatenate((mfcc_mean, chroma_mean, [zcr_mean, spec_centroid_mean]))
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm
    return feature_vector

def create_feature_matrix(audio_files, duration=20, offset=10):
    features = []
    for path in audio_files:
        feat = extract_features(path, duration=duration, offset=offset)
        features.append(feat)
    return np.vstack(features)