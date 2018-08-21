import numpy as np
import librosa
import math
import re
import os
from multiprocessing import Pool
from keras.utils import to_categorical

gender_dict = {'female': 0, 'male': 1}
region_dict = {'north': 0, 'central': 1, 'south': 2}


def get_file_path(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def getfeature(fname):
    timeseries_length = 128
    hop_length = 512
    data = np.zeros((timeseries_length, 83), dtype=np.float64)

    y, sr = librosa.load(fname)
    S = np.abs(librosa.stft(y))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    tonnets = librosa.feature.tonnetz(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)
    rmse = librosa.feature.rmse(y=y, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y)
    chroma_cens = librosa.feature.chroma_cens(y=y, hop_length=hop_length)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, hop_length=hop_length)
    poly_features = librosa.feature.poly_features(S=S, sr=sr)

    filelength = timeseries_length if mfcc.shape[1] >= timeseries_length else mfcc.shape[1]

    data[-filelength:, 0:13] = mfcc.T[0:timeseries_length, :]
    data[-filelength:, 13:14] = spectral_center.T[0:timeseries_length, :]
    data[-filelength:, 14:26] = chroma.T[0:timeseries_length, :]
    data[-filelength:, 26:33] = spectral_contrast.T[0:timeseries_length, :]
    data[-filelength:, 33:39] = tonnets.T[0:timeseries_length, :]
    data[-filelength:, 39:51] = chroma_stft.T[0:timeseries_length, :]
    data[-filelength:, 51:52] = spectral_rolloff.T[0:timeseries_length, :]
    data[-filelength:, 52:53] = spectral_bandwidth.T[0:timeseries_length, :]
    data[-filelength:, 53:54] = spectral_centroid.T[0:timeseries_length, :]
    data[-filelength:, 54:55] = spectral_flatness.T[0:timeseries_length, :]
    data[-filelength:, 55:56] = rmse.T[0:timeseries_length, :]
    data[-filelength:, 56:68] = chroma_cens.T[0:timeseries_length, :]
    data[-filelength:, 68:80] = chroma_cqt.T[0:timeseries_length, :]
    data[-filelength:, 80:82] = poly_features.T[0:timeseries_length, :]
    data[-filelength:, 82:83] = zcr.T[0:timeseries_length, :]

    distribution_values = extract_distribution_value(y, sr)

    vector = np.reshape(data, (timeseries_length * 83,))

    print(len(distribution_values))
    vector = np.concatenate([vector, distribution_values])
    return vector


def extract_distribution_value(y, sr):
    S = np.abs(librosa.stft(y))
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    rmse = librosa.feature.rmse(y=y)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    poly_features = librosa.feature.poly_features(S=S, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)

    vector = [
        tempo,
        sum(beats),
        np.average(beats),
        np.mean(chroma_stft),
        np.std(chroma_stft),
        np.var(chroma_stft),
        np.mean(chroma_cq),
        np.std(chroma_cq),
        np.var(chroma_cq),
        np.mean(chroma_cens),
        np.std(chroma_cens),
        np.var(chroma_cens),
        np.mean(melspectrogram),
        np.std(melspectrogram),
        np.var(melspectrogram),
        np.mean(mfcc),
        np.std(mfcc),
        np.var(mfcc),
        np.mean(mfcc_delta),
        np.std(mfcc_delta),
        np.var(mfcc_delta),
        np.mean(rmse),  # rmse
        np.std(rmse),
        np.var(rmse),
        np.mean(cent),
        np.std(cent),
        np.var(cent),
        np.mean(spec_bw),
        np.std(spec_bw),
        np.var(spec_bw),
        np.mean(contrast),
        np.std(contrast),
        np.var(contrast),
        np.mean(rolloff),
        np.std(rolloff),
        np.var(rolloff),
        np.mean(poly_features),
        np.std(poly_features),
        np.var(poly_features),
        np.mean(tonnetz),
        np.std(tonnetz),
        np.var(tonnetz),
        np.mean(zcr),
        np.std(zcr),
        np.var(zcr),
        np.mean(harmonic),
        np.std(harmonic),
        np.var(harmonic),
        np.mean(percussive),
        np.std(percussive),
        np.var(percussive),
        np.mean(frames_to_time),
        np.std(frames_to_time),
        np.var(frames_to_time),
    ]

    return vector


def analysis_train_name(fname):
    data = getfeature(fname)
    gender, region = fname.split('/')[-2].split('_')
    print(fname)

    return data, gender, region


def analysis_test_name(fname):
    data = getfeature(fname)
    name = fname.split('/')[-1]
    print(fname)

    return data, name


def create_data_train():
    files = list(get_file_path('data/accent_gender/train/'))
    p = Pool(40)
    data = p.map(analysis_train_name, files)
    X = [data[i][0] for i in range(len(data))]
    X = np.asarray(X)

    gender = [gender_dict[data[i][1]] for i in range(len(data))]
    gender = to_categorical(gender)

    region = [region_dict[data[i][2]] for i in range(len(data))]
    region = to_categorical(region)

    np.savez('data/accent_gender/train', X=X, gender=gender, region=region)


def create_data_test():
    files = list(get_file_path('data/accent_gender/public_test'))
    p = Pool(40)
    data = p.map(analysis_test_name, files)

    X = [data[i][0] for i in range(len(data))]
    X = np.asarray(X)

    name = [data[i][1] for i in range(len(data))]
    np.savez('data/accent_gender/public_test', X=X, name=name)


if __name__ == '__main__':
    # create_data_train()
    # create_data_test()
    print(len(getfeature('../test/test.mp3')))