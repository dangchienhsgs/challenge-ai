import os
import pandas as pd
import scipy
import librosa
from tqdm import tqdm
import sys


class ExtractFeature:

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.sample_rate = 32000
        self.data = {}

    def process(self):

        files = os.listdir(self.base_dir)

        data = {'file_name': []}
        for file in tqdm(files):
            full_path = f'{self.base_dir}/{file}'
            y, sr = librosa.load(full_path, sr=self.sample_rate)

            maps = [
                self.extract_signal_mean(y),
                self.extract_zero_crossing_rate(y),
                self.extract_rmse(y),
                self.extract_temp(y, self.sample_rate),
                self.extract_spectral_bandwidth(y, self.sample_rate, 2),
                self.extract_spectral_bandwidth(y, self.sample_rate, 3),
                self.extract_spectral_bandwidth(y, self.sample_rate, 4),
                self.extract_spectral_bandwidth(y, self.sample_rate, 5),
                self.extract_spectral_centroids(y, self.sample_rate),
                self.extract_spectral_roll_off(y, self.sample_rate),
                self.extract_spectral_contrast(y, self.sample_rate),
                self.extract_mfccs(y, self.sample_rate),
                self.extract_chroma(y, self.sample_rate),
                self.extract_tonetz(y, self.sample_rate)

            ]

            for m in maps:
                for feature in m:
                    if feature not in data:
                        data[feature] = []
                    data[feature].append(m[feature])

            data['file_name'].append(file)

        return pd.DataFrame(data)

    @staticmethod
    def extract_signal_mean(y):
        a = pd.Series(y).describe()

        return {
            'signal_mean': a['mean'],
            'signal_std': a['std'],
            'signal_25': a['25%'],
            'signal_50': a['50%'],
            'signal_75': a['75%'],
            'signal_skew': scipy.stats.skew(abs(y)),
            'signal_kurtoris': scipy.stats.kurtosis(y)
        }

    @staticmethod
    def extract_zero_crossing_rate(y):
        zcr = librosa.feature.zero_crossing_rate(y + 0.0001, frame_length=2048, hop_length=512)[0]
        a = pd.Series(zcr).describe()

        return {
            'zcr_mean': a['mean'],
            'zcr_std': a['std'],
            'zcr_25': a['25%'],
            'zcr_50': a['50%'],
            'zcr_75': a['75%'],
            'zcr_skew': scipy.stats.skew(abs(zcr)),
            'zcr_kurtoris': scipy.stats.kurtosis(zcr)
        }

    @staticmethod
    def extract_rmse(y):
        rmse = librosa.feature.rmse(y + 0.0001)[0]
        a = pd.Series(rmse).describe()

        return {
            'rmse_mean': a['mean'],
            'rmse_std': a['std'],
            'rmse_25': a['25%'],
            'rmse_50': a['50%'],
            'rmse_75': a['75%'],
            'rmse_skew': scipy.stats.skew(abs(rmse)),
            'rmse_kurtoris': scipy.stats.kurtosis(rmse)
        }

    @staticmethod
    def extract_temp(y, sr):
        tempo = librosa.beat.tempo(y, sr=sr)
        return {
            'tempo': tempo
        }

    @staticmethod
    def extract_spectral_centroids(y, sr):
        spectral_centroids = librosa.feature.spectral_centroid(y + 0.01, sr=sr)[0]
        a = pd.Series(spectral_centroids).describe()

        return {
            'spec_centroid_mean': a['mean'],
            'spec_centroid_std': a['std'],
            'spec_centroid_25': a['25%'],
            'spec_centroid_50': a['50%'],
            'spec_centroid_75': a['75%'],
            'spec_centroid_skew': scipy.stats.skew(abs(spectral_centroids)),
            'spec_centroid_kurtoris': scipy.stats.kurtosis(spectral_centroids)
        }

    @staticmethod
    def extract_spectral_bandwidth(y, sr, p):
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y + 0.01, sr=sr, p=p)[0]
        a = pd.Series(spectral_bandwidth).describe()

        return {
            'spec_centroid_bw_{0}_mean'.format(p): a['mean'],
            'spec_centroid_bw_{0}_std'.format(p): a['std'],
            'spec_centroid_bw_{0}_25'.format(p): a['25%'],
            'spec_centroid_bw_{0}_50'.format(p): a['50%'],
            'spec_centroid_bw_{0}_75'.format(p): a['75%'],
            'spec_centroid_bw_{0}_skew'.format(p): scipy.stats.skew(abs(spectral_bandwidth)),
            'spec_centroid_bw_{0}_kurtoris'.format(p): scipy.stats.kurtosis(spectral_bandwidth)
        }

    @staticmethod
    def extract_spectral_contrast(y, sr):
        spectral_contrast = librosa.feature.spectral_contrast(y, sr=sr, n_bands=6, fmin=200.0)

        m = {}
        for i in range(0, 7):
            a = pd.Series(spectral_contrast[i]).describe()

            m[f'spectral_contrast_{i}_mean'] = a['mean']
            m[f'spectral_contrast_{i}_std'] = a['std']
            m[f'spectral_contrast_{i}_25'] = a['25%']
            m[f'spectral_contrast_{i}_50'] = a['50%']
            m[f'spectral_contrast_{i}_75'] = a['75%']
            m[f'spectral_contrast_{i}_skew'] = scipy.stats.skew(abs(spectral_contrast[i]))
            m[f'spectral_contrast_{i}_kurtoris'] = scipy.stats.kurtosis(spectral_contrast[i])

        return m

    @staticmethod
    def extract_spectral_roll_off(y, sr):
        spectral_rolloff = librosa.feature.spectral_rolloff(y + 0.01, sr=sr, roll_percent=0.85)[0]
        a = pd.Series(spectral_rolloff).describe()

        return {
            'spectral_roll_off_mean': a['mean'],
            'spectral_roll_off_std': a['std'],
            'spectral_roll_off_25': a['25%'],
            'spectral_roll_off_50': a['50%'],
            'spectral_roll_off_75': a['75%'],
            'spectral_roll_off_skew': scipy.stats.skew(abs(spectral_rolloff)),
            'spectral_roll_off_kurtoris': scipy.stats.kurtosis(spectral_rolloff)
        }

    @staticmethod
    def extract_mfccs(y, sr):
        mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=20)

        m = {}

        for i in range(0, 20):
            a = pd.Series(mfccs[i]).describe()

            m[f'mfccs_{i}_mean'] = a['mean']
            m[f'mfccs_{i}_std'] = a['std']
            m[f'mfccs_{i}_25'] = a['25%']
            m[f'mfccs_{i}_50'] = a['50%']
            m[f'mfccs_{i}_75'] = a['75%']
            m[f'mfccs_{i}_skew'] = scipy.stats.skew(abs(mfccs[i]))
            m[f'mfccs_{i}_kurtoris'] = scipy.stats.kurtosis(mfccs[i])

        return m

    @staticmethod
    def extract_chroma(y, sr):
        chroma_stft = librosa.feature.chroma_stft(y, sr=sr, hop_length=1024)

        m = {}

        for i in range(0, 12):
            a = pd.Series(chroma_stft[i]).describe()
            m[f'chroma_{i}_mean'] =  a['mean']
            m[f'chroma_{i}_std'] = a['std']
            m[f'chroma_{i}_25'] = a['25%']
            m[f'chroma_{i}_50'] = a['50%']
            m[f'chroma_{i}_75'] = a['75%']
            m[f'chroma_{i}_skew'] = scipy.stats.skew(abs(chroma_stft[i])),
            m[f'chroma_{i}_kurtoris'] = scipy.stats.kurtosis(chroma_stft[i])

        return m

    @staticmethod
    def extract_tonetz(y, sr):
        tonez = librosa.feature.tonnetz(y, sr)

        m = {}
        for i in range(0, 5):
            a = pd.Series(tonez[i]).describe()
            m[f'tonetz_dim_{i}_mean'] = a['mean']
            m[f'tonetz_dim_{i}_std'] = a['std']
            m[f'tonetz_dim_{i}_25'] = a['25%']
            m[f'tonetz_dim_{i}_50'] = a['50%']
            m[f'tonetz_dim_{i}_75'] = a['75%']
            m[f'tonetz_dim_{i}_skew'] = scipy.stats.skew(abs(tonez[i]))
            m[f'tonetz_dim_{i}_kurtoris'] = scipy.stats.kurtosis(tonez[i])

        return m


if __name__ == "__main__":
    base_dir = sys.argv[1]
    label = sys.argv[2]
    df = ExtractFeature(base_dir).process()

    df.to_csv(f'new_{label}.csv')
