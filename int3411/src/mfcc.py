from scipy import signal
from scipy import fftpack
import numpy as np


class MFCC:
    def __init__(
        self,
        frame_size: int = 25,
        hop_size: int = 10,
        num_coeffs: int = 13,
        num_filters: int = 26,
        num_mel_bins: int = 40,
        pre_emphasis_coeff: float = 0.97,
        mode: str = "psd",
        window: str = 'hamming',
    ):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.num_coeffs = num_coeffs
        self.num_filters = num_filters
        self.num_mel_bins = num_mel_bins
        self.pre_emphasis_coeff = pre_emphasis_coeff
        self.mode = mode
        self.window = window

    def __call__(self, audio, sr):
        # pre-emphasis
        audio = np.append(audio[0], audio[1:] - self.pre_emphasis_coeff * audio[:-1])
        # stft
        window = signal.get_window(self.window, self.frame_size * sr // 1000)
        f, t, spectrogram = signal.spectrogram(audio,
                                               fs=sr,
                                               window=window,
                                               noverlap=self.hop_size * sr // 1000,
                                               mode=self.mode)
        # mel filter bank
        filterbank = self.mel_filterbank(sr, spectrogram.shape[0], 0, sr / 2)
        # mel spectrogram
        mel_spec = np.dot(filterbank, spectrogram)
        # log mel spectrogram
        log_mel_spec = np.log10(1 + 1000 * mel_spec)
        # dct
        dct_coeffs = fftpack.dct(log_mel_spec, axis=0, type=2, norm='ortho')
        # mfcc
        delta_mfcc = self.get_delta(dct_coeffs)
        delta_delta_mfcc = self.get_delta(delta_mfcc)

        mfcc = dct_coeffs[:self.num_coeffs]
        delta_mfcc = delta_mfcc[:self.num_coeffs]
        delta_delta_mfcc = delta_delta_mfcc[:self.num_coeffs]

        return np.vstack((mfcc, delta_mfcc, delta_delta_mfcc))

    def pre_emphasis(self, signal):
        emphasized_signal = np.append(signal[0], signal[1:] - self.pre_emphasis_coeff * signal[:-1])
        return emphasized_signal

    def mel_filterbank(self, sr, num_freq, min_freq, max_freq):
        num_fft = (num_freq - 1) * 2
        mel_min = self.hz_to_mel(min_freq)
        mel_max = self.hz_to_mel(max_freq)
        banks = np.linspace(mel_min, mel_max, self.num_mel_bins + 2)
        freqs = self.mel_to_hz(banks)
        bins = np.floor((num_fft + 1) * freqs / sr)
        fbank = np.zeros((self.num_mel_bins, num_fft // 2 + 1))
        for i in range(self.num_mel_bins):
            prev_bin, cur_bin, next_bin = bins[i:i + 3].astype(int)
            for j in range(prev_bin, next_bin):
                fbank[i, j] = (j - prev_bin) / (next_bin - prev_bin)
            fbank[i, cur_bin] = 1
        return fbank

    def get_delta(self, features):
        delta_features = np.zeros(shape=features.shape)
        for i in range(1, features.shape[1]):
            delta_features[:, i] = features[:, i] - features[:, i - 1]
        return delta_features

    @staticmethod
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)
    
    @staticmethod
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
        