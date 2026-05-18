


"""
音频增强特征提取模块
"""

import os
import numpy as np
import librosa
import scipy.signal
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AudioEnhancedFeatures:
    """
    音频增强特征提取器

    提供信噪比、频谱特征、情感声学特征和空间声学特征
    """

    def __init__(self, sample_rate: int = 16000, window_size: int = 2048, hop_length: int = 512):
        """
        初始化音频增强特征提取器

        Args:
            sample_rate: 采样率
            window_size: 窗口大小
            hop_length: 跳跃长度
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length

    def calculate_snr(self, audio: np.ndarray, noise_duration: float = 0.5) -> float:
        """
        计算信噪比 (Signal-to-Noise Ratio)

        Args:
            audio: 音频信号
            noise_duration: 噪声估计持续时间（秒）

        Returns:
            snr: 信噪比 (dB)
        """
        try:

            noise_samples = int(noise_duration * self.sample_rate)
            noise_samples = min(noise_samples, len(audio) // 4)

            if noise_samples < 1000:
                noise_power = np.var(audio[:1000]) if len(audio) > 1000 else np.var(audio)
            else:
                noise_power = np.var(audio[:noise_samples])


            signal_power = np.var(audio)


            if noise_power == 0:
                return 60.0


            snr = 10 * np.log10(signal_power / noise_power)


            snr = max(-20, min(60, snr))

            return float(snr)

        except Exception as e:
            logger.warning(f"计算SNR时出错: {e}")
            return 20.0

    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        提取频谱特征

        Args:
            audio: 音频信号

        Returns:
            spectral_features: 频谱特征字典
        """
        try:

            stft = librosa.stft(audio, n_fft=self.window_size, hop_length=self.hop_length)
            magnitude = np.abs(stft)


            spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=self.sample_rate)[0]
            spectral_centroid_mean = np.mean(spectral_centroids)


            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sample_rate)[0]
            spectral_bandwidth_mean = np.mean(spectral_bandwidth)


            spectral_contrast = librosa.feature.spectral_contrast(S=magnitude, sr=self.sample_rate)
            spectral_contrast_mean = np.mean(spectral_contrast)


            spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=self.sample_rate)[0]
            spectral_rolloff_mean = np.mean(spectral_rolloff)


            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_mean = np.mean(zcr)

            return {
                'spectral_centroid': float(spectral_centroid_mean),
                'spectral_bandwidth': float(spectral_bandwidth_mean),
                'spectral_contrast': float(spectral_contrast_mean),
                'spectral_rolloff': float(spectral_rolloff_mean),
                'zero_crossing_rate': float(zcr_mean)
            }

        except Exception as e:
            logger.warning(f"提取频谱特征时出错: {e}")
            return {
                'spectral_centroid': 0.0,
                'spectral_bandwidth': 0.0,
                'spectral_contrast': 0.0,
                'spectral_rolloff': 0.0,
                'zero_crossing_rate': 0.0
            }

    def extract_mfcc_features(self, audio: np.ndarray, n_mfcc: int = 13) -> Dict[str, float]:
        """
        提取MFCC特征

        Args:
            audio: 音频信号
            n_mfcc: MFCC系数数量

        Returns:
            mfcc_features: MFCC特征字典
        """
        try:

            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)


            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)

            features = {}
            for i in range(n_mfcc):
                features[f'mfcc_{i+1}_mean'] = float(mfcc_mean[i])
                features[f'mfcc_{i+1}_std'] = float(mfcc_std[i])

            return features

        except Exception as e:
            logger.warning(f"提取MFCC特征时出错: {e}")
            return {f'mfcc_{i+1}_mean': 0.0 for i in range(n_mfcc)} | \
                   {f'mfcc_{i+1}_std': 0.0 for i in range(n_mfcc)}

    def extract_emotion_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        提取音频情感特征

        Args:
            audio: 音频信号

        Returns:
            emotion_features: 情感特征字典
        """
        try:

            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0


            rms = librosa.feature.rms(y=audio)[0]
            energy_mean = np.mean(rms)
            energy_std = np.std(rms)


            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)


            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            chroma_mean = np.mean(chroma)
            chroma_std = np.std(chroma)

            return {
                'pitch_mean': float(pitch_mean),
                'energy_mean': float(energy_mean),
                'energy_std': float(energy_std),
                'tempo': float(tempo),
                'chroma_mean': float(chroma_mean),
                'chroma_std': float(chroma_std)
            }

        except Exception as e:
            logger.warning(f"提取情感特征时出错: {e}")
            return {
                'pitch_mean': 0.0,
                'energy_mean': 0.0,
                'energy_std': 0.0,
                'tempo': 0.0,
                'chroma_mean': 0.0,
                'chroma_std': 0.0
            }

    def extract_spatial_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        提取空间声学特征

        Args:
            audio: 音频信号

        Returns:
            spatial_features: 空间特征字典
        """
        try:

            if len(audio.shape) > 1 and audio.shape[1] > 1:

                left = audio[:, 0]
                right = audio[:, 1]


                correlation = np.corrcoef(left, right)[0, 1]


                mid = (left + right) / 2
                side = (left - right) / 2
                stereo_width = np.std(side) / (np.std(mid) + 1e-8)

                return {
                    'stereo_correlation': float(correlation),
                    'stereo_width': float(stereo_width),
                    'channel_balance': float(np.mean(np.abs(left)) / (np.mean(np.abs(right)) + 1e-8))
                }
            else:

                return {
                    'stereo_correlation': 1.0,
                    'stereo_width': 0.0,
                    'channel_balance': 1.0
                }

        except Exception as e:
            logger.warning(f"提取空间特征时出错: {e}")
            return {
                'stereo_correlation': 1.0,
                'stereo_width': 0.0,
                'channel_balance': 1.0
            }

    def extract_all_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        提取所有增强特征

        Args:
            audio: 音频信号

        Returns:
            all_features: 所有特征字典
        """
        features = {}


        features['snr'] = self.calculate_snr(audio)


        spectral_features = self.extract_spectral_features(audio)
        features.update(spectral_features)


        mfcc_features = self.extract_mfcc_features(audio)
        features.update(mfcc_features)


        emotion_features = self.extract_emotion_features(audio)
        features.update(emotion_features)


        spatial_features = self.extract_spatial_features(audio)
        features.update(spatial_features)

        return features
