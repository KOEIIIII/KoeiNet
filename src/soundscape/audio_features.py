


"""Signal and psychoacoustic-proxy feature extraction utilities."""

from __future__ import annotations

import logging
import math
import wave
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("soundscape.audio_features")

try:
    import librosa

    LIBROSA_AVAILABLE = True
except Exception:
    librosa = None
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except Exception:
    sf = None
    SOUNDFILE_AVAILABLE = False

try:
    from scipy.signal import hilbert

    SCIPY_HILBERT_AVAILABLE = True
except Exception:
    hilbert = None
    SCIPY_HILBERT_AVAILABLE = False


def load_audio_waveform(wav_path: str) -> Tuple[Optional[np.ndarray], Optional[int], str]:
    """
    Load mono waveform from local file with graceful fallbacks.

    Returns:
    - waveform float32 in [-1, 1] or None
    - sample_rate or None
    - loader_name
    """
    path = Path(wav_path)
    if not path.exists():
        return None, None, "missing"

    if LIBROSA_AVAILABLE:
        try:
            y, sr = librosa.load(path.as_posix(), sr=None, mono=True)
            return y.astype(np.float32), int(sr), "librosa"
        except Exception as exc:
            logger.warning("librosa 读取失败，回退其他方案: %s", exc)

    if SOUNDFILE_AVAILABLE:
        try:
            y, sr = sf.read(path.as_posix(), dtype="float32", always_2d=False)
            if y is None:
                return None, None, "soundfile_none"
            if isinstance(y, np.ndarray) and y.ndim > 1:
                y = np.mean(y, axis=1)
            return np.asarray(y, dtype=np.float32), int(sr), "soundfile"
        except Exception as exc:
            logger.warning("soundfile 读取失败，回退 wave: %s", exc)

    try:
        with wave.open(path.as_posix(), "rb") as wf:
            sr = int(wf.getframerate())
            n_channels = int(wf.getnchannels())
            sampwidth = int(wf.getsampwidth())
            n_frames = int(wf.getnframes())
            raw = wf.readframes(n_frames)

        if sampwidth == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            logger.warning("wave fallback 不支持采样宽度: %s", sampwidth)
            return None, None, "wave_unsupported_width"

        if n_channels > 1:
            arr = arr.reshape(-1, n_channels).mean(axis=1)
        return arr.astype(np.float32), sr, "wave"
    except Exception as exc:
        logger.error("wave fallback 读取失败: %s", exc)
        return None, None, "failed"


def slice_waveform(waveform: np.ndarray, sample_rate: int, start_sec: float, end_sec: float) -> np.ndarray:
    """Slice waveform by second-range with bound checks."""
    if waveform is None or len(waveform) == 0 or sample_rate <= 0:
        return np.asarray([], dtype=np.float32)
    s = max(0, int(round(float(start_sec) * float(sample_rate))))
    e = max(s, int(round(float(end_sec) * float(sample_rate))))
    e = min(e, len(waveform))
    return waveform[s:e].astype(np.float32)


def _frame_signal(x: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0, frame_length), dtype=np.float32)
    if x.size < frame_length:
        pad = np.zeros(frame_length - x.size, dtype=np.float32)
        return np.expand_dims(np.concatenate([x, pad]), axis=0)

    frames = []
    for i in range(0, x.size - frame_length + 1, hop):
        frames.append(x[i : i + frame_length])
    if not frames:
        return np.zeros((0, frame_length), dtype=np.float32)
    return np.asarray(frames, dtype=np.float32)


def _manual_spectral_features(
    x: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    rolloff_ratio: float,
) -> Dict[str, float]:
    frames = _frame_signal(x, frame_length=n_fft, hop=hop)
    if frames.shape[0] == 0:
        return {
            "spectral_centroid": float("nan"),
            "spectral_bandwidth": float("nan"),
            "spectral_rolloff": float("nan"),
            "spectral_flatness": float("nan"),
            "spectral_flux": float("nan"),
        }

    window = np.hanning(n_fft).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames * window, axis=1)) + 1e-12
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)

    mag_sum = np.sum(mag, axis=1) + 1e-12
    centroid = np.sum(mag * freqs[None, :], axis=1) / mag_sum
    bandwidth = np.sqrt(
        np.sum(((freqs[None, :] - centroid[:, None]) ** 2) * mag, axis=1) / mag_sum
    )

    cumsum = np.cumsum(mag, axis=1)
    threshold = (rolloff_ratio * mag_sum)[:, None]
    rolloff_idx = np.argmax(cumsum >= threshold, axis=1)
    rolloff = freqs[rolloff_idx]

    flatness = np.exp(np.mean(np.log(mag), axis=1)) / (np.mean(mag, axis=1) + 1e-12)

    mag_norm = mag / (np.sum(mag, axis=1, keepdims=True) + 1e-12)
    d = np.diff(mag_norm, axis=0)
    flux = np.sqrt(np.sum(np.maximum(d, 0.0) ** 2, axis=1))

    return {
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
        "spectral_rolloff": float(np.mean(rolloff)),
        "spectral_flatness": float(np.mean(flatness)),
        "spectral_flux": float(np.mean(flux) if flux.size else 0.0),
    }


def compute_segment_audio_features(
    waveform: Optional[np.ndarray],
    sample_rate: Optional[int],
    n_fft: int = 1024,
    hop: int = 512,
    rolloff_ratio: float = 0.85,
) -> Dict[str, float]:
    """
    Compute signal + psychoacoustic-proxy features for one waveform segment.

    Returns NaN-valued metrics when waveform is unavailable.
    """
    base = {
        "rms_energy": float("nan"),
        "zero_crossing_rate": float("nan"),
        "spectral_centroid": float("nan"),
        "spectral_bandwidth": float("nan"),
        "spectral_rolloff": float("nan"),
        "spectral_flatness": float("nan"),
        "spectral_flux": float("nan"),
        "loudness_proxy_db": float("nan"),
        "sharpness_proxy": float("nan"),
        "roughness_proxy": float("nan"),
    }
    if waveform is None or sample_rate is None or sample_rate <= 0 or waveform.size == 0:
        return base

    x = waveform.astype(np.float32)
    rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)).astype(np.float32))) if x.size > 1 else 0.0)

    if LIBROSA_AVAILABLE:
        try:
            centroid = float(
                np.mean(
                    librosa.feature.spectral_centroid(
                        y=x, sr=sample_rate, n_fft=n_fft, hop_length=hop
                    )
                )
            )
            bandwidth = float(
                np.mean(
                    librosa.feature.spectral_bandwidth(
                        y=x, sr=sample_rate, n_fft=n_fft, hop_length=hop
                    )
                )
            )
            rolloff = float(
                np.mean(
                    librosa.feature.spectral_rolloff(
                        y=x,
                        sr=sample_rate,
                        n_fft=n_fft,
                        hop_length=hop,
                        roll_percent=rolloff_ratio,
                    )
                )
            )
            flatness = float(
                np.mean(
                    librosa.feature.spectral_flatness(y=x, n_fft=n_fft, hop_length=hop)
                )
            )
            s = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)) + 1e-12
            s_norm = s / (np.sum(s, axis=0, keepdims=True) + 1e-12)
            d = np.diff(s_norm, axis=1)
            flux = float(np.mean(np.sqrt(np.sum(np.maximum(d, 0.0) ** 2, axis=0)))) if d.size else 0.0
            spec = {
                "spectral_centroid": centroid,
                "spectral_bandwidth": bandwidth,
                "spectral_rolloff": rolloff,
                "spectral_flatness": flatness,
                "spectral_flux": flux,
            }
        except Exception as exc:
            logger.warning("librosa 特征计算失败，回退手工谱特征: %s", exc)
            spec = _manual_spectral_features(x, sample_rate, n_fft=n_fft, hop=hop, rolloff_ratio=rolloff_ratio)
    else:
        spec = _manual_spectral_features(x, sample_rate, n_fft=n_fft, hop=hop, rolloff_ratio=rolloff_ratio)

    loudness_db = float(20.0 * np.log10(rms + 1e-12))
    sharpness = float(spec["spectral_centroid"] / max(sample_rate / 2.0, 1e-6))



    if SCIPY_HILBERT_AVAILABLE:
        env = np.abs(hilbert(x))
    else:
        env = np.abs(x)
    env = env - float(np.mean(env))
    if env.size > 1:
        env_mag = np.abs(np.fft.rfft(env))
        env_freq = np.fft.rfftfreq(env.size, d=1.0 / float(sample_rate))
        band = (env_freq >= 30.0) & (env_freq <= 150.0)
        roughness = float(np.sum(env_mag[band]) / (np.sum(env_mag) + 1e-12))
    else:
        roughness = float("nan")

    return {
        "rms_energy": rms,
        "zero_crossing_rate": zcr,
        "spectral_centroid": float(spec["spectral_centroid"]),
        "spectral_bandwidth": float(spec["spectral_bandwidth"]),
        "spectral_rolloff": float(spec["spectral_rolloff"]),
        "spectral_flatness": float(spec["spectral_flatness"]),
        "spectral_flux": float(spec["spectral_flux"]),
        "loudness_proxy_db": loudness_db,
        "sharpness_proxy": sharpness,
        "roughness_proxy": roughness,
    }

