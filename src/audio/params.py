














"""Hyperparameters for YAMNet."""

from dataclasses import dataclass





@dataclass(frozen=True)
class Params:
  sample_rate: float = 16000.0
  stft_window_seconds: float = 0.025
  stft_hop_seconds: float = 0.010
  mel_bands: int = 64
  mel_min_hz: float = 125.0
  mel_max_hz: float = 7500.0
  log_offset: float = 0.001
  patch_window_seconds: float = 0.96
  patch_hop_seconds: float = 0.48

  @property
  def patch_frames(self):
    return int(round(self.patch_window_seconds / self.stft_hop_seconds))

  @property
  def patch_bands(self):
    return self.mel_bands

  num_classes: int = 521
  conv_padding: str = 'same'
  batchnorm_center: bool = True
  batchnorm_scale: bool = False
  batchnorm_epsilon: float = 1e-4
  classifier_activation: str = 'sigmoid'

  tflite_compatible: bool = False
