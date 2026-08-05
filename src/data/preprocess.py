"""Audio preprocessing pipeline (TDD sec 3.2).

Every raw clip is converted to the model-ready representation:
- Resample to 32 kHz mono (PANNs' expected input rate).
- Pad with zeros / trim to a fixed clip length (10 s, matching MIMII).
- Return a float32 waveform tensor — PANNs' ``Cnn14`` computes its own
  log-mel spectrogram internally, so no spectrogram is precomputed here.

Also hosts the training-only augmentation transforms (TDD sec 3.5):
- Additive Gaussian noise (p=0.3)
- Random gain scaling (p=0.3)
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


def _resample_numpy(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample a 1-D waveform without librosa (fallback path only)."""
    if src_sr == dst_sr:
        return data
    import torchaudio.functional as ta_F

    tensor = torch.from_numpy(np.asarray(data, dtype=np.float32)).unsqueeze(0)
    return ta_F.resample(tensor, src_sr, dst_sr).squeeze(0).numpy()


def load_fixed_length_audio(path: str | Path, sr: int, target_len: int) -> torch.Tensor:
    """Load one file to a mono float32 waveform of exactly ``target_len`` samples.

    - Primary path: ``librosa.load(path, sr, mono=True)`` — resamples and
      downmixes to mono in one call.
    - Fallback path: ``soundfile`` reads the native-rate signal, manual
      mean-downmix to mono, then ``torchaudio`` resample if ``librosa`` fails.
    - Padding: a short clip is right-padded with zeros. Trimming: a long clip
      keeps only its first ``target_len`` samples.

    Returns a tensor of shape ``(target_len,)``.
    """
    path = str(path)
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as librosa_error:
        # Fallback: soundfile (native sr) + manual mono downmix + resample.
        try:
            data, native_sr = sf.read(path, dtype="float32", always_2d=False, fill_value=0.0)
        except Exception as exc:
            raise ValueError(f"Could not read audio file with librosa or soundfile: {path}") from exc
        if data.ndim > 1:
            data = data.mean(axis=1)
        y = _resample_numpy(data, int(native_sr), sr)

    waveform = torch.as_tensor(np.asarray(y, dtype=np.float32))

    if waveform.numel() > target_len:
        waveform = waveform[:target_len]
    elif waveform.numel() < target_len:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.numel()))

    if waveform.numel() != target_len:
        raise RuntimeError(f"Unexpected length {waveform.numel()} != {target_len} for {path}")

    return waveform


class Augment(torch.nn.Module):
    """Training-only light augmentation (TDD sec 3.5).

    Additive Gaussian noise and random gain scaling, each applied with a
    configured probability. Deliberately does NOT distort the acoustic fault
    signature (no pitch shifting etc.).
    """

    def __init__(self, noise_p: float = 0.3, gain_p: float = 0.3, noise_std: float = 0.005) -> None:
        super().__init__()
        self.noise_p = noise_p
        self.gain_p = gain_p
        self.noise_std = noise_std

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply each augmentation with probability p; returns same shape."""
        if torch.rand(1).item() < self.noise_p:
            waveform = waveform + torch.randn_like(waveform) * self.noise_std
        if torch.rand(1).item() < self.gain_p:
            waveform = waveform * (0.8 + 0.4 * torch.rand(1).item())
        return waveform