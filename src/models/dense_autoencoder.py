"""DCASE Task 2 baseline: dense autoencoder on log-mel spectrogram context windows.

Unlike ``TransferCnn14``/``anomaly.py``'s frozen-embedding density models
(AudioSet classification features, proven near-chance for cross-machine fan
faults — see ``diagnose.py`` AUC 0.5682 and ``anomaly.py`` mean AUC 0.60),
this learns spectral normality **from scratch, per machine, directly on log-
mel spectrograms** — no pretrained backbone involved. This is the standard
DCASE 2020/2022 Task 2 baseline recipe (published results: AUC 0.80-0.95 on
MIMII-family fan data), included here because the frozen-embedding approach
is a known mismatch: AudioSet tagging embeddings discard exactly the fine
spectral texture (bearing wear harmonics, imbalance) that separates healthy
from faulty industrial sound.

Recipe:
1. Log-mel spectrogram per clip (``LOGMEL_*`` constants match the DCASE
   baseline: 16 kHz, 1024-pt FFT, 512 hop, 128 mel bins).
2. Context vectors: ``CONTEXT_FRAMES`` (5) consecutive mel frames
   concatenated into one ``640``-d vector (a small temporal receptive field
   around each frame), stride 1 -> ``(n_frames - 4)`` vectors per clip.
3. ``DenseAutoencoder``: 640 -> 128 -> 128 -> 128 -> 128 -> 8 (bottleneck)
   -> 128 -> 128 -> 128 -> 128 -> 640, BatchNorm+ReLU on every hidden layer,
   trained with MSE reconstruction loss on ONE machine's normal-only context
   vectors pooled across all its normal clips.
4. Anomaly score for a clip = mean per-vector reconstruction MSE across all
   of its context vectors. Normal audio reconstructs well (low error);
   unfamiliar (faulty) spectral patterns don't.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

# Canonical DCASE 2020/2022 Task 2 baseline front-end config.
LOGMEL_SAMPLE_RATE = 16000
LOGMEL_N_FFT = 1024
LOGMEL_HOP = 512
LOGMEL_N_MELS = 128
LOGMEL_FMIN = 0
LOGMEL_FMAX = 8000

CONTEXT_FRAMES = 5
INPUT_DIM = CONTEXT_FRAMES * LOGMEL_N_MELS  # 640


class LogMelExtractor(nn.Module):
    """Waveform (batch, samples) -> log-mel spectrogram (batch, time, mel_bins)."""

    def __init__(self) -> None:
        super().__init__()
        self.spectrogram = Spectrogram(
            n_fft=LOGMEL_N_FFT,
            hop_length=LOGMEL_HOP,
            win_length=LOGMEL_N_FFT,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel = LogmelFilterBank(
            sr=LOGMEL_SAMPLE_RATE,
            n_fft=LOGMEL_N_FFT,
            n_mels=LOGMEL_N_MELS,
            fmin=LOGMEL_FMIN,
            fmax=LOGMEL_FMAX,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.spectrogram(waveform)  # (batch, 1, time, freq)
        x = self.logmel(x)  # (batch, 1, time, mel_bins)
        return x.squeeze(1)  # (batch, time, mel_bins)


def logmel_to_context_vectors(logmel: torch.Tensor) -> torch.Tensor:
    """One clip's log-mel (time, mel_bins) -> context vectors (time-4, 640).

    Each output row is ``CONTEXT_FRAMES`` consecutive mel frames flattened —
    the DCASE baseline's sliding local-context trick. Returns an empty
    ``(0, INPUT_DIM)`` tensor if the clip is too short for even one window.
    """
    t = logmel.shape[0]
    n_windows = t - CONTEXT_FRAMES + 1
    if n_windows <= 0:
        return torch.empty(0, INPUT_DIM)
    windows = logmel.unfold(0, CONTEXT_FRAMES, 1)  # (n_windows, mel_bins, CONTEXT_FRAMES)
    return windows.permute(0, 2, 1).reshape(n_windows, INPUT_DIM)


class DenseAutoencoder(nn.Module):
    """640 -> 128x4 -> 8 -> 128x4 -> 640 dense autoencoder (DCASE baseline)."""

    def __init__(self, input_dim: int = INPUT_DIM, bottleneck: int = 8, hidden: int = 128) -> None:
        super().__init__()

        def block(in_f: int, out_f: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.BatchNorm1d(out_f),
                nn.ReLU(inplace=True),
            )

        self.encoder = nn.Sequential(
            block(input_dim, hidden),
            block(hidden, hidden),
            block(hidden, hidden),
            block(hidden, hidden),
            nn.Linear(hidden, bottleneck),
        )
        self.decoder = nn.Sequential(
            block(bottleneck, hidden),
            block(hidden, hidden),
            block(hidden, hidden),
            block(hidden, hidden),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
