"""Convolutional autoencoder on whole-clip log-mel spectrograms.

The dense autoencoder (``dense_autoencoder.py``) flattens 5-frame windows
into plain 640-d vectors — that discards the spectrogram's 2D time-frequency
structure before the network ever sees it. Four different levers on top of
that representation (more training, more capacity, different score
aggregation) all converged near the same ~0.60-0.66 mean AUC ceiling, which
points at the flattening step itself as the limiting factor, not any of the
things tuned around it.

This operates on each clip's **whole log-mel spectrogram as a 2D image**
(batch, 1, time, mel_bins) through a small strided-conv encoder/decoder,
preserving local time-frequency structure (harmonic streaks, transient
bursts) that a flattened dense vector cannot represent. Same anomaly-score
principle as the dense model: reconstruction error on unfamiliar (faulty)
spectral patterns is higher than on the machine's own learned-normal ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    """Strided-conv encoder/decoder over a (1, T, mel_bins) log-mel spectrogram.

    Encoder halves both spatial dims four times (16x total downsampling);
    decoder mirrors it with nearest-neighbor upsampling + conv (avoids the
    checkerboard artifacts of transposed conv) and a final interpolate to
    exactly match the input's (T, mel_bins) — needed because T isn't
    guaranteed divisible by 16.
    """

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels

        def down(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        def up(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.encoder = nn.Sequential(
            down(1, c),
            down(c, c * 2),
            down(c * 2, c * 4),
            down(c * 4, c * 4),
        )
        self.decoder = nn.Sequential(
            up(c * 4, c * 4),
            up(c * 4, c * 2),
            up(c * 2, c),
            up(c, c),
        )
        self.out_conv = nn.Conv2d(c, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (batch, 1, T, mel_bins) -> reconstruction of the same shape."""
        target_shape = x.shape[-2:]
        z = self.encoder(x)
        recon = self.decoder(z)
        recon = self.out_conv(recon)
        if recon.shape[-2:] != target_shape:
            recon = F.interpolate(recon, size=target_shape, mode="bilinear", align_corners=False)
        return recon
