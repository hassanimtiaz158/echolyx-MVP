"""TransferCnn14: PANNs Cnn14 backbone + binary fan head (TDD sec 4).

Transfer-learning wrapper per the TDD:
- Builds the vendored ``Cnn14`` (src/models/cnn14.py) with the canonical
  AudioSet config (32 kHz / 1024 FFT / 320 hop / 64 mel / 50-14000 Hz).
- Loads the AudioSet-pretrained backbone weights (``Cnn14_mAP=0.431.pth``,
  classes_num=527) from a checkpoint path. Handles upstream checkpoint
  formats: a raw state_dict, a ``{"model": ...}`` dict, or ``module.``
  prefixed keys (DataParallel).
- Swaps the original ``fc_audioset`` head for ``nn.Linear(2048, num_classes)``
  -> raw logits over {Normal, Faulty} (TDD 4.2).
- ``freeze_base=True`` at init freezes every backbone parameter so only the
  head trains (Phase 1, TDD 4.3).
- ``unfreeze_last_blocks()`` unfreezes ``conv_block6`` + ``fc1`` only, for
  Phase 2 differential-LR fine-tuning (TDD 4.3).

Download the pretrained checkpoint with
``python -m src.models.download_checkpoint`` (Zenodo record 3987831).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.models.cnn14 import Cnn14, init_bn, init_layer

logger = logging.getLogger(__name__)

# Canonical PANNs CNN14 frontend config (matches the AudioSet training run).
CNN14_SAMPLE_RATE = 32000
CNN14_WINDOW_SIZE = 1024
CNN14_HOP_SIZE = 320
CNN14_MEL_BINS = 64
CNN14_FMIN = 50
CNN14_FMAX = 14000
AUDIOSET_CLASSES = 527


def _strip_module_prefix(state: dict) -> dict:
    """Remove a leading ``module.`` (DataParallel) prefix if present."""
    keys = list(state.keys())
    if keys and keys[0].startswith("module."):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


class TransferCnn14(nn.Module):
    """PANNs CNN14 backbone + binary fan classifier with phase-aware freezing."""

    def __init__(
        self,
        backbone_checkpoint: str | Path,
        num_classes: int = 2,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Build the full AudioSet-capable backbone and load pretrained weights.
        self.cnn14 = Cnn14(
            sample_rate=CNN14_SAMPLE_RATE,
            window_size=CNN14_WINDOW_SIZE,
            hop_size=CNN14_HOP_SIZE,
            mel_bins=CNN14_MEL_BINS,
            fmin=CNN14_FMIN,
            fmax=CNN14_FMAX,
            classes_num=AUDIOSET_CLASSES,
        )
        self._load_backbone(backbone_checkpoint)

        # Replace the AudioSet head with the binary fan head (TDD 4.2).
        self.cnn14.fc_audioset = nn.Linear(2048, num_classes, bias=True)
        init_layer(self.cnn14.fc_audioset)

        if freeze_base:
            self._set_base_trainable(False)

    # ------------------------------------------------------------------ #
    # Weight loading
    # ------------------------------------------------------------------ #
    def _load_backbone(self, backbone_checkpoint: str | Path) -> None:
        path = Path(backbone_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(
                f"Backbone checkpoint not found: {path}. "
                "Download it with `python -m src.models.download_checkpoint`."
            )
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        state = _strip_module_prefix(state)

        missing, unexpected = self.cnn14.load_state_dict(state, strict=False)
        if unexpected:
            logger.warning("Unexpected checkpoint keys (ignored): %s", unexpected[:5])
        if missing and missing != ["fc_audioset.weight", "fc_audioset.bias"]:
            logger.warning("Missing checkpoint keys: %s", missing)
        logger.info("Loaded pretrained backbone from %s", path)

    # ------------------------------------------------------------------ #
    # Phase-aware freezing (TDD 4.3)
    # ------------------------------------------------------------------ #
    def _set_base_trainable(self, trainable: bool) -> None:
        """Freeze (or unfreeze) every backbone parameter except the head."""
        for name, param in self.cnn14.named_parameters():
            if not name.startswith("fc_audioset"):
                param.requires_grad = trainable

    def unfreeze_last_blocks(self, n_blocks: int = 1) -> None:
        """Phase 2: unfreeze the last ``n_blocks`` conv blocks + ``fc1``.

        Default (``n_blocks=1``) is the TDD 4.3 recipe: ``conv_block6`` +
        ``fc1`` only. With substantially more fault data, ``n_blocks=2``
        (conv_block5+6) trades a little stability for faster feature
        adaptation toward the fault signatures.
        """
        block_names = {
            f"conv_block{6 - i}." for i in range(n_blocks)
        }
        block_names.add("fc1.")
        for name, param in self.cnn14.named_parameters():
            if name.startswith(tuple(sorted(block_names))):
                param.requires_grad = True

    def trainable_parameter_names(self) -> list[str]:
        """Names of parameters currently requiring gradients (for logging)."""
        return [
            name for name, p in self.cnn14.named_parameters() if p.requires_grad
        ]

    def param_groups(
        self,
        mode: str = "head",
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
    ) -> list[dict]:
        """Differential-LR optimizer groups for the current phase (TDD 4.3).

        - ``mode="head"`` (Phase 1): only the new head, at ``head_lr``.
        - ``mode="finetune"`` (Phase 2): trainable backbone (conv_block6+fc1)
          at ``backbone_lr`` plus the head at ``head_lr``.
        """
        head_params = list(self.cnn14.fc_audioset.parameters())
        if mode == "head":
            return [{"params": head_params, "lr": head_lr, "name": "head"}]
        if mode == "finetune":
            # Any currently-trainable backbone param (whatever was unfrozen).
            backbone_params = [
                p for name, p in self.cnn14.named_parameters()
                if p.requires_grad and not name.startswith("fc_audioset")
            ]
            return [
                {"params": backbone_params, "lr": backbone_lr, "name": "backbone"},
                {"params": head_params, "lr": head_lr, "name": "head"},
            ]
        raise ValueError(f"Unknown mode: {mode!r} (expected 'head' or 'finetune')")

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self, waveform: torch.Tensor, mixup_lambda: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Raw waveform (batch, n_samples) -> PANNs-style output dict.

        Contains ``'logits'`` (batch, num_classes) for CrossEntropy training
        and ``'embedding'`` (batch, 2048) for any downstream use.
        """
        return self.cnn14(waveform, mixup_lambda)