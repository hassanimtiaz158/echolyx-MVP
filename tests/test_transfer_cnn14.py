"""Unit tests for TransferCnn14 (TDD sec 4) without the 308 MB checkpoint.

These tests fabricate a structurally valid, randomly-initialized checkpoint so
the model can be instantiated and its freeze/fine-tune behaviour verified with
no network access. They skip with a clear message if ``torchlibrosa`` (required
to build the vendored Cnn14) is not installed.

Coverage:
- head swap to ``Linear(2048, 2)`` with correct logits shape
- ``freeze_base=True`` freezes everything except the head (Phase 1)
- ``unfreeze_last_blocks()`` unfreezes conv_block6 + fc1 only (Phase 2)
- ``param_groups`` yields the expected Phase 1 / Phase 2 LR groups
- upstream checkpoint variations (bare state_dict, {"model": ...},
  ``module.``-prefixed keys) all load
- missing checkpoint path raises a clear error
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "torchlibrosa",
    reason="torchlibrosa is required to build the vendored Cnn14; run: pip install -r requirements.txt",
)

from src.models.cnn14 import Cnn14
from src.models.transfer_cnn14 import (
    AUDIOSET_CLASSES,
    CNN14_FMAX,
    CNN14_FMIN,
    CNN14_HOP_SIZE,
    CNN14_MEL_BINS,
    CNN14_SAMPLE_RATE,
    CNN14_WINDOW_SIZE,
    TransferCnn14,
)

BATCH = 2
NUM_SAMPLES = CNN14_SAMPLE_RATE  # 1 second @ 32 kHz


@pytest.fixture()
def checkpoint_path(tmp_path: Path) -> Path:
    """A structurally valid random-weight Cnn14 checkpoint (527 classes)."""
    model = Cnn14(
        sample_rate=CNN14_SAMPLE_RATE,
        window_size=CNN14_WINDOW_SIZE,
        hop_size=CNN14_HOP_SIZE,
        mel_bins=CNN14_MEL_BINS,
        fmin=CNN14_FMIN,
        fmax=CNN14_FMAX,
        classes_num=AUDIOSET_CLASSES,
    )
    path = tmp_path / "fake_Cnn14_mAP=0.431.pth"
    torch.save(model.state_dict(), path)
    return path


def test_head_is_binary_linear_and_backbone_frozen(checkpoint_path: Path):
    model = TransferCnn14(checkpoint_path, num_classes=2, freeze_base=True)
    head = model.cnn14.fc_audioset
    assert isinstance(head, torch.nn.Linear)
    assert head.in_features == 2048 and head.out_features == 2

    # Requires the whole forward pass (including torchlibrosa STFT).
    model.eval()
    with torch.no_grad():
        wave = torch.zeros(BATCH, NUM_SAMPLES)
        out = model(wave)
    assert out["logits"].shape == (BATCH, 2)
    assert out["embedding"].shape == (BATCH, 2048)

    # Phase 1: every backbone param frozen, only head trainable.
    frozen = [n for n, p in model.cnn14.named_parameters() if not p.requires_grad]
    assert "conv_block1.conv1.weight" in frozen
    assert "fc1.weight" in frozen
    assert "fc_audioset.weight" not in frozen
    assert all(p.requires_grad for p in head.parameters())


def test_unfreeze_last_blocks_only(checkpoint_path: Path):
    model = TransferCnn14(checkpoint_path, num_classes=2, freeze_base=True)
    model.unfreeze_last_blocks()

    names = {n for n, p in model.cnn14.named_parameters() if p.requires_grad}
    # Phase 2 targets: conv_block6 + fc1 (+ head always trainable).
    assert "conv_block6.conv1.weight" in names
    assert "conv_block6.bn2.weight" in names
    assert "fc1.weight" in names
    assert "fc_audioset.weight" in names
    # Early layers stay frozen.
    assert "conv_block1.conv1.weight" not in names
    assert "conv_block5.conv2.weight" not in names


def test_param_groups(checkpoint_path: Path):
    model = TransferCnn14(checkpoint_path, num_classes=2, freeze_base=True)

    phase1 = model.param_groups("head", backbone_lr=1e-5, head_lr=1e-3)
    assert len(phase1) == 1
    assert phase1[0]["lr"] == 1e-3
    assert phase1[0]["name"] == "head"

    model.unfreeze_last_blocks()
    phase2 = model.param_groups("finetune", backbone_lr=1e-5, head_lr=1e-4)
    assert len(phase2) == 2
    assert phase2[0]["name"] == "backbone" and phase2[0]["lr"] == 1e-5
    assert phase2[1]["name"] == "head" and phase2[1]["lr"] == 1e-4


def test_model_wrapped_and_module_prefix_chkpts(tmp_path: Path, checkpoint_path: Path):
    # "model"-wrapped dict + module.-prefixed keys (DataParallel) both load.
    model = Cnn14(
        sample_rate=CNN14_SAMPLE_RATE,
        window_size=CNN14_WINDOW_SIZE,
        hop_size=CNN14_HOP_SIZE,
        mel_bins=CNN14_MEL_BINS,
        fmin=CNN14_FMIN,
        fmax=CNN14_FMAX,
        classes_num=AUDIOSET_CLASSES,
    )
    raw = model.state_dict()
    wrapped = {"model": {"module." + k: v for k, v in raw.items()}}

    path = tmp_path / "wrapped.pth"
    torch.save(wrapped, path)
    net = TransferCnn14(path, num_classes=2, freeze_base=True)
    assert net.cnn14.fc_audioset.out_features == 2


def test_missing_checkpoint_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="download_checkpoint"):
        TransferCnn14(tmp_path / "does_not_exist.pth", num_classes=2)