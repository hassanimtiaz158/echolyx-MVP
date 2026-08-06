"""Smoke tests for the shared inference path (TDD sec 6).

Uses a fabricated random-weight checkpoint (same trick as
``test_transfer_cnn14.py``) so no 300 MB download is needed. Skips gracefully
if torchlibrosa is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "torchlibrosa",
    reason="torchlibrosa required to build the vendored Cnn14; run: pip install -r requirements.txt",
)

import soundfile as sf

from src.inference import (
    Prediction,
    decide_label,
    load_model,
    predict_single_file,
)
from src.models.cnn14 import Cnn14
from src.models.transfer_cnn14 import (
    AUDIOSET_CLASSES,
    CNN14_FMIN,
    CNN14_FMAX,
    CNN14_HOP_SIZE,
    CNN14_MEL_BINS,
    CNN14_SAMPLE_RATE,
    CNN14_WINDOW_SIZE,
    TransferCnn14,
)

CONFIG = {
    "sample_rate": CNN14_SAMPLE_RATE,
    "clip_seconds": 1,
    "class_names": ["Normal", "Faulty"],
}
CLIP = CONFIG["sample_rate"] * CONFIG["clip_seconds"]


@pytest.fixture()
def model_and_config(tmp_path: Path):
    """A TransferCnn14 loaded from a fake random-weight checkpoint."""
    backbone = Cnn14(
        sample_rate=CNN14_SAMPLE_RATE,
        window_size=CNN14_WINDOW_SIZE,
        hop_size=CNN14_HOP_SIZE,
        mel_bins=CNN14_MEL_BINS,
        fmin=CNN14_FMIN,
        fmax=CNN14_FMAX,
        classes_num=AUDIOSET_CLASSES,
    )
    # 1) raw backbone weights (what TransferCnn14 consumes at build time)
    backbone_path = tmp_path / "fake_backbone.pth"
    torch.save(backbone.state_dict(), backbone_path)

    # 2) a full TransferCnn14 checkpoint with the 2-class head, exactly like
    #    src.train.py saves best.pt
    txf = TransferCnn14(backbone_path, num_classes=2, freeze_base=False)
    ckpt_path = tmp_path / "fake_ckpt.pth"
    torch.save({"model_state_dict": txf.state_dict(), "metadata": {}}, ckpt_path)

    model = load_model(backbone_path, ckpt_path, num_classes=2)
    model.eval()
    return model


def test_predict_single_file_shapes_and_probabilities(model_and_config, tmp_path):
    model = model_and_config
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.sin(2 * np.pi * 220 * np.arange(CLIP) / CNN14_SAMPLE_RATE).astype(np.float32), CNN14_SAMPLE_RATE)

    pred = predict_single_file(audio, model, CONFIG)

    assert isinstance(pred, Prediction)
    assert pred.label in {"Normal", "Faulty"}
    assert set(pred.probabilities) == {"Normal", "Faulty"}
    assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert pred.probabilities[pred.label] == max(pred.probabilities.values())


def test_decide_label_rules():
    names = ["Normal", "Faulty"]
    # argmax without cutoff
    assert decide_label({"Normal": 0.4, "Faulty": 0.6}, names) == "Faulty"
    assert decide_label({"Normal": 0.9, "Faulty": 0.1}, names) == "Normal"
    # calibrated cutoff: Faulty only above the cutoff
    assert decide_label({"Normal": 0.1, "Faulty": 0.9}, names, 0.95) == "Normal"
    assert decide_label({"Normal": 0.02, "Faulty": 0.98}, names, 0.95) == "Faulty"
    # uncertain-gate: middle band -> Uncertain
    assert decide_label({"Normal": 0.5, "Faulty": 0.5}, names, 0.95, 0.05) == "Uncertain"
    assert decide_label({"Normal": 0.99, "Faulty": 0.01}, names, 0.95, 0.05) == "Normal"
    assert decide_label({"Normal": 0.01, "Faulty": 0.99}, names, 0.95, 0.05) == "Faulty"


def test_predict_respects_configured_cutoff(model_and_config, tmp_path):
    model = model_and_config
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.sin(2 * np.pi * 220 * np.arange(CLIP) / CNN14_SAMPLE_RATE).astype(np.float32), CNN14_SAMPLE_RATE)

    # argmax label (baseline)
    base = predict_single_file(audio, model, CONFIG)

    # Extreme cutoffs flip the label deterministically: 0.0 -> always Faulty,
    # 1.0 -> always Normal. This proves the decision rule is applied.
    cfg_all_faulty = {**CONFIG, "faulty_cutoff": 0.0}
    cfg_all_normal = {**CONFIG, "faulty_cutoff": 1.0}
    assert predict_single_file(audio, model, cfg_all_faulty).label == "Faulty"
    assert predict_single_file(audio, model, cfg_all_normal).label == "Normal"

    # A cutoff at the argmax probability leaves the argmax decision unchanged.
    p_argmax = base.probabilities[base.label]
    cfg_same = {**CONFIG, "faulty_cutoff": p_argmax}
    assert predict_single_file(audio, model, cfg_same).label == base.label

    # Uncertain-gate active: label must stay within the three-way rule and the
    # confidence field always carries the top class probability.
    cfg_gate = {**CONFIG, "faulty_cutoff": 0.95, "uncertain_low": 0.05}
    gated = predict_single_file(audio, model, cfg_gate)
    assert gated.label in {"Normal", "Faulty", "Uncertain"}
    assert gated.confidence == max(gated.probabilities.values())


def test_load_model_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_model(tmp_path / "nope.pth", tmp_path / "nope.pth", num_classes=2)