"""Inference: the ONE place for preprocess -> model -> softmax (TDD sec 6).

``predict_single_file(path, model, config)`` is shared by ``evaluate.py``
(sanity check) and the Gradio app, so the preprocessing/probability logic
lives in exactly one spot. It reuses ``load_fixed_length_audio`` from the data
pipeline, so there is no duplicated audio loading.

Also exposes ``load_model`` (instantiate ``TransferCnn14`` + load the best
checkpoint) used by both consumers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.preprocess import load_fixed_length_audio
from src.models.transfer_cnn14 import TransferCnn14

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Result of a single-clip inference."""

    label: str  # e.g. "Faulty" (or "Uncertain" when the uncertain-gate is on)
    probabilities: dict[str, float]  # e.g. {"Normal": 0.12, "Faulty": 0.88}
    confidence: float  # P(label), the highest class probability


def decide_label(
    probabilities: dict[str, float],
    class_names: list[str],
    faulty_cutoff: float | None = None,
    uncertain_low: float | None = None,
) -> str:
    """Turn class probabilities into a decision label (TDD sec 5).

    - No cutoff: argmax (classic 0.5 boundary).
    - ``faulty_cutoff``: label Faulty only when P(Faulty) >= cutoff, else
      Normal — the calibrated operating point under class-weighted CE.
    - ``uncertain_low`` (with a cutoff): probabilities in
      (uncertain_low, faulty_cutoff) are "Uncertain" — the deployment gate
      that converts borderline guesses into "re-record" instead of a wrong
      confident answer.
    """
    p_faulty = probabilities[class_names[1]]
    if not isinstance(faulty_cutoff, (int, float)):
        return max(class_names, key=lambda name: probabilities[name])
    if isinstance(uncertain_low, (int, float)) and uncertain_low < p_faulty < faulty_cutoff:
        return "Uncertain"
    return class_names[1] if p_faulty >= faulty_cutoff else class_names[0]


def load_model(
    backbone_checkpoint: str | Path,
    checkpoint_path: str | Path,
    num_classes: int,
) -> TransferCnn14:
    """Instantiate TransferCnn14 and load best/final checkpoint weights.

    The checkpoint carries TDD sec 7 metadata (sample_rate, clip_length_sec,
    class_names) that callers may read from ``ckpt["metadata"]``.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path} - run `python -m src.train` first."
        )
    model = TransferCnn14(backbone_checkpoint, num_classes=num_classes, freeze_base=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded checkpoint %s (metadata: %s)", checkpoint_path, ckpt.get("metadata"))
    return model


def predict_single_file(
    path: str | Path,
    model: TransferCnn14,
    config: dict,
) -> Prediction:
    """Classify one audio clip into Normal/Faulty with softmax probabilities.

    Preprocessing follows the config (sample_rate / clip_seconds, TDD 3.2).
    Decision rule: config ``faulty_cutoff`` (+ optional ``uncertain_low``
    gate, TDD sec 5) — see ``decide_label``. Deterministic inference path:
    ``model.eval()`` and no gradient tracking.
    """
    waveform = load_fixed_length_audio(
        path,
        config["sample_rate"],
        config["sample_rate"] * config["clip_seconds"],
    )
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(waveform.unsqueeze(0).to(device))["logits"]
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    class_names = config["class_names"]
    probabilities = {name: float(p) for name, p in zip(class_names, probs)}
    label = decide_label(
        probabilities,
        class_names,
        config.get("faulty_cutoff"),
        config.get("uncertain_low"),
    )
    confidence = max(probabilities.values())
    return Prediction(label=label, probabilities=probabilities, confidence=confidence)


def predict_ensemble(
    path: str | Path,
    models: list[TransferCnn14],
    config: dict,
) -> Prediction:
    """Ensemble prediction by averaging probabilities across multiple models.

    More robust than single model - reduces variance and improves accuracy.
    """
    waveform = load_fixed_length_audio(
        path,
        config["sample_rate"],
        config["sample_rate"] * config["clip_seconds"],
    )
    device = next(models[0].parameters()).device
    class_names = config["class_names"]
    num_classes = len(class_names)

    all_probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(waveform.unsqueeze(0).to(device))["logits"]
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu()
        all_probs.append(probs)

    # Average probabilities
    avg_probs = torch.stack(all_probs).mean(dim=0).tolist()
    probabilities = {name: float(p) for name, p in zip(class_names, avg_probs)}
    label = decide_label(
        probabilities,
        class_names,
        config.get("faulty_cutoff"),
        config.get("uncertain_low"),
    )
    confidence = max(probabilities.values())
    return Prediction(label=label, probabilities=probabilities, confidence=confidence)
