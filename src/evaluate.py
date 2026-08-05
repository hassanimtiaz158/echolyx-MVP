"""Evaluation: metrics, confusion matrix, real-clip sanity check (TDD sec 5).

Flow (``python -m src.evaluate --config configs/config.yaml``):
1. Load ``configs/config.yaml`` (same loader as train.py) and the best
   checkpoint from ``<checkpoint_dir>/best.pt``.
2. Rebuild the data pipeline with the same seed/split logic as training and
   score the held-out **test** split (never augmented).
3. Report accuracy, precision, recall, F1 — binary with **Faulty as the
   positive class** — plus a full ``classification_report``. Recall on Faulty
   is the priority metric (a missed fault costs the most, TDD 5 / PRD 8).
4. Save artifacts: confusion matrix plot and training-curve plots (loss +
   accuracy over both phases, from ``training_history.json``) under
   ``<artifact_dir>/``.
5. Run inference on the held-out real ``fan's frame broken.mp3`` clip and
   print it explicitly under "REAL-WORLD SANITY CHECK". This number is a
   spot-check (TDD sec 8) and is NEVER mixed into the aggregate test metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe for Kaggle/Colab
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data.collect import collect, find_sanity_clip
from src.data.dataset import make_dataloaders
from src.inference import load_model, predict_single_file
from src.models.transfer_cnn14 import TransferCnn14
from src.train import load_config, set_seed

logger = logging.getLogger(__name__)

SANITY_HEADER = "REAL-WORLD SANITY CHECK"


def _predict_test(model: TransferCnn14, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run the full test split; returns (predictions, ground-truth labels)."""
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for waveforms, batch_labels in loader:
            output = model(waveforms.to(device))["logits"]
            preds.extend(output.argmax(dim=-1).cpu().tolist())
            labels.extend(batch_labels.tolist())
    return np.asarray(preds), np.asarray(labels)


def report_test_metrics(preds: np.ndarray, labels: np.ndarray, class_names: list[str]) -> None:
    """Print accuracy/precision/recall/F1 (Faulty = positive) + full report."""
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)

    print("=" * 60)
    print("TEST SPLIT METRICS (Faulty = positive class)")
    print("=" * 60)
    print(f"accuracy : {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall   : {rec:.4f}   <- priority metric (missed faults are costly)")
    print(f"F1       : {f1:.4f}")
    print()
    print(classification_report(labels, preds, target_names=class_names, digits=3, zero_division=0))


def save_confusion_matrix(preds: np.ndarray, labels: np.ndarray, class_names: list[str], path: Path) -> None:
    """Render and save the confusion matrix plot (pitch-deck artifact)."""
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], class_names)
    ax.set_yticks([0, 1], class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test split)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def save_training_curves(history_path: Path, out_path: Path) -> None:
    """Plot train/val loss + accuracy over both phases from the history JSON."""
    if not history_path.is_file():
        logger.warning("No training history at %s — skipping curve plots.", history_path)
        return

    history = json.loads(history_path.read_text(encoding="utf-8"))["epochs"]
    if not history:
        logger.warning("Training history is empty — skipping curve plots.")
        return

    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Loss (both phases)"); axes[0].legend()
    axes[1].plot(epochs, [r["train_acc"] for r in history], label="train")
    axes[1].plot(epochs, [r["val_acc"] for r in history], label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy (both phases)"); axes[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def run_sanity_check(
    model: TransferCnn14,
    clip_path: Path | None,
    config: dict,
) -> None:
    """Inference on the held-out real faulty clip; NEVER merged into metrics.

    Delegates to ``src.inference.predict_single_file`` (single source of
    truth for preprocess -> softmax).
    """
    print()
    print("=" * 60)
    print(f"{SANITY_HEADER} (excluded from training; not part of the metrics above)")
    print("=" * 60)
    if clip_path is None or not clip_path.is_file():
        print("Held-out clip NOT FOUND - sanity check skipped (reported honestly).")
        return

    pred = predict_single_file(clip_path, model, config)
    print(f"file        : {clip_path}")
    print(f"prediction  : {pred.label} (confidence {pred.probabilities[pred.label]:.3f})")
    print(f"probabilities: " + ", ".join(
        f"{name}: {p:.3f}" for name, p in pred.probabilities.items()
    ))
    print("note        : single real-world spot-check per TDD sec 5/8 - a "
          "credible demo data point, not a statistically meaningful validation set.")


def evaluate(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | config: %s", device, config_path)

    # Data pipeline (same seed as training -> identical test split).
    filepaths, labels = collect(
        cfg["mimii_root"], cfg["extra_normal_root"], cfg.get("sanity_clip_filename")
    )
    _, _, test_loader, _ = make_dataloaders(
        filepaths,
        labels,
        sample_rate=cfg["sample_rate"],
        num_samples=cfg["sample_rate"] * cfg["clip_seconds"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 0),
        seed=cfg["seed"],
    )

    # Model + best checkpoint.
    checkpoint_path = Path(cfg["checkpoint_dir"]) / "best.pt"
    model = load_model(
        cfg["backbone_checkpoint"], checkpoint_path, cfg["num_classes"]
    ).to(device)

    # Aggregate metrics on the test split only.
    preds, labels_arr = _predict_test(model, test_loader, device)
    report_test_metrics(preds, labels_arr, cfg["class_names"])

    # Artifacts.
    artifact_dir = Path(cfg["artifact_dir"])
    save_confusion_matrix(preds, labels_arr, cfg["class_names"], artifact_dir / "confusion_matrix.png")
    save_training_curves(artifact_dir / "training_history.json", artifact_dir / "training_curves.png")

    # Held-out real-world sanity check (kept strictly separate).
    sanity_path = find_sanity_clip(cfg["extra_normal_root"], cfg["sanity_clip_filename"])
    run_sanity_check(model, sanity_path, cfg)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate the Echolyx fan classifier.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()