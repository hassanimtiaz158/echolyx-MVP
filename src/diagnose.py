"""Frozen-embedding probe: is there ANY cross-machine fault signal?

Runs 1-3 all collapsed: the fine-tuned model learned a MIMII-vs-Freesound
shortcut instead of "broken fan". Before pivoting (more real data, one-class
anomaly, different backbone), answer the root question cheaply:

1. Freeze the AudioSet-pretrained backbone (no fine-tuning, no head training).
2. Extract the 2048-d ``embedding`` for the train/val/test splits (same
   grouped split as training so the numbers are directly comparable).
3. Fit a plain logistic-regression probe on the TRAIN embeddings only and
   score the grouped TEST split.

- If the probe separates test machines (AUC >> 0.5): the signal IS in the
  frozen features; the two-phase fine-tuning was destroying it (head overfit
  to the MIMII shortcut). Fix: freeze backbone, train probe head only.
- If the probe is at chance (~0.5): the AudioSet features cannot represent
  cross-machine fan faults from MIMII audio alone. MIMII-only training is
  capped; the path forward is real recorded data (Freesound/phone clips) or
  per-machine one-class adaptation.

Usage: ``python -m src.diagnose --config configs/config.yaml``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.collect import collect
from src.data.dataset import derive_group_keys, make_dataloaders
from src.models.transfer_cnn14 import TransferCnn14
from src.train import load_config, set_seed

logger = logging.getLogger(__name__)


def extract_embeddings(
    model: TransferCnn14,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a loader through the frozen backbone; returns (embeddings, labels)."""
    model.eval()
    embs: list[np.ndarray] = []
    labels: list[int] = []
    with torch.no_grad():
        for waveforms, batch_labels in loader:
            out = model(waveforms.to(device))
            embs.append(out["embedding"].cpu().numpy())
            labels.extend(batch_labels.tolist())
    return np.vstack(embs), np.asarray(labels)


def probe_split(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    test_emb: np.ndarray,
    test_labels: np.ndarray,
    seed: int = 42,
) -> dict:
    """Fit a logistic probe on train embeddings, score the test split.

    The probe is deliberately weak (plain L2-regularized LR, no tuning) so a
    strong result means the frozen features themselves carry the signal.
    """
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(train_emb, train_labels)
    probs = clf.predict_proba(test_emb)[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "probs": probs,
        "preds": preds,
        "labels": test_labels,
        "accuracy": accuracy_score(test_labels, preds),
        "precision": precision_score(test_labels, preds, pos_label=1, zero_division=0),
        "recall": recall_score(test_labels, preds, pos_label=1, zero_division=0),
        "auc": roc_auc_score(test_labels, probs),
    }


def diagnose(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | config: %s", device, config_path)

    filepaths, labels = collect(
        cfg["mimii_root"], cfg["extra_normal_root"], cfg.get("sanity_clip_filename")
    )
    groups = derive_group_keys(filepaths) if cfg.get("group_by_source", False) else None
    if groups is None:
        logger.warning("group_by_source is false — results are NOT cross-machine!")
    train_loader, _, test_loader, _ = make_dataloaders(
        filepaths,
        labels,
        sample_rate=cfg["sample_rate"],
        num_samples=cfg["sample_rate"] * cfg["clip_seconds"],
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 0),
        seed=cfg["seed"],
        groups=groups,
    )
    if groups is not None:
        logger.info("Grouped splits by machine/source (%d groups)", len(set(groups)))

    # Frozen backbone, head swapped out by TransferCnn14 but never trained.
    model = TransferCnn14(
        cfg["backbone_checkpoint"],
        num_classes=cfg["num_classes"],
        freeze_base=True,
    ).to(device)

    train_emb, train_labels = extract_embeddings(model, train_loader, device)
    test_emb, test_labels = extract_embeddings(model, test_loader, device)
    logger.info(
        "Embeddings: train %s (%d clips), test %s (%d clips)",
        train_emb.shape, len(train_labels), test_emb.shape, len(test_labels),
    )

    result = probe_split(train_emb, train_labels, test_emb, test_labels, seed=cfg["seed"])

    print("=" * 60)
    print("FROZEN-EMBEDDING PROBE (cross-machine test, Faulty = positive)")
    print("=" * 60)
    print(f"accuracy : {result['accuracy']:.4f}")
    print(f"precision: {result['precision']:.4f}")
    print(f"recall   : {result['recall']:.4f}")
    print(f"AUC      : {result['auc']:.4f}")
    if result["auc"] >= 0.75:
        print("VERDICT: frozen features separate unseen machines -> fine-tuning was")
        print("         destroying the signal. Keep the backbone frozen; train the")
        print("         probe head (and stop two-phase unfreezing).")
    elif result["auc"] >= 0.6:
        print("VERDICT: weak but real signal in frozen features. Worth a frozen-")
        print("         backbone head-only run; expect modest unseen-machine gains.")
    else:
        print("VERDICT: at chance. AudioSet features cannot represent cross-machine")
        print("         fan faults from MIMII audio. MIMII-only training is capped;")
        print("         pivot to real recorded data or per-machine one-class fit.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Frozen-embedding cross-machine probe.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()
    diagnose(args.config)


if __name__ == "__main__":
    main()
