"""Two-phase fine-tuning of TransferCnn14 (TDD sec 4.3-4.4, 3.4, 7).

Pipeline:
1. Load ``configs/config.yaml`` (paths resolved relative to the working
   directory — no hard-coded /kaggle/ paths; works locally, in Kaggle
   notebook cells, or Colab).
2. Build the labeled manifest + dataset/splits/loaders via ``src.data``
   (``collect`` -> ``make_dataloaders``), seeded for reproducibility.
3. Instantiate ``TransferCnn14`` with a frozen backbone.
4. Phase 1 (frozen): train only the new head with Adam at ``phase1_lr``
   (default 1e-3) for ``phase1_epochs`` (default 8).
5. Phase 2: ``unfreeze_last_blocks()`` (conv_block6 + fc1 only) and train
   with differential LRs (backbone 1e-5, head 1e-4) for ``phase2_epochs``
   (default 15).
6. Loss: class-weighted ``CrossEntropyLoss``, weights ``total/(2*count)``
   (TDD sec 3.4), computed on the training split.
7. Best checkpoint selected by validation accuracy and saved to
   ``<checkpoint_dir>/best.pt`` every epoch it improves; the last model is
   also saved as ``final.pt``. The checkpoint carries metadata (sample rate,
   clip length, class names, epoch, phase, val accuracy).
8. Every epoch's train/val loss + accuracy is printed to stdout and appended
   to a history dict written as JSON under ``<artifact_dir>/training_history.json``
   for the evaluation plots.

Usage: ``python -m src.train --config configs/config.yaml``
(``--phase1-epochs``/``--phase2-epochs`` override the config, handy for
quick smoke runs.)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.data.collect import collect
from src.data.dataset import derive_group_keys, make_dataloaders
from src.models.transfer_cnn14 import TransferCnn14

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
BACKBONE_FILENAME = "Cnn14_mAP=0.431.pth"

# Optional keys + defaults so a minimal config still trains end-to-end.
DEFAULTS = {
    "num_classes": 2,
    "class_names": ["Normal", "Faulty"],
    "num_workers": 0,
    "mixup_alpha": 0.0,
    "balance_sampling": False,
    "group_by_source": False,
    "use_cosine": False,
    "phase1_epochs": 8,
    "phase1_lr": 1e-3,
    "phase2_epochs": 15,
    "phase2_backbone_lr": 1e-5,
    "phase2_head_lr": 1e-4,
    "phase2_unfreeze_blocks": 1,
    "use_class_weights": True,
}


def set_seed(seed: int) -> None:
    """Fix Python/NumPy/PyTorch/CUDA seeds for reproducibility (TDD sec 7)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    """Load config.yaml and fill unset optional keys with TDD defaults.

    Paths are interpreted relative to the current working directory so the
    same config works locally and on Kaggle/Colab (point your roots there).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in DEFAULTS.items():
        cfg.setdefault(key, value)
    cfg.setdefault(
        "backbone_checkpoint",
        str(Path(cfg["checkpoint_dir"]) / BACKBONE_FILENAME),
    )
    return cfg


def build_criterion(
    class_weights: torch.Tensor,
    use_class_weights: bool,
    device: torch.device,
) -> nn.CrossEntropyLoss:
    """Build the training criterion (TDD sec 3.4 + recipe hardening).

    ``use_class_weights=True`` (default): class-weighted CE with weights
    ``total/(2*count)``. ``use_class_weights=False``: plain CE — rebalancing
    then comes from the sampler alone (``balance_sampling``). Weighting BOTH
    the sampler and the loss double-counts the minority class: with ~8%
    Faulty in the pool and weak cross-machine signal, that 11x loss pressure
    collapses the model into "always Faulty" (seen in run 2). The flag makes
    the two rebalancing mechanisms mutually exclusive.
    """
    if use_class_weights:
        return nn.CrossEntropyLoss(weight=class_weights.to(device))
    return nn.CrossEntropyLoss()


def _mixup_batch(
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup (TDD sec 3.5): interpolate pairs drawn with random permutation.

    Lambdas come from ``Beta(alpha, alpha)``. Labels are one-hot mixed so the
    loss is the expected CE over both constituents. Returns
    ``(mixed_waveforms, mixed_labels)``.
    """
    batch = waveforms.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample((batch, 1)).to(waveforms.device)
    index = torch.randperm(batch, device=waveforms.device)
    mixed_waveforms = lam * waveforms + (1.0 - lam) * waveforms[index]
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    mixed_labels = lam * one_hot + (1.0 - lam) * one_hot[index]
    return mixed_waveforms, mixed_labels


def _run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    train: bool,
    mixup_alpha: float = 0.0,
    num_classes: int = 2,
) -> tuple[float, float]:
    """Run one train (if ``train``) or eval pass; returns (avg_loss, accuracy)."""
    model.train(train)
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(train):
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            if train and mixup_alpha > 0.0:
                waveforms, mix_targets = _mixup_batch(
                    waveforms, labels, mixup_alpha, num_classes
                )
            else:
                mix_targets = None

            if train:
                optimizer.zero_grad()
            output = model(waveforms)["logits"]
            if mix_targets is not None:
                loss = criterion(output, mix_targets)
            else:
                loss = criterion(output, labels)
            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            # For mixup batches this compares against the pre-mix labels —
            # an approximation; val accuracy drives checkpoint selection anyway.
            correct += (output.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


def train_phase(
    model: nn.Module,
    phase_name: str,
    optimizer,
    train_loader,
    val_loader,
    criterion,
    device: torch.device,
    epochs: int,
    checkpoint_dir: Path,
    history: list[dict],
    best_state: dict,
    cfg: dict,
    scheduler=None,
) -> None:
    """Train one phase for ``epochs``, logging every epoch and saving best VT."""
    best_acc = best_state["val_acc"]
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
            mixup_alpha=cfg.get("mixup_alpha", 0.0), num_classes=cfg["num_classes"],
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        record = {
            "phase": phase_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
"lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        logger.info(
            "%s ep %2d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | lr %.1e",
            phase_name, epoch, train_loss, train_acc, val_loss, val_acc,
            optimizer.param_groups[0]["lr"],
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state.update(
                {
                    "phase": phase_name,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "train_acc": train_acc,
                }
            )
            _save_checkpoint(
                model,
                best_state,
                cfg,
                checkpoint_dir / "best.pt",
                is_best=True,
            )
            logger.info("  -> new best val acc %.4f, checkpoint saved", val_acc)

        if scheduler is not None:  # cosine decay warmed down through the phase
            scheduler.step()

    # Keep the last phase's state so final.pt metadata is accurate.
    best_state["phase"] = phase_name
    best_state["epoch"] = epochs


def _save_checkpoint(
    model: nn.Module,
    state: dict,
    cfg: dict,
    path: Path,
    is_best: bool = False,
) -> None:
    """Save model weights + TDD sec 7 metadata (+ best-val accuracy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "sample_rate": cfg["sample_rate"],
            "clip_length_sec": cfg["clip_seconds"],
            "num_classes": cfg["num_classes"],
            "class_names": cfg["class_names"],
            "phase": state.get("phase"),
            "epoch": state.get("epoch"),
            "val_acc": state.get("val_acc"),
            "best": is_best,
        },
    }
    torch.save(ckpt, path)
    logger.info("Saved %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train Echolyx fan classifier.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--phase1-epochs", type=int, default=None)
    parser.add_argument("--phase2-epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.phase1_epochs is not None:
        cfg["phase1_epochs"] = args.phase1_epochs
    if args.phase2_epochs is not None:
        cfg["phase2_epochs"] = args.phase2_epochs

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | config: %s", device, args.config)

    # 1) Data pipeline (TDD sec 3).
    filepaths, labels = collect(
        cfg["mimii_root"], cfg["extra_normal_root"], cfg.get("sanity_clip_filename")
    )
    logger.info("Manifest: %d clips (%d normal / %d faulty)", len(filepaths),
                labels.count(0), labels.count(1))
    if len(filepaths) == 0:
        raise RuntimeError("No training clips found — check data roots in config.yaml")

    num_samples = cfg["sample_rate"] * cfg["clip_seconds"]
    groups = derive_group_keys(filepaths) if cfg.get("group_by_source", False) else None
    train_loader, val_loader, _, class_weights = make_dataloaders(
        filepaths,
        labels,
        sample_rate=cfg["sample_rate"],
        num_samples=num_samples,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
        balance_train=cfg.get("balance_sampling", False),
        groups=groups,
    )
    if groups is not None:
        logger.info("Grouped splits by machine/source (%d groups)", len(set(groups)))
    use_class_weights = cfg.get("use_class_weights", True)
    criterion = build_criterion(class_weights, use_class_weights, device)
    if use_class_weights:
        logger.info(
            "Loss: class-weighted CE (weights=%s)",
            [round(w, 4) for w in class_weights.tolist()],
        )
    else:
        # Rebalance via the sampler only (balance_sampling), NOT the loss:
        # weighting both double-counts the minority and collapses the model
        # toward the minority when cross-machine signal is weak.
        logger.info("Loss: plain CE (rebalance via balance_sampling sampler only)")

    # 2) Model (TDD sec 4).
    model = TransferCnn14(
        cfg["backbone_checkpoint"],
        num_classes=cfg["num_classes"],
        freeze_base=True,
    ).to(device)
    logger.info(
        "TransferCnn14 ready; trainable params (Phase 1 head only): %d",
        sum(p.numel() for p in model.cnn14.fc_audioset.parameters()),
    )

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    artifact_dir = Path(cfg["artifact_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_state = {"val_acc": float("-inf")}

    # 3) Phase 1 — frozen backbone, head only (TDD 4.3).
    logger.info("Phase 1: frozen backbone, head-only Adam lr=%g, %d epochs",
                cfg["phase1_lr"], cfg["phase1_epochs"])
    opt1 = torch.optim.Adam(
        model.param_groups("head", head_lr=cfg["phase1_lr"]), lr=cfg["phase1_lr"]
    )
    sched1 = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            opt1, T_max=cfg["phase1_epochs"], eta_min=cfg["phase1_lr"] * 0.01
        )
        if cfg.get("use_cosine", False)
        else None
    )
    train_phase(
        model, "phase1", opt1, train_loader, val_loader, criterion, device,
        cfg["phase1_epochs"], checkpoint_dir, history, best_state, cfg,
        scheduler=sched1,
    )

    # 4) Phase 2 — unfreeze last blocks, differential LR (TDD 4.3).
    model.unfreeze_last_blocks(n_blocks=cfg.get("phase2_unfreeze_blocks", 1))
    logger.info("Phase 2: unfreezing conv_block6 + fc1; backbone lr=%g head lr=%g",
                cfg["phase2_backbone_lr"], cfg["phase2_head_lr"])
    opt2 = torch.optim.Adam(
        model.param_groups(
            "finetune",
            backbone_lr=cfg["phase2_backbone_lr"],
            head_lr=cfg["phase2_head_lr"],
        ),
        lr=cfg["phase2_head_lr"],
    )
    sched2 = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            opt2, T_max=cfg["phase2_epochs"], eta_min=cfg["phase2_head_lr"] * 0.01
        )
        if cfg.get("use_cosine", False)
        else None
    )
    train_phase(
        model, "phase2", opt2, train_loader, val_loader, criterion, device,
        cfg["phase2_epochs"], checkpoint_dir, history, best_state, cfg,
        scheduler=sched2,
    )

    # 5) Artifacts: final model + JSON history for plotting.
    _save_checkpoint(model, best_state, cfg, checkpoint_dir / "final.pt", is_best=False)

    (artifact_dir / "training_history.json").write_text(
        json.dumps({"best": dict(best_state), "epochs": history}, indent=2),
        encoding="utf-8",
    )
    logger.info("Best val acc %.4f reached at %s epoch %d",
                best_state["val_acc"], best_state.get("phase"), best_state.get("epoch"))
    logger.info("Artifacts: %s, %s",
                checkpoint_dir / "best.pt", artifact_dir / "training_history.json")


if __name__ == "__main__":
    main()