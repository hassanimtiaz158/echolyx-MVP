"""Leave-one-group-out cross-validation: honest, low-variance unseen-machine
accuracy (addendum to TDD sec 5).

Why this exists: with a single grouped train/val/test split, the reported
"unseen-machine" test accuracy depends entirely on which 1-2 machine groups
the random split happened to draw into test -- with only a handful of
Faulty-containing ("mixed") machine groups, that single draw is noisy (this
is what caused epoch-to-epoch val accuracy to swing 30+ points in early
runs). This script instead trains one model per mixed group, holding that
whole group out as the fold's test split, and reports the mean/std across
folds -- a far more trustworthy estimate of "how will this do on a fan the
model has never heard."

As a side benefit, the fold checkpoints (none of which ever saw the
real-world holdout set) can be ensembled for a free accuracy/robustness
bump on that holdout via ``src.inference.predict_ensemble``.

Usage: ``python -m src.cross_validate --config configs/config.yaml``
``--phase1-epochs``/``--phase2-epochs`` override the config per fold (handy
for a cheap smoke-test CV pass before committing to the full epoch budget
across every fold). ``--max-folds N`` runs only the first N folds (in sorted
group-name order) -- useful to fit within a single Kaggle session's time
budget; rerun without it (or with a larger N) later to continue, since
folds whose checkpoint already exists are skipped and just re-evaluated
unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.collect import collect
from src.data.dataset import (
    FanSoundDataset,
    build_leave_one_group_out_splits,
    derive_group_keys,
    get_class_weights,
    get_sampling_weights,
)
from src.evaluate import _metrics_at_threshold
from src.inference import load_model, predict_ensemble
from src.models.transfer_cnn14 import TransferCnn14
from src.train import build_criterion, load_config, set_seed, train_phase

logger = logging.getLogger(__name__)

_SAFE_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _safe_name(group: str) -> str:
    """Turn a group key (e.g. ``dcase2020:id_00``) into a filesystem-safe name."""
    return _SAFE_RE.sub("_", group)


def _make_fold_loaders(train_split, val_split, test_split, cfg):
    num_samples = cfg["sample_rate"] * cfg["clip_seconds"]
    train_ds = FanSoundDataset(
        train_split.filepaths, train_split.labels, cfg["sample_rate"], num_samples, augment=True
    )
    val_ds = FanSoundDataset(
        val_split.filepaths, val_split.labels, cfg["sample_rate"], num_samples, augment=False
    )
    test_ds = FanSoundDataset(
        test_split.filepaths, test_split.labels, cfg["sample_rate"], num_samples, augment=False
    )

    sampler = None
    if cfg.get("balance_sampling", False):
        weights = get_sampling_weights(train_split.labels, power=cfg.get("sampling_power", 1.0))
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    num_workers = cfg.get("num_workers", 0)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=sampler is None,
        sampler=sampler, num_workers=num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=num_workers)
    class_weights = get_class_weights(train_split.labels)
    return train_loader, val_loader, test_loader, class_weights


def _collect_probs(model, loader, device):
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for waveforms, batch_labels in loader:
            logits = model(waveforms.to(device))["logits"]
            p_faulty = torch.softmax(logits, dim=-1)[:, 1]
            probs.extend(p_faulty.cpu().tolist())
            labels.extend(batch_labels.tolist())
    return np.asarray(probs), np.asarray(labels)


def train_fold(held_out_group, train_split, val_split, cfg, device, fold_ckpt_dir):
    """Train one fold's model (phase1 + phase2) and save its best checkpoint."""
    train_loader, val_loader, _, class_weights = _make_fold_loaders(
        train_split, val_split, val_split, cfg  # test loader unused here, val stands in
    )
    use_focal_loss = cfg.get("use_focal_loss", True)
    use_class_weights = cfg.get("use_class_weights", False)
    criterion = build_criterion(class_weights, use_class_weights, device, use_focal_loss)

    model = TransferCnn14(
        cfg["backbone_checkpoint"],
        num_classes=cfg["num_classes"],
        freeze_base=True,
        use_attention_head=cfg.get("use_attention_head", True),
    ).to(device)

    history: list[dict] = []
    best_state = {"val_acc": float("-inf")}
    grad_accum = cfg.get("grad_accum_steps", 1)

    opt1 = torch.optim.Adam(
        model.param_groups("head", head_lr=cfg["phase1_lr"]), lr=cfg["phase1_lr"]
    )
    sched1 = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            opt1, T_max=max(cfg["phase1_epochs"], 1), eta_min=cfg["phase1_lr"] * 0.01
        )
        if cfg.get("use_cosine", False) else None
    )
    train_phase(
        model, "phase1", opt1, train_loader, val_loader, criterion, device,
        cfg["phase1_epochs"], fold_ckpt_dir, history, best_state, cfg,
        scheduler=sched1, warmup_epochs=cfg.get("phase1_warmup_epochs", 0),
        grad_accum_steps=grad_accum,
    )

    model.unfreeze_last_blocks(n_blocks=cfg.get("phase2_unfreeze_blocks", 1))
    opt2 = torch.optim.Adam(
        model.param_groups(
            "finetune", backbone_lr=cfg["phase2_backbone_lr"], head_lr=cfg["phase2_head_lr"]
        ),
        lr=cfg["phase2_head_lr"],
    )
    sched2 = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            opt2, T_max=max(cfg["phase2_epochs"], 1), eta_min=cfg["phase2_head_lr"] * 0.01
        )
        if cfg.get("use_cosine", False) else None
    )
    train_phase(
        model, "phase2", opt2, train_loader, val_loader, criterion, device,
        cfg["phase2_epochs"], fold_ckpt_dir, history, best_state, cfg,
        scheduler=sched2, grad_accum_steps=grad_accum,
    )

    (fold_ckpt_dir / "history.json").write_text(
        json.dumps({"held_out_group": held_out_group, "best": best_state, "epochs": history}, indent=2),
        encoding="utf-8",
    )


def evaluate_fold(held_out_group, train_split, val_split, test_split, cfg, device, fold_ckpt_dir):
    """Load the fold's best checkpoint and score it on the held-out group's clips."""
    best_ckpt = fold_ckpt_dir / "best.pt"
    if not best_ckpt.is_file():
        raise FileNotFoundError(f"No checkpoint at {best_ckpt} -- training for this fold failed")

    model = load_model(cfg["backbone_checkpoint"], best_ckpt, cfg["num_classes"]).to(device)

    _, val_loader, test_loader, _ = _make_fold_loaders(train_split, val_split, test_split, cfg)

    val_probs, val_labels = _collect_probs(model, val_loader, device)
    configured_cutoff = cfg.get("faulty_cutoff")
    if configured_cutoff is not None:
        best_t = float(configured_cutoff)
    else:
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.05, 0.96, 0.01):
            _, _, _, f1 = _metrics_at_threshold(val_probs, val_labels, t)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)

    test_probs, test_labels = _collect_probs(model, test_loader, device)
    preds = (test_probs >= best_t).astype(int)

    result = {
        "held_out_group": held_out_group,
        "n_test": len(test_labels),
        "n_faulty": int(sum(test_labels)),
        "cutoff": best_t,
        "accuracy": accuracy_score(test_labels, preds),
        "precision": precision_score(test_labels, preds, pos_label=1, zero_division=0),
        "recall": recall_score(test_labels, preds, pos_label=1, zero_division=0),
        "f1": f1_score(test_labels, preds, pos_label=1, zero_division=0),
    }
    # AUC needs both classes present in the fold's test split, which every
    # "mixed" group has by construction -- but guard anyway.
    try:
        result["auc"] = roc_auc_score(test_labels, test_probs)
    except ValueError:
        result["auc"] = float("nan")
    return result


def run_ensemble_holdout_check(cfg, fold_ckpt_dirs, device):
    """Score the real-world holdout set with an ensemble of every fold model.

    None of the fold models ever trained on this set (it's excluded from
    every fold's train/val/test by ``collect()``), so averaging their
    probabilities here is a legitimate free robustness/accuracy bump.
    """
    broken_root = cfg.get("broken_fan_root")
    if not broken_root:
        return
    root = Path(broken_root)
    if not root.is_dir():
        logger.info("Real-world holdout root not found: %s -- skipping ensemble check.", root)
        return

    models = []
    for d in fold_ckpt_dirs:
        ckpt = d / "best.pt"
        if ckpt.is_file():
            models.append(load_model(cfg["backbone_checkpoint"], ckpt, cfg["num_classes"]).to(device))
    if not models:
        logger.warning("No fold checkpoints available -- skipping ensemble holdout check.")
        return

    audio_suffixes = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    clips = sorted(p for p in root.rglob("*") if p.suffix.lower() in audio_suffixes)
    if not clips:
        return

    print()
    print("=" * 60)
    print(f"ENSEMBLE REAL-WORLD HOLD-OUT CHECK ({len(models)} fold models)")
    print("=" * 60)
    flagged = 0
    print(f"{'file':<62} {'predicted':<9} {'conf':>6}  {'P(Faulty)':>9}")
    for clip in clips:
        pred = predict_ensemble(clip, models, cfg)
        p_faulty = pred.probabilities[cfg["class_names"][1]]
        if pred.label == cfg["class_names"][1]:
            flagged += 1
        print(f"{clip.name[:60]:<62} {pred.label:<9} {pred.confidence:>6.3f}  {p_faulty:>9.3f}")
    print(f"-> ensemble flagged {flagged} of {len(clips)} real broken-fan clips as Faulty")


def cross_validate(cfg: dict, max_folds: int | None = None, force: bool = False) -> None:
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    filepaths, labels = collect(
        cfg["mimii_root"], cfg["extra_normal_root"], cfg.get("sanity_clip_filename")
    )
    groups = derive_group_keys(filepaths)
    folds = build_leave_one_group_out_splits(filepaths, labels, groups, seed=cfg["seed"])
    logger.info("Built %d leave-one-group-out folds: %s",
                len(folds), [g for g, *_ in folds])

    if max_folds is not None:
        folds = folds[:max_folds]
        logger.info("Running only the first %d fold(s) this session.", len(folds))

    checkpoint_dir = Path(cfg["checkpoint_dir"]) / "logo_cv"
    artifact_dir = Path(cfg["artifact_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    results = []
    fold_ckpt_dirs = []
    for held_out, train_split, val_split, test_split in folds:
        fold_ckpt_dir = checkpoint_dir / _safe_name(held_out)
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)
        fold_ckpt_dirs.append(fold_ckpt_dir)

        if force or not (fold_ckpt_dir / "best.pt").is_file():
            logger.info("=== Fold: held out %s (train=%d val=%d test=%d) ===",
                        held_out, len(train_split.filepaths), len(val_split.filepaths),
                        len(test_split.filepaths))
            train_fold(held_out, train_split, val_split, cfg, device, fold_ckpt_dir)
        else:
            logger.info("=== Fold: held out %s -- checkpoint exists, skipping training "
                        "(pass --force to retrain) ===", held_out)

        result = evaluate_fold(held_out, train_split, val_split, test_split, cfg, device, fold_ckpt_dir)
        results.append(result)
        logger.info(
            "  -> %s: acc=%.3f prec=%.3f recall=%.3f f1=%.3f auc=%.3f (n=%d, %d faulty)",
            held_out, result["accuracy"], result["precision"], result["recall"],
            result["f1"], result["auc"], result["n_test"], result["n_faulty"],
        )

    print()
    print("=" * 70)
    print("LEAVE-ONE-GROUP-OUT CROSS-VALIDATION (unseen-machine accuracy)")
    print("=" * 70)
    header = f"{'held-out group':<28} {'n':>5} {'faulty':>6} {'acc':>7} {'prec':>7} {'recall':>7} {'f1':>7} {'auc':>7}"
    print(header)
    for r in results:
        print(f"{r['held_out_group']:<28} {r['n_test']:>5} {r['n_faulty']:>6} "
              f"{r['accuracy']:>7.3f} {r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['f1']:>7.3f} {r['auc']:>7.3f}")

    if results:
        for metric in ("accuracy", "precision", "recall", "f1", "auc"):
            vals = [r[metric] for r in results if not np.isnan(r[metric])]
            if vals:
                print(f"mean {metric:<9}: {np.mean(vals):.4f}  (std {np.std(vals):.4f}, n={len(vals)} folds)")

    (artifact_dir / "logo_cv_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    logger.info("Saved %s", artifact_dir / "logo_cv_results.json")

    run_ensemble_holdout_check(cfg, fold_ckpt_dirs, device)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Leave-one-group-out cross-validation for the Echolyx fan classifier."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--phase1-epochs", type=int, default=None)
    parser.add_argument("--phase2-epochs", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None,
                         help="Run only the first N folds this session (fits Kaggle time limits).")
    parser.add_argument("--force", action="store_true",
                         help="Retrain folds even if a checkpoint already exists for them.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.phase1_epochs is not None:
        cfg["phase1_epochs"] = args.phase1_epochs
    if args.phase2_epochs is not None:
        cfg["phase2_epochs"] = args.phase2_epochs

    cross_validate(cfg, max_folds=args.max_folds, force=args.force)


if __name__ == "__main__":
    main()
