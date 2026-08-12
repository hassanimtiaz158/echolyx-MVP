"""Per-machine log-mel autoencoder anomaly detection (DCASE Task 2 baseline).

Trains one small ``DenseAutoencoder`` per machine, from scratch, on that
machine's own normal-only log-mel context vectors — no pretrained backbone.
Reuses MIMII clip parsing/grouping from ``src/anomaly.py`` (``parse_mimii_
clips``, ``group_by_machine``, ``machine_train_normal``, ``machine_test_
pool``) so both detectors are evaluated on identical per-machine train/test
pools — an apples-to-apples AUC comparison against the frozen-embedding
approach.

Usage: ``python -m src.train_autoencoder --config configs/config.yaml``
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
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.anomaly import (
    Clip,
    _clip_labels,
    group_by_machine,
    machine_test_pool,
    machine_train_normal,
    parse_mimii_clips,
)
from src.data.preprocess import load_fixed_length_audio
from src.models.dense_autoencoder import (
    LOGMEL_SAMPLE_RATE,
    DenseAutoencoder,
    LogMelExtractor,
    logmel_to_context_vectors,
)
from src.train import load_config, set_seed

logger = logging.getLogger(__name__)

MAX_TRAIN_VECTORS = 150_000  # cap per machine, memory/time guard
EPOCHS = 20
BATCH_SIZE = 512
LR = 1e-3
CLIP_BATCH = 32  # clips per log-mel extraction batch


def _clip_context_vectors(
    clip_paths: list[Path],
    logmel_extractor: LogMelExtractor,
    num_samples: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Log-mel context vectors for each clip, one tensor per clip (in order)."""
    logmel_extractor.eval()
    out: list[torch.Tensor] = []
    for start in range(0, len(clip_paths), CLIP_BATCH):
        batch_paths = clip_paths[start : start + CLIP_BATCH]
        waveforms = torch.stack(
            [load_fixed_length_audio(p, LOGMEL_SAMPLE_RATE, num_samples) for p in batch_paths]
        ).to(device)
        logmel = logmel_extractor(waveforms)  # (batch, time, mel_bins)
        for i in range(logmel.shape[0]):
            out.append(logmel_to_context_vectors(logmel[i].cpu()))
    return out


def train_machine_autoencoder(
    train_vectors: torch.Tensor, device: torch.device, seed: int
) -> DenseAutoencoder:
    """Fit one autoencoder on a machine's pooled normal context vectors."""
    if train_vectors.shape[0] > MAX_TRAIN_VECTORS:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(train_vectors.shape[0], generator=g)[:MAX_TRAIN_VECTORS]
        train_vectors = train_vectors[idx]

    mean = train_vectors.mean(dim=0, keepdim=True)
    std = train_vectors.std(dim=0, keepdim=True) + 1e-6
    normed = (train_vectors - mean) / std

    model = DenseAutoencoder().to(device)
    model.mean = mean.to(device)  # stashed on the module for scoring-time normalization
    model.std = std.to(device)

    loader = DataLoader(TensorDataset(normed), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            logger.info("  epoch %2d/%d  train_mse=%.5f", epoch, EPOCHS, total_loss / max(n_batches, 1))

    model.eval()
    return model


@torch.no_grad()
def score_clip_vectors(model: DenseAutoencoder, vectors: torch.Tensor, device: torch.device) -> float:
    """Mean per-vector reconstruction MSE for one clip's context vectors."""
    if vectors.shape[0] == 0:
        return float("nan")
    vectors = vectors.to(device)
    normed = (vectors - model.mean) / model.std
    recon = model(normed)
    mse = ((recon - normed) ** 2).mean(dim=1)
    return float(mse.mean().item())


def run(config_path: str | Path) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | config: %s", device, config_path)

    clips = parse_mimii_clips(cfg["mimii_root"])
    by_machine = group_by_machine(clips)
    logger.info("Parsed %d clips across %d machines", len(clips), len(by_machine))

    logmel_extractor = LogMelExtractor().to(device)
    num_samples = LOGMEL_SAMPLE_RATE * cfg["clip_seconds"]

    results: dict[str, dict] = {}
    for machine, machine_clips in sorted(by_machine.items()):
        train_norm = machine_train_normal(machine_clips)
        test_pool = machine_test_pool(machine_clips)
        if not train_norm or not test_pool:
            logger.warning(
                "Skipping %s: %d train-normal clips, %d test-pool clips",
                machine, len(train_norm), len(test_pool),
            )
            continue

        logger.info("Machine %s: %d train-normal clips, %d test clips", machine, len(train_norm), len(test_pool))
        train_vec_list = _clip_context_vectors(
            [c.path for c in train_norm], logmel_extractor, num_samples, device
        )
        train_vectors = torch.cat(train_vec_list, dim=0)
        model = train_machine_autoencoder(train_vectors, device, cfg["seed"])

        test_vec_list = _clip_context_vectors(
            [c.path for c in test_pool], logmel_extractor, num_samples, device
        )
        scores = np.array([score_clip_vectors(model, v, device) for v in test_vec_list])
        labels = _clip_labels(test_pool)

        valid = ~np.isnan(scores)
        auc = (
            roc_auc_score(labels[valid], scores[valid])
            if valid.sum() and len(set(labels[valid])) > 1
            else float("nan")
        )
        results[machine] = {
            "machine": machine,
            "n_train_normal": len(train_norm),
            "n_test_pool": len(test_pool),
            "n_faulty": int(labels.sum()),
            "auc": float(auc),
            "scores": scores.tolist(),
            "labels": labels.tolist(),
            "files": [str(c.path.name) for c in test_pool],
        }

    rows = [r for r in results.values()]
    aucs = [r["auc"] for r in rows if not np.isnan(r["auc"])]
    results["_summary"] = {
        "n_machines": len(aucs),
        "mean_auc": float(np.mean(aucs)) if aucs else None,
    }
    return results


def _print_table(results: dict) -> None:
    print("=" * 60)
    print("PER-MACHINE LOG-MEL AUTOENCODER (reconstruction-error anomaly score)")
    print("=" * 60)
    print(f"{'machine':<28} {'trainN':>6} {'faulty':>6} {'AUC':>8}")
    for key, r in sorted(results.items()):
        if key == "_summary":
            continue
        auc = "  n/a " if np.isnan(r["auc"]) else f"{r['auc']:.4f}"
        print(f"{r['machine']:<28} {r['n_train_normal']:>6} {r['n_faulty']:>6} {auc:>8}")
    s = results.get("_summary", {})
    print("-" * 60)
    if s.get("mean_auc") is not None:
        print(f"mean AUC over {s['n_machines']} machines: {s['mean_auc']:.4f}")
        print("(DCASE metric; 0.5 = chance, 1.0 = perfect separation)")
    print()
    print("Reading: each autoencoder is trained from scratch on ONE machine's")
    print("own normal log-mel spectrograms — no pretrained backbone. Compare")
    print("against artifacts/anomaly_scores.json (frozen-embedding approach).")


def save_auc_plot(results: dict, path: Path) -> None:
    rows = [r for k, r in sorted(results.items()) if k != "_summary" and not np.isnan(r["auc"])]
    if not rows:
        logger.warning("No AUCs to plot — skipping %s", path)
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    names = [r["machine"] for r in rows]
    x = np.arange(len(names))
    ax.bar(x, [r["auc"] for r in rows], color="#55A868")
    ax.axhline(0.5, color="red", linestyle="--", label="chance")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC")
    ax.set_title("Per-machine log-mel autoencoder AUC")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Per-machine log-mel autoencoder anomaly detection.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()

    results = run(args.config)
    _print_table(results)

    artifact_dir = Path(load_config(args.config)["artifact_dir"])
    (artifact_dir / "autoencoder_scores.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    save_auc_plot(results, artifact_dir / "autoencoder_auc.png")
    logger.info(
        "Artifacts: %s, %s",
        artifact_dir / "autoencoder_scores.json",
        artifact_dir / "autoencoder_auc.png",
    )


if __name__ == "__main__":
    main()
