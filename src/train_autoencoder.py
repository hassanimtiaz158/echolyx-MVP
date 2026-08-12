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
from src.models.conv_autoencoder import ConvAutoencoder
from src.models.dense_autoencoder import (
    LOGMEL_SAMPLE_RATE,
    DenseAutoencoder,
    LogMelExtractor,
    logmel_to_context_vectors,
)
from src.train import load_config, set_seed

logger = logging.getLogger(__name__)

MAX_TRAIN_VECTORS = 150_000  # cap per machine, memory/time guard
EPOCHS = 60
BATCH_SIZE = 512
LR = 1e-3
CLIP_BATCH = 32  # clips per log-mel extraction batch
VAL_FRACTION = 0.1
HIDDEN = 256
BOTTLENECK = 20

# Conv model uses whole-clip spectrograms (far fewer samples per machine than
# the dense model's per-window pool), so fewer, larger batches and more
# epochs to reach a comparable number of gradient updates.
CONV_EPOCHS = 80
CONV_BATCH_SIZE = 32


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
    """Fit one autoencoder on a machine's pooled normal context vectors.

    Holds out ``VAL_FRACTION`` of the (capped) pool as a validation slice —
    still normal-only audio, just unseen during weight updates — and keeps
    the best-val-loss epoch's weights rather than whatever epoch training
    happens to end on. Trains for ``EPOCHS`` with cosine LR decay: the
    initial 20-epoch/fixed-LR recipe was still visibly improving at the
    final epoch for most machines (underfit, not overfit), so more capacity
    (``HIDDEN``/``BOTTLENECK``) and more training time are the first levers.
    """
    g = torch.Generator().manual_seed(seed)
    if train_vectors.shape[0] > MAX_TRAIN_VECTORS:
        idx = torch.randperm(train_vectors.shape[0], generator=g)[:MAX_TRAIN_VECTORS]
        train_vectors = train_vectors[idx]

    n = train_vectors.shape[0]
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * VAL_FRACTION))
    val_vectors = train_vectors[perm[:n_val]]
    fit_vectors = train_vectors[perm[n_val:]]

    mean = fit_vectors.mean(dim=0, keepdim=True)
    std = fit_vectors.std(dim=0, keepdim=True) + 1e-6
    fit_normed = (fit_vectors - mean) / std
    val_normed = ((val_vectors - mean) / std).to(device)

    model = DenseAutoencoder(hidden=HIDDEN, bottleneck=BOTTLENECK).to(device)
    model.mean = mean.to(device)  # stashed on the module for scoring-time normalization
    model.std = std.to(device)

    loader = DataLoader(TensorDataset(fit_normed), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
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
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_normed), val_normed).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            logger.info(
                "  epoch %2d/%d  train_mse=%.5f  val_mse=%.5f  best_val=%.5f",
                epoch, EPOCHS, total_loss / max(n_batches, 1), val_loss, best_val,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


AGGREGATIONS = ("mean", "p90", "p95", "max")


@torch.no_grad()
def score_clip_vectors(
    model: DenseAutoencoder, vectors: torch.Tensor, device: torch.device
) -> dict[str, float]:
    """Per-clip anomaly score under several aggregations of per-frame MSE.

    Plain ``mean`` dilutes a short/localized fault (e.g. a brief rattle)
    across a mostly-normal 10s clip. ``p90``/``p95``/``max`` instead ask "how
    bad does this clip's WORST moment look", which should be more sensitive
    to intermittent faults — computed together here so one training run lets
    us compare all four instead of guessing which aggregation to commit to.
    """
    if vectors.shape[0] == 0:
        return {agg: float("nan") for agg in AGGREGATIONS}
    vectors = vectors.to(device)
    normed = (vectors - model.mean) / model.std
    recon = model(normed)
    mse = ((recon - normed) ** 2).mean(dim=1).cpu().numpy()  # per-frame MSE
    return {
        "mean": float(mse.mean()),
        "p90": float(np.percentile(mse, 90)),
        "p95": float(np.percentile(mse, 95)),
        "max": float(mse.max()),
    }


def _clip_spectrograms(
    clip_paths: list[Path],
    logmel_extractor: LogMelExtractor,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Whole-clip log-mel spectrograms, batched: (n_clips, 1, T, mel_bins).

    Unlike ``_clip_context_vectors`` (windowed, for the dense model), this
    keeps each clip as one 2D image — every clip has the same fixed sample
    count, so T is identical across clips and they stack directly.
    """
    logmel_extractor.eval()
    out: list[torch.Tensor] = []
    for start in range(0, len(clip_paths), CLIP_BATCH):
        batch_paths = clip_paths[start : start + CLIP_BATCH]
        waveforms = torch.stack(
            [load_fixed_length_audio(p, LOGMEL_SAMPLE_RATE, num_samples) for p in batch_paths]
        ).to(device)
        logmel = logmel_extractor(waveforms)  # (batch, time, mel_bins)
        out.append(logmel.unsqueeze(1).cpu())  # (batch, 1, time, mel_bins)
    return torch.cat(out, dim=0)


def train_machine_conv_autoencoder(
    train_spectrograms: torch.Tensor, device: torch.device, seed: int
) -> ConvAutoencoder:
    """Fit one ConvAutoencoder on a machine's normal-only spectrograms.

    Same held-out-validation / cosine-LR / best-checkpoint pattern as
    ``train_machine_autoencoder``, just on whole-clip spectrograms instead of
    windowed context vectors.
    """
    g = torch.Generator().manual_seed(seed)
    n = train_spectrograms.shape[0]
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * VAL_FRACTION))
    val_specs = train_spectrograms[perm[:n_val]]
    fit_specs = train_spectrograms[perm[n_val:]]

    mean = fit_specs.mean()
    std = fit_specs.std() + 1e-6
    fit_normed = (fit_specs - mean) / std
    val_normed = ((val_specs - mean) / std).to(device)

    model = ConvAutoencoder().to(device)
    model.mean = mean.to(device)
    model.std = std.to(device)

    loader = DataLoader(TensorDataset(fit_normed), batch_size=CONV_BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONV_EPOCHS)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    for epoch in range(1, CONV_EPOCHS + 1):
        model.train()
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
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_normed), val_normed).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == CONV_EPOCHS:
            logger.info(
                "  epoch %2d/%d  train_mse=%.5f  val_mse=%.5f  best_val=%.5f",
                epoch, CONV_EPOCHS, total_loss / max(n_batches, 1), val_loss, best_val,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def score_clip_spectrogram(
    model: ConvAutoencoder, spectrogram: torch.Tensor, device: torch.device
) -> dict[str, float]:
    """Per-clip anomaly score under several aggregations of per-pixel MSE."""
    x = spectrogram.unsqueeze(0).to(device)  # (1, 1, T, mel_bins)
    normed = (x - model.mean) / model.std
    recon = model(normed)
    mse = ((recon - normed) ** 2).squeeze(0).squeeze(0).cpu().numpy()  # (T, mel_bins)
    flat = mse.reshape(-1)
    return {
        "mean": float(flat.mean()),
        "p90": float(np.percentile(flat, 90)),
        "p95": float(np.percentile(flat, 95)),
        "max": float(flat.max()),
    }


def run(config_path: str | Path, model_type: str = "dense") -> dict:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | config: %s | model: %s", device, config_path, model_type)

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

        if model_type == "conv":
            train_specs = _clip_spectrograms(
                [c.path for c in train_norm], logmel_extractor, num_samples, device
            )
            model = train_machine_conv_autoencoder(train_specs, device, cfg["seed"])
            test_specs = _clip_spectrograms(
                [c.path for c in test_pool], logmel_extractor, num_samples, device
            )
            score_dicts = [score_clip_spectrogram(model, test_specs[i], device) for i in range(test_specs.shape[0])]
        else:
            train_vec_list = _clip_context_vectors(
                [c.path for c in train_norm], logmel_extractor, num_samples, device
            )
            train_vectors = torch.cat(train_vec_list, dim=0)
            model = train_machine_autoencoder(train_vectors, device, cfg["seed"])

            test_vec_list = _clip_context_vectors(
                [c.path for c in test_pool], logmel_extractor, num_samples, device
            )
            score_dicts = [score_clip_vectors(model, v, device) for v in test_vec_list]

        labels = _clip_labels(test_pool)

        per_agg_scores = {agg: np.array([d[agg] for d in score_dicts]) for agg in AGGREGATIONS}
        valid = ~np.isnan(per_agg_scores["mean"])
        per_agg_auc = {}
        for agg in AGGREGATIONS:
            s = per_agg_scores[agg]
            per_agg_auc[agg] = (
                float(roc_auc_score(labels[valid], s[valid]))
                if valid.sum() and len(set(labels[valid])) > 1
                else float("nan")
            )

        results[machine] = {
            "machine": machine,
            "n_train_normal": len(train_norm),
            "n_test_pool": len(test_pool),
            "n_faulty": int(labels.sum()),
            "auc": per_agg_auc,
            "scores": {agg: per_agg_scores[agg].tolist() for agg in AGGREGATIONS},
            "labels": labels.tolist(),
            "files": [str(c.path.name) for c in test_pool],
        }

    rows = list(results.values())
    summary: dict[str, object] = {"n_machines": len(rows)}
    for agg in AGGREGATIONS:
        aucs = [r["auc"][agg] for r in rows if not np.isnan(r["auc"][agg])]
        summary[f"mean_auc_{agg}"] = float(np.mean(aucs)) if aucs else None
    results["_summary"] = summary
    return results


def _print_table(results: dict) -> None:
    print("=" * 60)
    print("PER-MACHINE LOG-MEL AUTOENCODER (reconstruction-error, 4 score aggregations)")
    print("=" * 60)
    header = f"{'machine':<28} {'trainN':>6} {'faulty':>6}"
    for agg in AGGREGATIONS:
        header += f" {agg:>8}"
    print(header)
    for key, r in sorted(results.items()):
        if key == "_summary":
            continue
        row = f"{r['machine']:<28} {r['n_train_normal']:>6} {r['n_faulty']:>6}"
        for agg in AGGREGATIONS:
            v = r["auc"][agg]
            row += f" {'  n/a ' if np.isnan(v) else f'{v:.4f}':>8}"
        print(row)
    s = results.get("_summary", {})
    print("-" * 60)
    best_agg, best_mean = None, -1.0
    for agg in AGGREGATIONS:
        m = s.get(f"mean_auc_{agg}")
        if m is not None:
            print(f"mean AUC ({agg:<4}) over {s['n_machines']} machines: {m:.4f}")
            if m > best_mean:
                best_agg, best_mean = agg, m
    print("(DCASE metric; 0.5 = chance, 1.0 = perfect separation)")
    if best_agg is not None:
        print(f"\n-> best aggregation: {best_agg} (mean AUC {best_mean:.4f})")
    print()
    print("Reading: each autoencoder is trained from scratch on ONE machine's")
    print("own normal log-mel spectrograms — no pretrained backbone. Compare")
    print("against artifacts/anomaly_scores.json (frozen-embedding approach).")


def save_auc_plot(results: dict, path: Path) -> None:
    rows = [r for k, r in sorted(results.items()) if k != "_summary"]
    rows = [r for r in rows if not all(np.isnan(v) for v in r["auc"].values())]
    if not rows:
        logger.warning("No AUCs to plot — skipping %s", path)
        return
    fig, ax = plt.subplots(figsize=(11, 4.5))
    names = [r["machine"] for r in rows]
    x = np.arange(len(names))
    n_agg = len(AGGREGATIONS)
    w = 0.8 / n_agg
    colors = ["#55A868", "#4C72B0", "#C44E52", "#DD8452"]
    for i, agg in enumerate(AGGREGATIONS):
        vals = [r["auc"][agg] for r in rows]
        ax.bar(x + (i - (n_agg - 1) / 2) * w, vals, w, label=agg, color=colors[i % len(colors)])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC")
    ax.set_title("Per-machine log-mel autoencoder AUC by score aggregation")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Per-machine log-mel autoencoder anomaly detection.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument(
        "--model", choices=["dense", "conv"], default="dense",
        help="dense: flattened context-window autoencoder (original). "
        "conv: whole-clip 2D convolutional autoencoder (preserves time-frequency structure).",
    )
    args = parser.parse_args()

    results = run(args.config, model_type=args.model)
    _print_table(results)

    artifact_dir = Path(load_config(args.config)["artifact_dir"])
    suffix = "" if args.model == "dense" else f"_{args.model}"
    scores_path = artifact_dir / f"autoencoder_scores{suffix}.json"
    plot_path = artifact_dir / f"autoencoder_auc{suffix}.png"
    scores_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_auc_plot(results, plot_path)
    logger.info("Artifacts: %s, %s", scores_path, plot_path)


if __name__ == "__main__":
    main()
