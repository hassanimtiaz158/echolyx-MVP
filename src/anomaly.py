"""Per-machine industrial anomaly detection on MIMII DUE/DG (DCASE-2022 Task 2 framing).

The global two-class run proved at chance (frozen-probe AUC 0.51, all fine-tuned
runs collapsing): AudioSet features cannot learn "cross-machine fan fault" from
MIMII audio. Industrial monitoring does not need that — it monitors ONE machine
whose normal audio it has:

1. Parse MIMII DUE/DG filenames into per-machine clips, keeping the
   domain-shift tokens the challenge data was built around:
   ``section_XX_source_train_normal|anomaly`` (train pool, labeled) and
   ``section_XX_target_test_normal|anomaly`` (test pool, scored).
   (``target_train_normal`` clips are per-machine few-shot normal for the
   source-adaptation recipe.)
2. For each machine: extract frozen PANNs embeddings of the NORMAL clips
   (source train-normal, plus target train-normal when present), fit a
   low-rank Gaussian (PCA + Mahalanobis), and score the machine's target test
   pool by distance. A healthy clip sits in the normal cloud; an anomaly
   (fault, bearing knock, imbalance, ...) falls outside it.
3. Report per-section AUC + mean AUC (the DCASE 2022 metric), per-section
   flag tables, and write artifacts (scores JSON + AUC bar chart).

The model is deliberately per-machine: fit on THIS machine's normal audio,
score THIS machine's clips — no cross-machine assumption. That is what makes
it reliable for industrial use (one sensor, one fan, known healthy baseline).

Usage: ``python -m src.anomaly --config configs/config.yaml``
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe for Kaggle/Colab
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from src.data.collect import _walk
from src.data.preprocess import load_fixed_length_audio
from src.models.transfer_cnn14 import TransferCnn14
from src.train import load_config, set_seed

logger = logging.getLogger(__name__)

# MIMII DUE/DG filename tokens, e.g.
#   section_00_source_train_normal_0000_spd_1.wav
#   section_03_target_test_anomaly_0012_id_00.wav
_MII = re.compile(
    r"section_(?P<section>\d+)_"
    r"(?P<domain>source|target)_"
    r"(?P<split>train|test)_"
    r"(?P<label>normal|anomaly)_"
)

# Original MIMII / DCASE2020 Task 2 filename token, e.g.
#   normal_id_00_00000000.wav / anomaly_id_00_00000000.wav
# Predates the source/target domain-shift concept (one domain only) and the
# train/test split is implied by the parent directory, not the filename.
_DCASE2020 = re.compile(r"^(?P<label>normal|anomaly)_id_(?P<section>\d+)_")

# Raw MIMII 2019 directory layout, e.g. fan/id_00/abnormal/00000000.wav --
# the filename is just a sequence number; label comes from the immediate
# parent folder ("normal"/"abnormal", note: "abnormal" not "anomaly") and the
# machine id from the grandparent folder ("id_00").
_MIMII_RAW_ID_DIR = re.compile(r"^id_(?P<section>\d+)$")


@dataclass(frozen=True)
class Clip:
    """One MIMII clip with the domain-shift tokens kept intact."""

    path: Path
    machine: str        # e.g. "due:fan:section_00" or "dg:fan_dg:section_02"
    domain: str         # "source" | "target"
    split: str          # "train" | "test"
    label: str          # "normal" | "anomaly"

    @property
    def is_faulty(self) -> bool:
        return self.label == "anomaly"


def _archive_of(machine_dir_name: str) -> str:
    """DUE vs DG archive token, mirroring ``derive_group_key`` (dataset.py).

    DUE (dev_fan) and DG (fan) reuse section ids across separate recording
    campaigns, so they must never share a machine id. On the Kaggle mount the
    machine-name directory directly under ``mimii_root`` is ``fan`` (DUE) or
    ``fan_dg`` (DG); a trailing ``_dg`` (or exactly ``dg``) names the DG
    inventory.
    """
    return "dg" if machine_dir_name == "dg" or machine_dir_name.endswith("_dg") else "due"


def parse_mimii_clips(root: str | Path) -> list[Clip]:
    """Walk ``root`` and parse every MIMII DUE/DG/DCASE2020/raw-MIMII wav into a ``Clip``.

    The machine id fuses the inventory token (due/dg/dcase2020), the
    machine-name directory directly under ``root`` (``fan`` / ``fan_dg`` /
    whatever DCASE2020 fan folder is mounted as), and the section/id number —
    so distinct machines never collide. DCASE2020/original-MIMII clips (no
    source/target domain-shift concept) get ``domain="source"``. Three
    filename/layout conventions are recognized: DUE/DG (``section_XX_source_
    train_normal_...``), DCASE2020-flattened (``normal_id_00_...wav``, split
    from the parent dir), and raw MIMII 2019 (``id_00/{normal,abnormal}/
    00000000.wav``, split inferred as train=normal/test=abnormal since that
    layout has no predefined split). Files matching none of these are
    skipped with a warning.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"MIMII root not found: {root}")
    clips: list[Clip] = []
    for wav in _walk(root, {".wav"}):
        m = _MII.search(wav.name)
        if m is not None:
            machine_name = str(wav.relative_to(root).parts[0])
            machine = f"{_archive_of(machine_name)}:{machine_name}:section_{m.group('section')}"
            clips.append(
                Clip(
                    path=wav,
                    machine=machine,
                    domain=m.group("domain"),
                    split=m.group("split"),
                    label=m.group("label"),
                )
            )
            continue
        m2 = _DCASE2020.match(wav.name)
        if m2 is not None:
            parts_lower = {p.lower() for p in wav.parts}
            split = "train" if "train" in parts_lower else "test" if "test" in parts_lower else None
            if split is None:
                logger.warning("DCASE2020 wav with no train/test ancestor dir: %s", wav.name)
                continue
            machine = f"dcase2020:fan:id_{m2.group('section')}"
            clips.append(
                Clip(
                    path=wav,
                    machine=machine,
                    domain="source",
                    split=split,
                    label=m2.group("label"),
                )
            )
            continue
        label_dir = wav.parent.name.lower()
        id_dir_match = _MIMII_RAW_ID_DIR.match(wav.parent.parent.name.lower())
        if id_dir_match is not None and label_dir in ("normal", "abnormal"):
            machine = f"dcase2020:fan:id_{id_dir_match.group('section')}"
            clips.append(
                Clip(
                    path=wav,
                    machine=machine,
                    domain="source",
                    # No predefined split in this layout: normal clips feed
                    # the per-machine training pool, abnormal clips are
                    # scored (machine_test_pool folds them into whatever
                    # test pool that machine already has, e.g. from
                    # dc2020task2 if that's also attached).
                    split="train" if label_dir == "normal" else "test",
                    label="normal" if label_dir == "normal" else "anomaly",
                )
            )
            continue
        logger.warning("Skipping non-MIMII wav: %s", wav.name)
    clips.sort(key=lambda c: str(c.path))
    return clips


def group_by_machine(clips: list[Clip]) -> dict[str, list[Clip]]:
    """Group clips by machine id, preserving per-machine domain streams."""
    by_machine: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        by_machine[clip.machine].append(clip)
    return dict(by_machine)


def machine_train_normal(clips: list[Clip]) -> list[Clip]:
    """Normal-only training pool for one machine.

    source/train-normal (the main pool) plus target/train-normal when the
    DUE dev set provides few-shot target-domain normal clips (DCASE 2022
    few-shot adaptation setup). Anomaly clips are never used for fitting.
    """
    return [
        c
        for c in clips
        if c.label == "normal"
        and c.split == "train"
    ]


def machine_test_pool(clips: list[Clip]) -> list[Clip]:
    """The scored pool for one machine: target/test clips (both classes).

    Falls back to source/test when a machine has no target domain at all —
    original MIMII/DCASE2020 clips predate the domain-shift concept and only
    ever have a single "source" domain, so this keeps them scoreable without
    changing behavior for DUE/DG machines that do have target-domain data.
    """
    target_test = [c for c in clips if c.split == "test" and c.domain == "target"]
    if target_test:
        return target_test
    return [c for c in clips if c.split == "test" and c.domain == "source"]


def _extract_embeddings(
    model: TransferCnn14,
    clip_paths: list[Path],
    device: torch.device,
    sample_rate: int,
    num_samples: int,
) -> np.ndarray:
    """Frozen-backbone 2048-d embeddings for the given clips, in order.

    Each clip is loaded/resampled/padded to the fixed length via
    ``load_fixed_length_audio`` (same path as training/inference), passed
    through the frozen ``Cnn14`` in eval mode, and its 2048-d embedding
    collected. Embeddings accumulate row-major in ``clip_paths`` order so
    per-clip scores line up with the clip list.
    """
    model.eval()
    embs: list[np.ndarray] = []
    with torch.no_grad():
        for path in clip_paths:
            waveform = load_fixed_length_audio(path, sample_rate, num_samples)
            out = model(waveform.unsqueeze(0).to(device))
            embs.append(out["embedding"].squeeze(0).cpu().numpy())
    return np.vstack(embs)


def fit_machine_detector(
    train_embs: np.ndarray, n_components: int = 64
) -> tuple[PCA, np.ndarray]:
    """PCA (``n_components`` dims) + per-dim variance of the normal cloud.

    Returns ``(pca, var)`` so scoring can compute the squared Mahalanobis
    distance of any clip to the normal distribution in PCA space. The
    component count is clamped to ``n_samples - 1`` (a PCA with as many
    components as samples is rank-deficient) and to the embedding width.
    """
    n_samples, n_features = train_embs.shape
    n_components = max(1, min(n_components, n_samples - 1, n_features))
    pca = PCA(n_components=n_components)
    pca.fit(train_embs)
    transformed = pca.transform(train_embs)
    var = transformed.var(axis=0) + 1e-6
    return pca, var


def score_clips(
    pca: PCA, var: np.ndarray, embs: np.ndarray
) -> np.ndarray:
    """Squared Mahalanobis distance of each embedding to the normal cloud."""
    z = pca.transform(embs)
    return np.sum(z * z / var, axis=1)


def _clip_labels(clips: list[Clip]) -> np.ndarray:
    return np.asarray([1 if c.is_faulty else 0 for c in clips], dtype=int)


def fit_gmm_detector(
    pca: PCA, train_embs: np.ndarray, n_components: int = 4
) -> "GaussianMixture":
    """Fit a multi-mode Gaussian mixture on the PCA'd normal cloud.

    MIMII normal audio is inherently multi-modal (RPM / blade harmonics,
    gear whine, airflow) — a single Gaussian cannot capture those several
    healthy modes, so any fault living in a low-density *valley between*
    two normal modes is under-separated. A small GMM fits them jointly.

    ``n_components`` is clamped to the number of available (PCA-reduced)
    samples. Returns the fitted model; ``score_gmm`` scores clips on it.
    """
    from sklearn.mixture import GaussianMixture

    n_samples = len(train_embs)
    n_components = max(1, min(n_components, n_samples))
    gm = GaussianMixture(
        n_components=n_components, covariance_type="full",
        random_state=0, reg_covar=1e-4,
    )
    gm.fit(pca.transform(train_embs))
    return gm


def score_gmm(gm: "GaussianMixture", pca: PCA, embs: np.ndarray) -> np.ndarray:
    """Negative log-likelihood of each clip under the normal GMM (distance)."""
    return -gm.score_samples(pca.transform(embs))


def run_anomaly(config_path: str | Path) -> dict:
    """Per-machine anomaly detection; returns per-machine results dict."""
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | config: %s", device, config_path)

    clips = parse_mimii_clips(cfg["mimii_root"])
    by_machine = group_by_machine(clips)
    logger.info("Parsed %d clips across %d machines", len(clips), len(by_machine))

    model = TransferCnn14(
        cfg["backbone_checkpoint"],
        num_classes=cfg["num_classes"],
        freeze_base=True,
    ).to(device)
    gmm_components = int(cfg.get("anomaly_gmm_components", 4))
    logger.info("GMM density with %d components", gmm_components)

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

        train_embs = _extract_embeddings(
            model, [c.path for c in train_norm], device,
            cfg["sample_rate"], cfg["sample_rate"] * cfg["clip_seconds"],
        )
        pca, var = fit_machine_detector(train_embs)
        gm = fit_gmm_detector(pca, train_embs, n_components=gmm_components)
        test_embs = _extract_embeddings(
            model, [c.path for c in test_pool], device,
            cfg["sample_rate"], cfg["sample_rate"] * cfg["clip_seconds"],
        )
        scores_mahal = score_clips(pca, var, test_embs)
        scores_gmm = score_gmm(gm, pca, test_embs)
        labels = _clip_labels(test_pool)

        auc_mahal = roc_auc_score(labels, scores_mahal) if len(set(labels)) > 1 else float("nan")
        auc_gmm = roc_auc_score(labels, scores_gmm) if len(set(labels)) > 1 else float("nan")
        results[machine] = {
            "machine": machine,
            "n_train_normal": len(train_norm),
            "n_test_pool": len(test_pool),
            "n_faulty": int(labels.sum()),
            "auc_mahalanobis": float(auc_mahal),
            "auc_gmm": float(auc_gmm),
            "scores_mahalanobis": scores_mahal.tolist(),
            "scores_gmm": scores_gmm.tolist(),
            "labels": labels.tolist(),
            "files": [str(c.path.name) for c in test_pool],
        }

    rows = [r for r in results.values() if "auc_gmm" in r]
    aucs = [r["auc_mahalanobis"] for r in rows if not np.isnan(r["auc_mahalanobis"])]
    gmm_aucs = [r["auc_gmm"] for r in rows if not np.isnan(r["auc_gmm"])]
    results["_summary"] = {
        "n_machines": len(aucs),
        "mean_auc_mahalanobis": float(np.mean(aucs)) if aucs else None,
        "mean_auc_gmm": float(np.mean(gmm_aucs)) if gmm_aucs else None,
    }
    return results


def _print_table(results: dict) -> None:
    print("=" * 60)
    print("PER-MACHINE ANOMALY DETECTION (Mahalanobis vs GMM, frozen embeddings)")
    print("=" * 60)
    header = f"{'machine':<28} {'trainN':>6} {'faulty':>6} {'MahalAUC':>9} {'GMMAUC':>8}"
    print(header)
    for key, r in sorted(results.items()):
        if key == "_summary":
            continue
        mahal = "  n/a " if np.isnan(r["auc_mahalanobis"]) else f"{r['auc_mahalanobis']:.4f}"
        gmm = "  n/a " if np.isnan(r["auc_gmm"]) else f"{r['auc_gmm']:.4f}"
        print(
            f"{r['machine']:<28} {r['n_train_normal']:>6} {r['n_faulty']:>6} "
            f"{mahal:>9} {gmm:>8}"
        )
    s = results.get("_summary", {})
    print("-" * 60)
    if s.get("mean_auc_mahalanobis") is not None:
        print(f"mean AUC over {s['n_machines']} machines:")
        print(f"  Mahalanobis: {s['mean_auc_mahalanobis']:.4f}   GMM: {s['mean_auc_gmm']:.4f}")
        print("(DCASE-2022 metric; 0.5 = chance, 1.0 = perfect separation)")
    print()
    print("Reading: each machine's model is fit on ITS OWN normal audio only.")
    print("GMM wins => multi-modal healthy sound; expect fault valleys filled by")
    print("a 4-component fit. Both near 0.5 => frozen AudioSet features are too")
    print("coarse; next is per-machine normal-only fine-tune of the backbone.")


def save_auc_plot(results: dict, path: Path) -> None:
    """Bar chart of per-machine AUC, Mahalanobis vs GMM (NaN rows omitted)."""
    rows = [r for k, r in sorted(results.items())
            if k != "_summary" and not np.isnan(r["auc_mahalanobis"])]
    if not rows:
        logger.warning("No AUCs to plot — skipping %s", path)
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    names = [r["machine"] for r in rows]
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, [r["auc_mahalanobis"] for r in rows], w,
           label="Mahalanobis", color="#4C72B0")
    ax.bar(x + w / 2, [r["auc_gmm"] for r in rows], w,
           label="GMM", color="#DD8452")
    ax.axhline(0.5, color="red", linestyle="--", label="chance")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC")
    ax.set_title("Per-machine anomaly AUC (frozen PANNs embeddings)")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Per-machine industrial anomaly detection.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()

    results = run_anomaly(args.config)
    _print_table(results)

    artifact_dir = Path(load_config(args.config)["artifact_dir"])
    (artifact_dir / "anomaly_scores.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    save_auc_plot(results, artifact_dir / "anomaly_auc.png")
    logger.info("Artifacts: %s, %s",
                artifact_dir / "anomaly_scores.json", artifact_dir / "anomaly_auc.png")


if __name__ == "__main__":
    main()
