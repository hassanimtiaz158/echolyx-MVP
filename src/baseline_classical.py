"""Classical ML baseline: hand-crafted audio features + tree ensembles
(TDD sec 2, "classical baseline (optional)").

Why this exists: the PANNs CNN14 transfer-learning pipeline (``src/train.py``)
has plateaued around 65-72% train accuracy regardless of data volume or
epoch budget -- a sign the fine-tuning setup itself (frozen 80M-parameter
AudioSet backbone, tiny differential learning rates, partial unfreezing) may
not be the right fit for this narrow mechanical-fault domain, not purely a
data problem. This trains fast (minutes, not hours) classical models on
interpretable spectral features -- MFCCs, spectral centroid/bandwidth/
rolloff, zero-crossing rate, RMS energy, log-mel band energies -- evaluated
on the exact same honest grouped/unseen-machine split as ``evaluate.py``, so
it's a fair, fast test of whether the accuracy ceiling is about the model
architecture or the underlying task/data difficulty.

Usage: ``python -m src.baseline_classical --config configs/config.yaml``
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.collect import collect
from src.data.dataset import build_splits, derive_group_keys
from src.data.preprocess import load_fixed_length_audio
from src.train import load_config

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False

N_MFCC = 20
N_MELS = 64

FEATURE_NAMES = (
    [f"mfcc{i}_mean" for i in range(N_MFCC)] + [f"mfcc{i}_std" for i in range(N_MFCC)]
    + [f"mfcc_delta{i}_mean" for i in range(N_MFCC)] + [f"mfcc_delta{i}_std" for i in range(N_MFCC)]
    + [
        "centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std",
        "rolloff_mean", "rolloff_std", "zcr_mean", "zcr_std", "rms_mean", "rms_std",
    ]
    + [f"logmel{i}_mean" for i in range(N_MELS)]
)


def extract_features(waveform: np.ndarray, sr: int) -> np.ndarray:
    """One fixed-length clip -> a fixed-length vector of spectral features."""
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(waveform)
    rms = librosa.feature.rms(y=waveform)
    log_mel = librosa.power_to_db(librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=N_MELS))

    return np.concatenate(
        [
            mfcc.mean(axis=1), mfcc.std(axis=1),
            delta.mean(axis=1), delta.std(axis=1),
            [
                centroid.mean(), centroid.std(),
                bandwidth.mean(), bandwidth.std(),
                rolloff.mean(), rolloff.std(),
                zcr.mean(), zcr.std(),
                rms.mean(), rms.std(),
            ],
            log_mel.mean(axis=1),
        ]
    ).astype(np.float32)


def build_feature_matrix(
    filepaths: list[Path], sr: int, num_samples: int, cache_path: Path | None = None
) -> np.ndarray:
    """Extract features for every filepath, cached on disk by exact manifest.

    The cache is keyed by the ordered list of file paths -- if the manifest
    changes (different data, different split), the cache is invalidated and
    recomputed rather than silently reused, since feature extraction over
    thousands of 10s clips is by far the slowest step here.
    """
    paths_str = [str(p) for p in filepaths]
    if cache_path is not None and cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=True)
        if list(cached["paths"]) == paths_str:
            logger.info("Loaded cached features from %s (%d clips)", cache_path, len(paths_str))
            return cached["features"]
        logger.info("Feature cache at %s is stale (manifest changed) -- recomputing.", cache_path)

    features = np.empty((len(filepaths), len(FEATURE_NAMES)), dtype=np.float32)
    for i, path in enumerate(filepaths):
        waveform = load_fixed_length_audio(path, sr, num_samples).numpy()
        features[i] = extract_features(waveform, sr)
        if (i + 1) % 500 == 0:
            logger.info("  extracted features for %d/%d clips", i + 1, len(filepaths))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, features=features, paths=np.array(paths_str, dtype=object))
        logger.info("Cached features to %s", cache_path)
    return features


def _metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float):
    preds = (probs >= threshold).astype(int)
    return (
        accuracy_score(labels, preds),
        precision_score(labels, preds, pos_label=1, zero_division=0),
        recall_score(labels, preds, pos_label=1, zero_division=0),
        f1_score(labels, preds, pos_label=1, zero_division=0),
    )


def _report(
    name: str,
    probs_val: np.ndarray, labels_val: np.ndarray,
    probs_test: np.ndarray, labels_test: np.ndarray,
    class_names: list[str],
) -> dict:
    """Print argmax + validation-tuned-threshold metrics (mirrors evaluate.py)."""
    print()
    print("=" * 60)
    print(f"{name} -- TEST SPLIT METRICS (Faulty = positive class, cutoff=0.5)")
    print("=" * 60)
    acc0, prec0, rec0, f10 = _metrics_at_threshold(probs_test, labels_test, 0.5)
    preds_05 = (probs_test >= 0.5).astype(int)
    print(f"accuracy : {acc0:.4f}")
    print(f"precision: {prec0:.4f}")
    print(f"recall   : {rec0:.4f}   <- priority metric (missed faults are costly)")
    print(f"F1       : {f10:.4f}")
    try:
        print(f"AUC      : {roc_auc_score(labels_test, probs_test):.4f}")
    except ValueError:
        pass
    print(classification_report(labels_test, preds_05, target_names=class_names, digits=3, zero_division=0))

    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        _, _, _, f1v = _metrics_at_threshold(probs_val, labels_val, t)
        if f1v > best_f1:
            best_f1, best_t = f1v, float(t)

    acc_t, prec_t, rec_t, f1_t = _metrics_at_threshold(probs_test, labels_test, best_t)
    preds = (probs_test >= best_t).astype(int)
    cm = confusion_matrix(labels_test, preds, labels=[0, 1])
    print(f"\nTHRESHOLD-TUNED (cutoff={best_t:.2f}, chosen on validation F1={best_f1:.3f})")
    print(f"accuracy : {acc_t:.4f}  precision: {prec_t:.4f}  recall: {rec_t:.4f}  F1: {f1_t:.4f}")
    print("confusion matrix (rows=actual, cols=predicted, order %s):" % " / ".join(class_names))
    for row, cname in zip(cm.tolist(), class_names):
        print(f"  {cname:<8} {row}")

    return {
        "argmax": {"accuracy": acc0, "precision": prec0, "recall": rec0, "f1": f10},
        "threshold_tuned": {"cutoff": best_t, "accuracy": acc_t, "precision": prec_t, "recall": rec_t, "f1": f1_t},
    }


def run_baseline(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    filepaths, labels = collect(
        cfg["mimii_root"], cfg["extra_normal_root"], cfg.get("sanity_clip_filename")
    )
    logger.info(
        "Manifest: %d clips (%d normal / %d faulty)",
        len(filepaths), labels.count(0), labels.count(1),
    )

    groups = derive_group_keys(filepaths) if cfg.get("group_by_source", False) else None
    train_split, val_split, test_split = build_splits(filepaths, labels, seed=cfg["seed"], groups=groups)
    if groups is not None:
        logger.info(
            "Grouped split (unseen-machine test): train=%d val=%d test=%d",
            len(train_split.filepaths), len(val_split.filepaths), len(test_split.filepaths),
        )

    num_samples = cfg["sample_rate"] * cfg["clip_seconds"]
    artifact_dir = Path(cfg["artifact_dir"])
    cache_dir = artifact_dir / "classical_feature_cache"

    logger.info("Extracting features (cached to %s)...", cache_dir)
    X_train = build_feature_matrix(train_split.filepaths, cfg["sample_rate"], num_samples, cache_dir / "train.npz")
    X_val = build_feature_matrix(val_split.filepaths, cfg["sample_rate"], num_samples, cache_dir / "val.npz")
    X_test = build_feature_matrix(test_split.filepaths, cfg["sample_rate"], num_samples, cache_dir / "test.npz")
    y_train = np.array(train_split.labels)
    y_val = np.array(val_split.labels)
    y_test = np.array(test_split.labels)

    results: dict = {}

    logger.info("Training Random Forest (%d train clips, %d features)...", len(y_train), X_train.shape[1])
    rf = RandomForestClassifier(
        n_estimators=400, class_weight="balanced_subsample", n_jobs=-1, random_state=cfg["seed"],
    )
    rf.fit(X_train, y_train)
    results["random_forest"] = _report(
        "RANDOM FOREST",
        rf.predict_proba(X_val)[:, 1], y_val,
        rf.predict_proba(X_test)[:, 1], y_test,
        cfg["class_names"],
    )

    top_idx = np.argsort(rf.feature_importances_)[::-1][:15]
    print("\nTop 15 features (Random Forest importance):")
    for i in top_idx:
        print(f"  {FEATURE_NAMES[i]:<20} {rf.feature_importances_[i]:.4f}")

    if HAVE_XGBOOST:
        neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
        logger.info("Training XGBoost (scale_pos_weight=%.2f)...", neg / max(pos, 1))
        xgb = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            scale_pos_weight=neg / max(pos, 1), eval_metric="logloss",
            random_state=cfg["seed"], n_jobs=-1,
        )
        xgb.fit(X_train, y_train)
        results["xgboost"] = _report(
            "XGBOOST",
            xgb.predict_proba(X_val)[:, 1], y_val,
            xgb.predict_proba(X_test)[:, 1], y_test,
            cfg["class_names"],
        )
    else:
        logger.info("xgboost not installed -- skipping (pip install xgboost to include it).")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "classical_baseline_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Saved %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Classical ML baseline for the Echolyx fan classifier.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()
    run_baseline(args.config)


if __name__ == "__main__":
    main()
