"""Tests for per-machine industrial anomaly detection (src/anomaly.py).

Runs entirely on synthetic audio in ``tmp_path``; no real MIMII data needed.
Covers the DCASE-2022-Task-2 framing pieces:
- filename token parser (section/domain/split/label) -> Clip
- per-machine grouping keeps DUE vs DG archives and sections separate
- train-normal pool vs target-test pool are correctly split
- PCA+Mahalanobis detector separates a synthetic normal cloud from anomalies
- end-to-end ``run_anomaly`` on a tiny synthetic tree (CLI-level smoke)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.anomaly import (
    fit_machine_detector,
    group_by_machine,
    machine_test_pool,
    machine_train_normal,
    parse_mimii_clips,
    run_anomaly,
    score_clips,
)
from src.train import load_config

SR = 32000


def _make_wav(path: Path, base_hz: float, secs: float = 1.0, seed: int = 0) -> Path:
    """Write a synthetic sine-with-noise wav at ``path`` (returns path)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    rng = np.random.default_rng(seed)
    signal = 0.5 * np.sin(2 * np.pi * base_hz * t) + 0.02 * rng.standard_normal(
        t.size
    )
    sf.write(str(path), signal.astype(np.float32), SR)
    return path


def _mimii_name(section: int, domain: str, split: str, label: str, n: int) -> str:
    return f"section_{section:02d}_{domain}_{split}_{label}_{n:04d}_spd_1.wav"


def _factory(tmp_path: Path) -> Path:
    """Build a fake MIMII DUE tree: fan/ (DUE) + fan_dg/ (DG), two sections."""
    root = tmp_path / "mimii"
    for section in range(2):
        for domain, split, label, n in [
            ("source", "train", "normal", 12),
            ("source", "train", "anomaly", 6),
            ("target", "train", "normal", 4),
            ("target", "test", "normal", 4),
            ("target", "test", "anomaly", 4),
        ]:
            for i in range(n):
                _make_wav(
                    root / "fan" / "train"
                    / _mimii_name(section, domain, split, label, i),
                    base_hz=200.0 + 10 * section,
                    seed=section * 100 + i,
                )
    # DG archive reuses section ids -> must be a SEPARATE machine.
    for i in range(4):
        _make_wav(root / "fan_dg" / "id_00" / "normal"
                  / f"section_00_source_train_normal_{i:04d}_spd_1.wav",
                  base_hz=180.0, seed=50 + i)
        _make_wav(root / "fan_dg" / "id_00" / "abnormal"
                  / f"section_00_source_train_anomaly_{i:04d}_spd_1.wav",
                  base_hz=180.0, seed=60 + i)
    return root


def test_parse_mimii_clips_reads_tokens(tmp_path: Path):
    _factory(tmp_path)
    clips = parse_mimii_clips(tmp_path / "mimii")
    assert len(clips) > 0
    by_name = {c.path.name: c for c in clips}
    key = _mimii_name(0, "target", "test", "anomaly", 3)
    clip = by_name[key]
    assert clip.domain == "target"
    assert clip.split == "test"
    assert clip.label == "anomaly"
    assert clip.is_faulty is True
    assert clip.machine == "due:fan:section_00"


def test_machines_isolate_due_dg_and_sections(tmp_path: Path):
    _factory(tmp_path)
    by_machine = group_by_machine(parse_mimii_clips(tmp_path / "mimii"))
    # DUE fan sections 00/01, plus the DG archive's own section_00.
    assert set(by_machine) == {
        "due:fan:section_00",
        "due:fan:section_01",
        "dg:fan_dg:section_00",
    }
    # Every clip in a machine's group keeps the machine's id.
    for machine, clips in by_machine.items():
        assert all(c.machine == machine for c in clips)


def test_train_normal_pool_excludes_anomalies_and_test(tmp_path: Path):
    _factory(tmp_path)
    by_machine = group_by_machine(parse_mimii_clips(tmp_path / "mimii"))
    train_norm = machine_train_normal(by_machine["due:fan:section_00"])
    # source/train-normal (12) + target/train-normal few-shot (4) = 16.
    assert len(train_norm) == 16
    assert all(c.label == "normal" for c in train_norm)
    assert all(c.split == "train" for c in train_norm)

    pool = machine_test_pool(by_machine["due:fan:section_00"])
    # target/test only: 4 normal + 4 anomaly.
    assert len(pool) == 8
    assert all(c.split == "test" and c.domain == "target" for c in pool)
    assert sum(1 for c in pool if c.is_faulty) == 4


def test_mahalanobis_detector_separates_synthetic_clouds():
    rng = np.random.default_rng(3)
    # Normal cloud: tight sphere near origin. Anomalies: far along a few dims.
    normal = rng.normal(0.0, 0.2, size=(400, 128))
    anomaly = rng.normal(3.0, 0.3, size=(40, 128))
    pca, var = fit_machine_detector(normal, n_components=32)
    scores_normal = score_clips(pca, var, normal[:50])
    scores_anomaly = score_clips(pca, var, anomaly)
    assert scores_anomaly.mean() > scores_normal.mean() * 4


def test_gmm_detector_separates_synthetic_clouds():
    from src.anomaly import fit_gmm_detector, score_gmm

    rng = np.random.default_rng(5)
    normal = rng.normal(0.0, 0.2, size=(500, 128))
    anomaly = rng.normal(3.0, 0.3, size=(50, 128))
    pca, _ = fit_machine_detector(normal, n_components=32)
    gm = fit_gmm_detector(pca, normal, n_components=3)
    scores_normal = score_gmm(gm, pca, normal[:60])
    scores_anomaly = score_gmm(gm, pca, anomaly)
    # Negative log-likelihood: anomalies sit in low-density tails.
    assert scores_anomaly.mean() > scores_normal.mean() + 3.0
    assert scores_anomaly.min() > scores_normal.max()


def test_run_anomaly_end_to_end(tmp_path: Path):
    pytest.importorskip(
        "torchlibrosa",
        reason="torchlibrosa required to build the vendored Cnn14",
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
    )

    root = _factory(tmp_path)
    # Fabricate a random-weight backbone (no 300 MB download, same trick as
    # test_inference.py) so run_anomaly can build the frozen model.
    backbone = Cnn14(
        sample_rate=CNN14_SAMPLE_RATE,
        window_size=CNN14_WINDOW_SIZE,
        hop_size=CNN14_HOP_SIZE,
        mel_bins=CNN14_MEL_BINS,
        fmin=CNN14_FMIN,
        fmax=CNN14_FMAX,
        classes_num=AUDIOSET_CLASSES,
    )
    backbone_path = tmp_path / "fake_backbone.pth"
    torch.save(backbone.state_dict(), backbone_path)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "sample_rate: 32000\nclip_seconds: 1\nnum_classes: 2\nclass_names: [Normal, Faulty]\n"
        f"mimii_root: '{root}'\n"
        f"backbone_checkpoint: '{backbone_path}'\n"
        "batch_size: 4\nnum_workers: 0\nseed: 42\ncheckpoint_dir: 'checkpoints'\n"
        "artifact_dir: 'artifacts'\ngroup_by_source: false\n",
        encoding="utf-8",
    )
    results = run_anomaly(cfg_path)
    assert "_summary" in results
    # Both DUE fan sections have train-normal + target-test pools -> scored.
    assert "due:fan:section_00" in results
    assert "due:fan:section_01" in results
    # DG has no target/test pool in the factory -> skipped (not in results).
    assert "dg:fan_dg:section_00" not in results
    assert results["_summary"]["n_machines"] >= 2
    section = results["due:fan:section_00"]
    # Both density models produce scores + AUCs (fields renamed after GMM).
    assert len(section["scores_mahalanobis"]) == len(section["labels"])
    assert len(section["scores_gmm"]) == len(section["labels"])
    assert "auc_mahalanobis" in section and "auc_gmm" in section
    assert results["_summary"]["mean_auc_mahalanobis"] is not None
    assert results["_summary"]["mean_auc_gmm"] is not None


def test_machine_id_archives_mirror_derive_group_key():
    # Same directory-name convention as dataset.derive_group_key's dg-detection.
    from src.anomaly import _archive_of

    assert _archive_of("fan") == "due"
    assert _archive_of("fan_dg") == "dg"
    assert _archive_of("dg") == "dg"
