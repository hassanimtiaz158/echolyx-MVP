"""Smoke tests for the data pipeline (TDD sec 3.2-3.5).

These run entirely on synthetic audio generated in ``tmp_path``, so no real
dataset is needed. Coverage:
- preprocess: ``load_fixed_length_audio`` pad/trim to exact length, mono.
- dataset: ``build_splits`` stratified 70/15/15, ``get_class_weights``
  (total/(2*count)), ``FanSoundDataset`` + ``make_dataloaders`` shapes/labels,
  grouped (machine/source) splits with no leakage, and ``derive_group_key``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.data.dataset import (
    FanSoundDataset,
    build_splits,
    derive_group_key,
    derive_group_keys,
    get_class_weights,
    make_dataloaders,
)
from src.data.preprocess import load_fixed_length_audio

SR = 32000
CLIP_SAMPLES = SR * 10  # 10 s


@pytest.fixture(scope="module")
def wav_dir(tmp_path_factory):
    """Write a small balanced corpus of synthetic wav files: N=8, 4 normal + 4 faulty."""
    base = tmp_path_factory.mktemp("fan_audio")
    files: list = []
    labels: list[int] = []
    for i in range(8):
        path = base / f"clip_{i}.wav"
        # 1-second random noise; short enough to trigger zero-padding later.
        sf.write(path, np.random.default_rng(i).random(1024, dtype=np.float32), 8000)
        files.append(path)
        labels.append(0 if i < 4 else 1)
    return files, labels


def test_load_fixed_length_pads_short(wav_dir):
    files, _ = wav_dir
    wave = load_fixed_length_audio(files[0], sr=SR, target_len=CLIP_SAMPLES)
    assert wave.shape == (CLIP_SAMPLES,)
    assert wave.dtype == torch.float32


def test_load_fixed_length_trims_long(tmp_path):
    long_path = tmp_path / "long.wav"
    sf.write(long_path, np.random.default_rng(1).random(SR * 2, dtype=np.float32), SR)
    wave = load_fixed_length_audio(long_path, sr=SR, target_len=CLIP_SAMPLES)
    assert wave.shape == (CLIP_SAMPLES,)


def test_load_fixed_length_mono(tmp_path):
    stereo_path = tmp_path / "stereo.wav"
    stereo = np.stack(
        [np.random.default_rng(2).random(SR, dtype=np.float32),
         np.random.default_rng(3).random(SR, dtype=np.float32)],
        axis=1,
    )
    sf.write(stereo_path, stereo, SR)
    wave = load_fixed_length_audio(stereo_path, sr=SR, target_len=CLIP_SAMPLES)
    assert wave.shape == (CLIP_SAMPLES,)


def test_get_class_weights():
    labels = [0, 0, 0, 1]
    weights = get_class_weights(labels)
    # total = 4 -> w0 = 4/(2*3) = 2/3, w1 = 4/(2*1) = 2
    assert torch.allclose(weights, torch.tensor([2.0 / 3.0, 2.0]))


def test_build_splits_stratified():
    # 40 files, 20 per class -> exact 70/15/15 = 28/6/6 (no rounding drift).
    files = [Path(f"fake/{i}.wav") for i in range(40)]
    labels = [0] * 20 + [1] * 20
    train, val, test = build_splits(files, labels, seed=42)
    assert len(train.filepaths) + len(val.filepaths) + len(test.filepaths) == 40
    assert len(train.filepaths) == 28
    assert len(val.filepaths) == 6
    assert len(test.filepaths) == 6
    # Stratified: every split keeps the 0/1 mix (14/14, 3/3, 3/3).
    assert train.labels.count(0) == 14 and train.labels.count(1) == 14
    assert val.labels.count(0) == 3 and val.labels.count(1) == 3
    assert test.labels.count(0) == 3 and test.labels.count(1) == 3
    # No overlaps between splits.
    train_set = set(train.filepaths)
    assert train_set.isdisjoint(val.filepaths)
    assert train_set.isdisjoint(test.filepaths)


def test_derive_group_key():
    # MIMII DUE vs DG share section ids -> different groups (separate archives).
    due = Path("data/raw/mimii/fan/section_00_source_train_normal_0000_spd_1.wav")
    dg = Path("data/raw/mimii/fan_dg/section_00_source_train_anomaly_0000.wav")
    assert derive_group_key(due) == "due:section_00_source"
    assert derive_group_key(dg) == "dg:section_00_source"
    assert derive_group_key(due) != derive_group_key(dg)
    # Same machine, other pool/domain tokens -> same group.
    other = Path("data/raw/mimii/fan/section_00_target_test_anomaly_0007_spd_3.wav")
    assert derive_group_key(other) == "due:section_00_target"
    # Freesound clips are one group each.
    fs = Path("data/raw/freesound/desk_fan.wav")
    assert derive_group_key(fs) == "fs:desk_fan"


def test_build_splits_grouped_no_machine_leakage():
    # 4 MIMII machines (2 normal + 2 faulty clips each) + 3 Freesound clips:
    # every clip of a machine must stay in ONE split.
    files: list[Path] = []
    labels: list[int] = []
    for sec in (0, 1, 2, 3):
        for kind, lbl in (("normal", 0), ("anomaly", 1)):
            files.append(Path(f"data/raw/mimii/fan/section_0{sec}_source_train_{kind}_0000_spd_1.wav"))
            labels.append(lbl)
            files.append(Path(f"data/raw/mimii/fan/section_0{sec}_source_train_{kind}_0001_spd_1.wav"))
            labels.append(lbl)
    for i in range(3):
        files.append(Path(f"data/raw/freesound/extra_{i}.wav"))
        labels.append(0)

    groups = derive_group_keys(files)
    train, val, test = build_splits(files, labels, seed=42, groups=groups)

    assert len(train.filepaths) + len(val.filepaths) + len(test.filepaths) == len(files)
    # No group straddles two splits.
    def group_set(split):
        return {derive_group_key(p) for p in split.filepaths}
    assert group_set(train).isdisjoint(group_set(val))
    assert group_set(train).isdisjoint(group_set(test))
    assert group_set(val).isdisjoint(group_set(test))
    # Both classes present in every split (mixed groups carry both).
    assert set(train.labels) == {0, 1}
    assert set(val.labels) == {0, 1}
    assert set(test.labels) == {0, 1}


def test_make_dataloaders_and_dataset(wav_dir):
    files, labels = wav_dir
    train_loader, val_loader, test_loader, weights = make_dataloaders(
        files, labels, sample_rate=8000, num_samples=1024, batch_size=2, seed=42
    )

    assert weights.shape == (2,)
    assert weights.sum() > 0

    wave, label = next(iter(train_loader))
    assert wave.shape == (2, 1024)  # (batch, num_samples)
    assert wave.dtype == torch.float32
    assert set(label.tolist()).issubset({0, 1})

    # Non-training splits load the same shapes without augmentation.
    wave_v, _ = next(iter(val_loader))
    assert wave_v.shape == (2, 1024)