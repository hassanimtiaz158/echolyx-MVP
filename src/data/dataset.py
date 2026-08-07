"""PyTorch Dataset/DataLoader construction (TDD sec 3.2-3.4).

Responsible for:
- ``FanSoundDataset`` wrapping ``(filepath, label)`` pairs, applying
  preprocessing (load/resample/pad) and optional train augmentation.
- ``build_splits``: stratified 70/15/15 train/validation/test split by
  label, fixed seed (TDD sec 3.3).
- ``get_class_weights``: class-weighted CrossEntropy weights
  ``total / (2 * class_count)`` per class (TDD sec 3.4).
- ``make_dataloaders``: DataLoader helper for all three splits.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.preprocess import Augment, load_fixed_length_audio


def derive_group_key(path: Path) -> str:
    """Machine/source group key for one clip path (grouped-split support).

    MIMII DUE/DG clips carry a ``section_XX_source`` / ``section_XX_target``
    token in the filename — the recording session/machine. Clips from the
    same machine must never straddle train/val/test, or accuracy is
    optimistic for unseen machines. DUE and DG archives are kept as separate
    groups (different recording campaigns that happen to reuse section ids).
    Original MIMII / DCASE2020 clips carry an ``id_XX`` token instead (no
    source/target domain-shift concept) and are grouped by that id, tagged
    ``dcase2020`` so they never collide with a DUE/DG section id. Non-MIMII
    clips (Freesound) are one group each (no known session).
    """
    s = str(path).replace("\\", "/")
    m = re.search(r"section_\d+_(?:source|target)", s)
    if m is not None:
        archive = "dg" if re.search(r"(fan_dg|/dg/|_dg)", s) else "due"
        return f"{archive}:{m.group(0)}"
    m2 = re.search(r"^(?:normal|anomaly)_(id_\d+)_", path.name)
    if m2 is not None:
        return f"dcase2020:{m2.group(1)}"
    return f"fs:{path.stem}"


def derive_group_keys(filepaths: list[Path]) -> list[str]:
    """Group key per clip — feed to ``build_splits``/``make_dataloaders``."""
    return [derive_group_key(p) for p in filepaths]


@dataclass
class Split:
    """A labeled subset (train/val/test)."""

    filepaths: list[Path]
    labels: list[int]


class FanSoundDataset(Dataset):
    """Map labeled audio paths to ``(waveform, label)`` tensors."""

    def __init__(
        self,
        filepaths: list[Path],
        labels: list[int],
        sample_rate: int,
        num_samples: int,
        augment: bool = False,
    ) -> None:
        super().__init__()
        if len(filepaths) != len(labels):
            raise ValueError("filepaths and labels must have equal length")
        self.filepaths = [Path(p) for p in filepaths]
        self.labels = list(labels)
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        # Training-only augmentation (TDD sec 3.5).
        self.augment = Augment() if augment else None

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        waveform = load_fixed_length_audio(
            self.filepaths[index], self.sample_rate, self.num_samples
        )
        if self.augment is not None:
            waveform = self.augment(waveform)
        return waveform, self.labels[index]


def _build_grouped_splits(
    filepaths: list[Path],
    labels: list[int],
    groups: list[str],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> tuple[Split, Split, Split]:
    """70/15/15 split at the GROUP level (no group in more than one split).

    Groups are partitioned by stratifying on their label composition
    (``"mixed"`` for groups containing both classes, ``"single-<label>"``
    otherwise), which keeps class balance across splits while guaranteeing
    every split still contains both classes (mixed groups carry both).
    """
    if len(filepaths) != len(labels) or len(filepaths) != len(groups):
        raise ValueError("filepaths, labels and groups must have equal length")

    by_group: dict[str, list[tuple[Path, int]]] = {}
    for fp, lbl, g in zip(filepaths, labels, groups):
        by_group.setdefault(g, []).append((fp, lbl))

    names = sorted(by_group)
    if len(names) < 3:
        raise ValueError(
            "grouped split needs at least 3 distinct groups (machines/sources)"
        )
    strat = [
        "mixed"
        if len({lbl for _, lbl in by_group[g]}) > 1
        else f"single-{next(lbl for _, lbl in by_group[g])}"
        for g in names
    ]

    test_size = 1.0 - train_frac - val_frac
    n_strat = len(set(strat))
    test_count = max(int(round(test_size * len(names))), n_strat)
    val_count = max(int(round(val_frac * len(names))), n_strat)
    if test_count + val_count >= len(names):
        raise ValueError("train_frac + val_frac too small for this group count")

    idx = list(range(len(names)))
    rem_idx, test_idx = train_test_split(
        idx, test_size=test_count, stratify=strat,
        random_state=seed, shuffle=True,
    )
    rem_strat = [strat[i] for i in rem_idx]
    train_idx, val_idx = train_test_split(
        rem_idx, test_size=val_count, stratify=rem_strat,
        random_state=seed + 1, shuffle=True,
    )

    def _materialize(group_idx: list[int]) -> Split:
        files: list[Path] = []
        lbls: list[int] = []
        for gi in group_idx:
            for fp, lbl in by_group[names[gi]]:
                files.append(fp)
                lbls.append(lbl)
        return Split(files, lbls)

    return (
        _materialize(train_idx),
        _materialize(val_idx),
        _materialize(test_idx),
    )


def build_splits(
    filepaths: list[Path],
    labels: list[int],
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    groups: list[str] | None = None,
) -> tuple[Split, Split, Split]:
    """70/15/15 train/val/test split (TDD sec 3.3).

    Per-clip stratified (default, TDD baseline) or, when ``groups`` is given,
    **grouped**: every clip from the same machine/source (``derive_group_key``)
    stays in one split, so test clips never share a recording session with
    training clips — the honest accuracy for unseen machines.

    The per-clip path uses ``sklearn.train_test_split`` twice (test first,
    then train/val), both stratified by label and seeded. Requires at least 2
    samples per class.
    """
    if groups is not None:
        return _build_grouped_splits(
            filepaths, labels, groups, seed, train_frac, val_frac
        )
    if len(filepaths) != len(labels):
        raise ValueError("filepaths and labels must have equal length")
    if not filepaths:
        raise ValueError("cannot split an empty dataset")

    test_size = 1.0 - train_frac - val_frac
    if test_size <= 0:
        raise ValueError("train_frac + val_frac must be < 1")

    n = len(filepaths)
    n_classes = len(set(labels))
    # Integer counts keep the split deterministic. Stratified splitting also
    # needs each split to contain every class, so floor test/val at n_classes.
    test_count = max(int(round(test_size * n)), n_classes)
    val_count = max(int(round(val_frac * n)), n_classes)
    if test_count + val_count >= n or n - test_count - val_count < n_classes:
        raise ValueError("train_frac + val_frac too small for this dataset/class count")

    rem_files, test_files, rem_labels, test_labels = train_test_split(
        list(filepaths),
        list(labels),
        test_size=test_count,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )

    train_files, val_files, train_labels, val_labels = train_test_split(
        rem_files,
        rem_labels,
        test_size=val_count,
        stratify=rem_labels,
        random_state=seed + 1,
        shuffle=True,
    )

    return (
        Split(train_files, train_labels),
        Split(val_files, val_labels),
        Split(test_files, test_labels),
    )


def get_class_weights(labels: list[int]) -> torch.Tensor:
    """Weight per class = total / (2 * class_count) (TDD sec 3.4).

    Returns a float tensor of shape ``(num_classes,)`` where ``num_classes``
    is ``max(labels) + 1``. An absent class receives weight 0 (its loss term
    never fires, avoiding divide-by-zero).
    """
    if not labels:
        return torch.empty(0, dtype=torch.float32)
    counts = Counter(labels)
    num_classes = max(labels) + 1
    total = len(labels)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls in range(num_classes):
        count = counts.get(cls, 0)
        if count > 0:
            weights[cls] = total / (2.0 * count)
    return weights


def make_dataloaders(
    filepaths: list[Path],
    labels: list[int],
    sample_rate: int,
    num_samples: int,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    balance_train: bool = False,
    groups: list[str] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Build train (augmented), validation, and test DataLoaders.

    Returns ``(train_loader, val_loader, test_loader, class_weights)`` where
    class weights are computed from the training split only (TDD 3.4).
    If ``balance_train``, the train loader draws with replacement using
    inverse-class-frequency weights, so the minority (Faulty) class is seen
    every epoch instead of being swamped by Normal (TDD 3.4). If ``groups``
    is given, splits are built at the machine/source level (no clip from the
    same recording session straddles splits) — see ``build_splits``.
    """
    train_split, val_split, test_split = build_splits(
        filepaths, labels, seed=seed, groups=groups
    )

    train_ds = FanSoundDataset(
        train_split.filepaths, train_split.labels, sample_rate, num_samples, augment=True
    )
    val_ds = FanSoundDataset(
        val_split.filepaths, val_split.labels, sample_rate, num_samples, augment=False
    )
    test_ds = FanSoundDataset(
        test_split.filepaths, test_split.labels, sample_rate, num_samples, augment=False
    )

    sampler = None
    if balance_train:
        weights = get_class_weights(train_split.labels)
        sample_weights = torch.tensor(
            [weights[l].item() for l in train_split.labels], dtype=torch.double
        )
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=sampler is None,
        sampler=sampler, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    weights = get_class_weights(train_split.labels)
    return train_loader, val_loader, test_loader, weights