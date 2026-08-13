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
    source/target domain-shift concept) — either in the filename
    (``normal_id_00_...wav``) or as a path component (the raw MIMII layout,
    ``fan/id_00/{normal,abnormal}/00000000.wav``) — and are grouped by that
    id, tagged ``dcase2020`` regardless of which of the two layouts it came
    from. That's deliberate: both are widely believed to repackage the same
    underlying 2019 MIMII fan recordings, so sharing one group per id_XX
    means a clip that's a byte-identical duplicate across two different
    downloads can never straddle train/test even if it is. Non-MIMII clips
    (Freesound) are one group each (no known session).
    """
    s = str(path).replace("\\", "/")
    m = re.search(r"section_\d+_(?:source|target)", s)
    if m is not None:
        archive = "dg" if re.search(r"(fan_dg|/dg/|_dg)", s) else "due"
        return f"{archive}:{m.group(0)}"
    m2 = re.search(r"^(?:normal|anomaly)_(id_\d+)_", path.name)
    if m2 is not None:
        return f"dcase2020:{m2.group(1)}"
    m3 = re.search(r"/(id_\d+)/", s)
    if m3 is not None:
        return f"dcase2020:{m3.group(1)}"
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


def build_leave_one_group_out_splits(
    filepaths: list[Path],
    labels: list[int],
    groups: list[str],
    seed: int = 42,
    val_frac: float = 0.15,
) -> list[tuple[str, Split, Split, Split]]:
    """Leave-one-group-out folds over every group that contains a Faulty clip.

    A single grouped train/val/test split (``build_splits``) puts the honest
    unseen-machine test accuracy at the mercy of which 1-2 "mixed" groups the
    random split happened to draw into test -- with only a handful of Faulty
    machine groups, that single draw is noisy. This instead builds one fold
    per mixed group: that whole group is held out as the fold's test split
    (never seen in training), and train/val are built from every remaining
    group using the same stratified grouped logic as ``build_splits``.
    Averaging metrics across folds gives a low-variance unseen-machine
    accuracy estimate, and the per-fold models can be ensembled on data none
    of them trained on (e.g. the real-world holdout set).

    Returns a list of ``(held_out_group_name, train, val, test)`` tuples, one
    per mixed group, in sorted group-name order (reproducible fold order).
    """
    if len(filepaths) != len(labels) or len(filepaths) != len(groups):
        raise ValueError("filepaths, labels and groups must have equal length")

    by_group: dict[str, list[tuple[Path, int]]] = {}
    for fp, lbl, g in zip(filepaths, labels, groups):
        by_group.setdefault(g, []).append((fp, lbl))

    mixed_groups = sorted(
        g for g, items in by_group.items() if len({lbl for _, lbl in items}) > 1
    )
    if not mixed_groups:
        raise ValueError(
            "no group contains both classes -- cannot build leave-one-group-out folds"
        )

    folds: list[tuple[str, Split, Split, Split]] = []
    for held_out in mixed_groups:
        test_files = [fp for fp, _ in by_group[held_out]]
        test_labels = [lbl for _, lbl in by_group[held_out]]

        remaining = [g for g in by_group if g != held_out]
        strat = [
            "mixed"
            if len({lbl for _, lbl in by_group[g]}) > 1
            else f"single-{next(lbl for _, lbl in by_group[g])}"
            for g in remaining
        ]
        n_strat = len(set(strat))
        val_count = max(int(round(val_frac * len(remaining))), n_strat)
        if val_count >= len(remaining):
            raise ValueError(
                f"val_frac={val_frac} too large for {len(remaining)} remaining groups "
                f"(held out {held_out!r})"
            )

        idx = list(range(len(remaining)))
        strat_counts = Counter(strat)
        # sklearn's stratify requires >=2 groups per stratum; with few mixed
        # groups a fold can leave only 1 remaining (e.g. 2 mixed groups total,
        # one held out). Fall back to an unstratified split rather than crash
        # -- val composition is less controlled in that rare case, but the
        # fold still runs.
        can_stratify = min(strat_counts.values()) >= 2
        train_idx, val_idx = train_test_split(
            idx, test_size=val_count,
            stratify=strat if can_stratify else None,
            random_state=seed, shuffle=True,
        )

        def _materialize(group_idx: list[int]) -> Split:
            files: list[Path] = []
            lbls: list[int] = []
            for gi in group_idx:
                for fp, lbl in by_group[remaining[gi]]:
                    files.append(fp)
                    lbls.append(lbl)
            return Split(files, lbls)

        folds.append(
            (held_out, _materialize(train_idx), _materialize(val_idx), Split(test_files, test_labels))
        )

    return folds


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


def get_sampling_weights(labels: list[int], power: float = 1.0) -> torch.Tensor:
    """Per-sample ``WeightedRandomSampler`` weights, softened by ``power``.

    ``power=1.0`` is exact inverse-class-frequency weighting: every class is
    drawn with equal probability per batch (full 50/50 for binary) -- this
    was the sole ``balance_sampling`` behavior and it over-corrects once the
    real class ratio isn't extreme anymore: the model trains on an
    artificial 50/50 prior, then has to be recalibrated back down to the
    true (much more Normal-heavy) test-time prior, and in practice that
    recalibration doesn't fully happen (eval logs show argmax predicting
    Faulty on ~83% of a test split that's only ~20% actually Faulty).

    ``power=0.0`` gives every sample equal weight -- natural class
    frequency, the sampler has no rebalancing effect. Values in between
    interpolate: the minority class is still oversampled (so it isn't
    swamped and forgotten), just not all the way to artificial parity.
    """
    if not labels:
        return torch.empty(0, dtype=torch.float64)
    counts = Counter(labels)
    num_classes = max(labels) + 1
    total = len(labels)
    class_weight = torch.zeros(num_classes, dtype=torch.float64)
    for cls in range(num_classes):
        count = counts.get(cls, 0)
        if count > 0:
            class_weight[cls] = (total / count) ** power
    return torch.tensor([class_weight[l].item() for l in labels], dtype=torch.float64)


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
    sampling_power: float = 1.0,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Build train (augmented), validation, and test DataLoaders.

    Returns ``(train_loader, val_loader, test_loader, class_weights)`` where
    class weights are computed from the training split only (TDD 3.4).
    If ``balance_train``, the train loader draws with replacement using
    ``sampling_power``-softened inverse-class-frequency weights (see
    ``get_sampling_weights``), so the minority (Faulty) class is seen more
    often without necessarily forcing an artificial 50/50 batch prior
    (``sampling_power=1.0`` reproduces that full-parity behavior; lower
    values soften it). If ``groups`` is given, splits are built at the
    machine/source level (no clip from the same recording session straddles
    splits) — see ``build_splits``.
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
        sample_weights = get_sampling_weights(train_split.labels, power=sampling_power)
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