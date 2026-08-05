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

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.data.preprocess import Augment, load_fixed_length_audio


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


def build_splits(
    filepaths: list[Path],
    labels: list[int],
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[Split, Split, Split]:
    """Stratified 70/15/15 train/val/test split (TDD sec 3.3).

    Uses ``sklearn.train_test_split`` twice: first reserve the 15% test
    split, then split the remainder into train/val at ``val_frac`` of the
    total. Both steps are stratified by label and seeded.

    Requires at least 2 samples per class (the stratified splitter needs a
    representative for each class in every split).
    """
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
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Build train (augmented), validation, and test DataLoaders.

    Returns ``(train_loader, val_loader, test_loader, class_weights)`` where
    class weights are computed from the training split only (TDD 3.4).
    """
    train_split, val_split, test_split = build_splits(filepaths, labels, seed=seed)

    train_ds = FanSoundDataset(
        train_split.filepaths, train_split.labels, sample_rate, num_samples, augment=True
    )
    val_ds = FanSoundDataset(
        val_split.filepaths, val_split.labels, sample_rate, num_samples, augment=False
    )
    test_ds = FanSoundDataset(
        test_split.filepaths, test_split.labels, sample_rate, num_samples, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    weights = get_class_weights(train_split.labels)
    return train_loader, val_loader, test_loader, weights