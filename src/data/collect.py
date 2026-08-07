"""Build the labeled file list from MIMII + Freesound sources (TDD sec 3.1).

Labels:
- MIMII fan subset (``mimii_root``): **MIMII DUE format** —
  ``fan/{train,test}/section_XX_{source,target}_{train,test}_{normal,anomaly}_NNNN_spd_M.wav``.
  There are no ``id_XX``/``{normal,abnormal}`` folders; the label is read
  from the **filename** via regex: ``_normal_`` -> 0 (Normal),
  ``_anomaly_`` -> 1 (Faulty). DUE's own ``train``/``test`` folders are
  deliberately **re-pooled** — that split exists for the DCASE domain-shift
  challenge task and is not used here (TDD sec 3.1).
- Freesound clips (``extra_normal_root``): every ``.mp3``/``.wav`` file is
  treated as additional ``Normal`` -> 0.

HELD-OUT SANITY-CLIP NOTE (do not regress):
The single real-world faulty Freesound recording ``fan's frame broken.mp3``
is the reserved sanity-check clip (TDD sec 3.1 / PRD FR7). It is EXCLUDED
from the training set here and must NEVER be added to training or
evaluation splits. The exclusion rule is deliberately conservative: any
filename containing the substring ``"broken"`` is skipped. To run the
explicit post-hoc sanity check on it (TDD sec 5), locate it with
``find_sanity_clip()`` — it is never part of ``collect()``'s output.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from collections import Counter
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Labels as literal ints to keep dataset/metrics code ambiguity-free.
LABEL_NORMAL = 0
LABEL_FAULTY = 1

# Filenames containing this substring are always excluded from training:
# the held-out real-world sanity-check clip (e.g. "fan's frame broken.mp3").
SANITY_SUBSTRING = "broken"


# MIMII DUE/DG filenames embed the label, e.g.
# section_00_source_train_normal_0000_spd_1.wav -> "normal"
MIMII_LABEL_RE = re.compile(r"_(normal|anomaly)_")

# Original MIMII / DCASE2020 Task 2 filenames use a different convention,
# e.g. normal_id_00_00000000.wav / anomaly_id_00_00000000.wav -> label is a
# leading token, not a substring flanked by underscores on both sides.
DCASE2020_LABEL_RE = re.compile(r"^(normal|anomaly)_id_\d+_")


def _label_from_filename(wav_path: Path) -> int | None:
    """Derive the label from the filename (TDD sec 3.1).

    Tries the MIMII DUE/DG convention (``_normal_``/``_anomaly_`` substring)
    first, then the older MIMII/DCASE2020 convention (``normal_id_XX_...`` /
    ``anomaly_id_XX_...`` leading token). Returns ``None`` for files matching
    neither so they can be skipped with a warning.
    """
    match = MIMII_LABEL_RE.search(wav_path.name)
    if match is None:
        match = DCASE2020_LABEL_RE.match(wav_path.name)
    if match is None:
        return None
    return LABEL_NORMAL if match.group(1) == "normal" else LABEL_FAULTY


def _walk(root: Path, suffixes: set[str]):
    """Yield sorted files under ``root`` matching ``suffixes``.

    Uses ``os.walk(followlinks=True)`` because data roots may contain
    symlinked subdirectories (e.g. ``data/raw/mimii/fan_dg -> /kaggle/input/...``
    on Kaggle). ``Path.rglob`` faithfully does NOT descend into symlinked
    directories, silently dropping whole MIMII sections.
    """
    out: list[Path] = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if Path(name).suffix.lower() in suffixes:
                out.append(Path(dirpath) / name)
    out.sort()
    return out


def collect(
    mimii_root: str | Path,
    extra_normal_root: str | Path,
    sanity_filename: str | None = None,
) -> tuple[list[Path], list[int]]:
    """Scan both roots and return ``(filepaths, labels)`` for training.

    - MIMII ``.wav`` files (DUE format) are labeled by filename substring
      (``_normal_`` -> 0, ``_anomaly_`` -> 1); DUE's train/test folders are
      re-pooled into a single source per TDD sec 3.1.
    - Freesound ``.mp3``/``.wav`` files are labeled ``Normal`` (0), except any
      file whose name contains ``"broken"`` (the held-out sanity clip), which
      is excluded entirely.

    Raises ``FileNotFoundError`` if either configured root does not exist,
    so a misconfigured path fails loudly before training.
    """
    mimii_root = Path(mimii_root)
    extra_normal_root = Path(extra_normal_root)
    if not mimii_root.is_dir():
        raise FileNotFoundError(f"MIMII root not found: {mimii_root}")
    if not extra_normal_root.is_dir():
        raise FileNotFoundError(f"Extra-normal root not found: {extra_normal_root}")

    filepaths: list[Path] = []
    labels: list[int] = []

    # MIMII fan subset (DUE format): recursive scan, label by filename
    # substring (_normal_ -> 0, _anomaly_ -> 1). DUE's train/test folder
    # split is re-pooled (TDD sec 3.1) - every clip is training material.
    for wav in _walk(mimii_root, {".wav"}):
        label = _label_from_filename(wav)
        if label is None:
            logger.warning(
                "Skipping MIMII file without a _normal_/_anomaly_ tag: %s", wav
            )
            continue
        filepaths.append(wav)
        labels.append(label)

    # Freesound clips: all Normal (0) EXCEPT any name containing "broken"
    # (the held-out sanity clip) — excluded from training entirely.
    for clip in _walk(extra_normal_root, {".mp3", ".wav"}):
        if SANITY_SUBSTRING in clip.name.lower():
            logger.info(
                "Excluding held-out sanity clip from training: %s", clip.name
            )
            continue
        filepaths.append(clip)
        labels.append(LABEL_NORMAL)

    if sanity_filename is not None and not any(
        SANITY_SUBSTRING in p.name.lower() for p in filepaths
    ):
        logger.info("Sanity clip '%s' not in manifest (expected: it is held out).", sanity_filename)

    return filepaths, labels


def find_sanity_clip(
    extra_normal_root: str | Path, sanity_filename: str
) -> Path | None:
    """Locate the reserved real-world sanity clip outside the training set.

    Returns ``None`` if it was not found so the sanity check can be reported
    honestly rather than crash (TDD sec 5: report the result whether correct
    or not).
    """
    extra_normal_root = Path(extra_normal_root)
    for clip in _walk(extra_normal_root, {".mp3", ".wav", ".flac", ".ogg"}):
        if clip.name.lower() == sanity_filename.lower():
            return clip
    return None


def summarize(filepaths: list[Path], labels: list[int]) -> str:
    """Human-readable class/source counts for the pre-training audit."""
    counts = Counter(labels)
    mimii_normal = sum(
        1 for p, lbl in zip(filepaths, labels) if lbl == LABEL_NORMAL and "mimii" in str(p).lower()
    )
    freesound_normal = sum(
        1 for p, lbl in zip(filepaths, labels) if lbl == LABEL_NORMAL and "freesound" in str(p).lower()
    )
    return (
        f"Total clips: {len(filepaths)}\n"
        f"  Normal  (0): {counts[LABEL_NORMAL]}  (MIMII: {mimii_normal}, Freesound: {freesound_normal})\n"
        f"  Faulty  (1): {counts[LABEL_FAULTY]}  (all MIMII, from '_anomaly_' filenames)"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Scan MIMII + Freesound into a labeled manifest.")
    default_cfg = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
    parser.add_argument("--config", type=Path, default=default_cfg)
    args = parser.parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    filepaths, labels = collect(
        cfg["mimii_root"],
        cfg["extra_normal_root"],
        cfg.get("sanity_clip_filename"),
    )

    print(f"Manifest from config: {args.config}")
    print(summarize(filepaths, labels))

    sanity = find_sanity_clip(cfg["extra_normal_root"], cfg.get("sanity_clip_filename", ""))
    if sanity is not None:
        print(f"Held-out sanity clip (never in train/eval splits): {sanity}")


if __name__ == "__main__":
    main()