"""Download the PANNs CNN14 AudioSet-pretrained checkpoint (TDD sec 4.1).

Fetches ``Cnn14_mAP=0.431.pth`` (~308 MB) from Zenodo record 3987831
(the official PANNs weights) into ``checkpoint_dir``.

Usage:
    python -m src.models.download_checkpoint --dest checkpoints

Resumes gracefully if the target file already exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import urllib.request

CHECKPOINT_URL = (
    "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth"
)
CHECKPOINT_FILENAME = "Cnn14_mAP=0.431.pth"

# Approx 308 MB (optional guard against double-downloading a truncated file).
MIN_EXPECTED_BYTES = 300 * 1024 * 1024


def download(dest: str | Path) -> Path:
    """Download the pretrained Cnn14 checkpoint to ``dest``; returns its path."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / CHECKPOINT_FILENAME

    if out_path.exists():
        if out_path.stat().st_size >= MIN_EXPECTED_BYTES:
            print(f"Checkpoint already present: {out_path}")
            return out_path
        print(f"Existing file looks incomplete ({out_path.stat().st_size} bytes); re-downloading.")

    print(f"Downloading {CHECKPOINT_URL}")
    print(f"-> {out_path}  (this is ~308 MB; may take a while)")
    urllib.request.urlretrieve(CHECKPOINT_URL, out_path)
    print(f"Done: {out_path} ({out_path.stat().st_size} bytes)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the PANNs CNN14 checkpoint.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to save Cnn14_mAP=0.431.pth into (default: checkpoints/)",
    )
    args = parser.parse_args()
    try:
        download(args.dest)
    except Exception as exc:  # e.g. sandboxed/offline environment
        print(f"Download failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()