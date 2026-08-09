"""Hugging Face Spaces entrypoint (TDD sec 6 post-MVP hosting).

Spaces runs this file directly (``app_file: space_app.py`` in the Space's
``README.md`` front matter) and serves whatever Gradio object it finds. Not
named ``app.py`` deliberately — that collides with the ``app/`` package
directory (Python can't disambiguate a same-named file and package in the
same directory). This file is deployment glue only — the actual UI/inference
logic is ``app/gradio_app.py``, reused as-is (single source of truth, same
code path as the local/Kaggle demo).

The ~308 MB AudioSet backbone is fetched from Zenodo on first startup (same
``src.models.download_checkpoint`` used locally/on Kaggle) rather than
committed to the Space repo. ``checkpoints/best.pt`` (the fine-tuned model)
is NOT fetched automatically — it must be committed to the Space repo
directly (see README "Deploying to Hugging Face Spaces").
"""

from pathlib import Path

from app.gradio_app import build_demo
from src.models.download_checkpoint import download
from src.train import load_config

CONFIG_PATH = Path(__file__).parent / "configs" / "config.yaml"

cfg = load_config(CONFIG_PATH)
download(cfg["checkpoint_dir"])  # AudioSet backbone; no-ops if already present

demo = build_demo(CONFIG_PATH)

if __name__ == "__main__":
    demo.launch()
