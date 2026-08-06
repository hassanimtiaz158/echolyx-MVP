"""Investor-facing Gradio demo (TDD sec 6, PRD FR1/FR8).

Backed by ``src.inference.predict_single_file`` (single source of truth).
- Input: audio upload OR microphone recording (``gr.Audio``).
- Output: ``gr.Label`` showing BOTH class probabilities (Normal/Faulty).
- Model + checkpoint paths come from ``configs/config.yaml``.

Launch:
- Local:    ``python app/gradio_app.py``   (share=False)
- Notebook: ``python app/gradio_app.py --share`` for a temporary public
  Gradio share link

NOTE: ``share=True`` is ONLY for ephemeral notebook/demo contexts (the share
link is session-bound). It is NOT implied as a permanent/production
deployment - a hosted version belongs on a Hugging Face Space (post-MVP,
TDD sec 6).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gradio as gr
import torch

from src.inference import load_model, predict_single_file
from src.train import load_config

logger = logging.getLogger(__name__)


def build_demo(config_path: str | Path) -> gr.Interface:
    """Build (but do not launch) the Gradio interface around the trained model.

    The returned ``gr.Interface`` is launched by the caller, which keeps the
    screen-off/CI import path side-effect free.
    """
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        cfg["backbone_checkpoint"],
        Path(cfg["checkpoint_dir"]) / "best.pt",
        cfg["num_classes"],
    ).to(device)
    model.eval()
    logger.info("Demo model loaded on %s | config: %s", device, config_path)

    def classify(audio_path: str | None) -> dict[str, float]:
        if not audio_path:
            raise gr.Error("No audio provided - upload a clip or record from your microphone.")
        try:
            pred = predict_single_file(audio_path, model, cfg)
            return pred.probabilities  # {Normal: p0, Faulty: p1} for gr.Label
        except Exception as exc:  # surface audio errors in the UI instead of crashing
            logger.error("Prediction failed: %s", exc)
            raise gr.Error(f"Prediction failed: {exc}")

    def hint() -> str:
        low = cfg.get("uncertain_low")
        high = cfg.get("faulty_cutoff")
        gate = (
            f"P(Faulty) between {low} and {high} is reported as 'Uncertain'."
            if low is not None and high is not None
            else "Decisions use P(Faulty) >= 0.5 (argmax)."
        )
        return (
            f"Decision rule: Faulty when P(Faulty) >= {high}. {gate}"
        )

    return gr.Interface(
        fn=classify,
        inputs=gr.Audio(
            type="filepath",
            sources=["upload", "microphone"],
            label="Fan audio (upload or record)",
        ),
        outputs=gr.Label(
            num_top_classes=2,
            label="Prediction - Normal vs Faulty probabilities",
        ),
        title="Echolyx AI - Fan Anomaly Detector",
        description="Proof-of-concept demo: upload or record a ~10 s fan recording for an "
        "instant Normal vs Faulty readout with confidence. Echolyx AI uses PANNs "
        "CNN14 transfer learning fine-tuned on real fan audio (see PRD/TDD). "
        "Research prototype, not production monitoring.",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run the Echolyx AI fan demo.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument(
        "--share",
        action="store_true",
        help="Emit a temporary public gradio share link (notebook/demo contexts only).",
    )
    args = parser.parse_args()
    demo = build_demo(args.config)
    # share=True only for temporary notebook/demo links (session-bound); not a
    # production deployment - that would live in a hosted HF Space (post-MVP).
    demo.launch(share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()