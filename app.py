"""
Synthetic Face Demo — Gradio UI
By Islas Nawaz

Tabs:
  1. Face Swap   — swap a face from one image onto another
  2. Detection   — analyze an image for deepfake artifacts
"""

import cv2
import numpy as np
import gradio as gr
from PIL import Image

from core.swap import FaceSwapper
from core.watermark import watermark
from core.detect import analyze

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Tab 1 — Face Swap
# ---------------------------------------------------------------------------

_swapper = None

def get_swapper():
    global _swapper
    if _swapper is None:
        _swapper = FaceSwapper()
    return _swapper


CONSENT_TEXT = (
    "I confirm that I have the explicit consent of every person whose likeness "
    "appears in these images and that I will not use the output to deceive, "
    "impersonate, or harm anyone."
)


def run_swap(source_pil, target_pil, consent: bool):
    if not consent:
        return None, "You must confirm consent before proceeding."

    if source_pil is None or target_pil is None:
        return None, "Please upload both a source and a target image."

    try:
        source_bgr = pil_to_bgr(source_pil)
        target_bgr = pil_to_bgr(target_pil)

        result_bgr = get_swapper().swap(source_bgr, target_bgr)
        result_bgr = watermark(result_bgr)

        return bgr_to_pil(result_bgr), "Done. Output is watermarked — Created by Islas Nawaz."
    except Exception as e:
        return None, f"Error: {e}"


# ---------------------------------------------------------------------------
# Tab 2 — Detection
# ---------------------------------------------------------------------------

def run_detect(img_pil):
    if img_pil is None:
        return "Please upload an image."

    bgr = pil_to_bgr(img_pil)
    result = analyze(bgr)

    score_pct = result["score"] * 100
    return (
        f"**{result['label']}**\n\n"
        f"Confidence: {score_pct:.1f}%\n\n"
        f"Method: {result['method']}\n\n"
        f"*Score > 50% = more likely synthetic.*"
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Synthetic Face Demo — Islas Nawaz") as demo:
    gr.Markdown(
        "# Synthetic Face Demo\n"
        "**By Islas Nawaz** — educational demonstration of face-swap technology and deepfake detection.\n\n"
        "> All outputs are watermarked. Read [ETHICS.md](ETHICS.md) before use."
    )

    with gr.Tab("Face Swap"):
        gr.Markdown("Upload a **source** face (to copy from) and a **target** image (where it will be placed).")
        with gr.Row():
            source_img = gr.Image(type="pil", label="Source face")
            target_img = gr.Image(type="pil", label="Target image")
        consent_box = gr.Checkbox(label=CONSENT_TEXT)
        swap_btn = gr.Button("Swap faces", variant="primary")
        with gr.Row():
            output_img = gr.Image(type="pil", label="Result")
            status_txt = gr.Textbox(label="Status", interactive=False)
        swap_btn.click(run_swap, inputs=[source_img, target_img, consent_box], outputs=[output_img, status_txt])

    with gr.Tab("Deepfake Detection"):
        gr.Markdown("Upload any image to analyze whether it appears synthetic.")
        detect_img = gr.Image(type="pil", label="Image to analyze")
        detect_btn = gr.Button("Analyze", variant="primary")
        detect_out = gr.Markdown()
        detect_btn.click(run_detect, inputs=[detect_img], outputs=[detect_out])

    gr.Markdown("---\n*Created by Islas Nawaz for educational purposes.*")


if __name__ == "__main__":
    demo.launch()
