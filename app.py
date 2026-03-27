"""
Synthetic Face Demo — Real-time Webcam Face Swap
By Islas Nawaz

Flow:
  1. Upload a photo of the face you want to become
  2. Hit "Start" — your webcam feed has that face swapped onto you live
  3. Detection tab lets you analyse any image for deepfake artifacts
"""

import cv2
import numpy as np
import gradio as gr
from PIL import Image

from core.watermark import apply_visible
from core.detect import analyze

# ---------------------------------------------------------------------------
# Model loading (done once at startup)
# ---------------------------------------------------------------------------

_app = None
_swapper = None


def _load_models():
    global _app, _swapper
    if _app is not None:
        return

    from insightface.app import FaceAnalysis
    import insightface
    from pathlib import Path

    model_dir = Path(__file__).parent / "models"
    _app = FaceAnalysis(name="buffalo_l", root=str(model_dir))
    _app.prepare(ctx_id=0, det_size=(640, 640))

    swap_path = model_dir / "inswapper_128.onnx"
    if not swap_path.exists():
        raise FileNotFoundError(
            "models/inswapper_128.onnx not found.\n"
            "Run: curl -L https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx "
            "-o models/inswapper_128.onnx"
        )
    _swapper = insightface.model_zoo.get_model(str(swap_path), download=False)


try:
    _load_models()
    MODEL_OK = True
except FileNotFoundError as e:
    MODEL_OK = False
    MODEL_ERROR = str(e)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

# Cache the extracted source face embedding so we don't re-detect every frame
_source_face = None


def _detect_with_fallback(bgr):
    """Try detection at multiple scales to handle close-up / large faces."""
    faces = _app.get(bgr)
    if faces:
        return faces

    # If no face found, try resizing down — helps with very close-up shots
    for scale in [0.75, 0.5, 0.35]:
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        faces = _app.get(small)
        if faces:
            # Scale bounding boxes back up so embeddings align with original
            for f in faces:
                f.bbox /= scale
                f.kps /= scale
            return faces

    return []


def set_source(photo_pil):
    """Extract and cache the face embedding from the uploaded photo."""
    global _source_face

    if photo_pil is None:
        _source_face = None
        return "No photo uploaded."

    if not MODEL_OK:
        return f"Model not loaded: {MODEL_ERROR}"

    bgr = cv2.cvtColor(np.array(photo_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = _detect_with_fallback(bgr)

    if not faces:
        _source_face = None
        return "No face detected. Try a photo with better lighting or a slightly wider crop."

    # Pick the largest face
    _source_face = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )[0]

    return "Face locked in. Ready — start your webcam stream below."


_frame_count = 0
_cached_target_faces = None
DETECT_EVERY_N = 4      # re-run face detection only every 4 frames
PROCESS_WIDTH = 480     # downscale webcam feed before swap (faster on CPU)


def process_frame(webcam_frame):
    """
    Called for every webcam frame.
    webcam_frame: numpy array (RGB) from Gradio's webcam stream.
    Returns: numpy array (RGB) to display.
    """
    global _frame_count, _cached_target_faces

    if webcam_frame is None:
        return webcam_frame

    _frame_count += 1

    # Gradio sends RGB; InsightFace needs BGR
    bgr = cv2.cvtColor(webcam_frame, cv2.COLOR_RGB2BGR)

    if _source_face is None or not MODEL_OK:
        overlay = apply_visible(bgr)
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Downscale for processing, keep original for output blending
    orig_h, orig_w = bgr.shape[:2]
    scale = PROCESS_WIDTH / orig_w
    small = cv2.resize(bgr, (PROCESS_WIDTH, int(orig_h * scale)))

    # Re-detect faces only every N frames to reduce CPU load
    if _frame_count % DETECT_EVERY_N == 1 or _cached_target_faces is None:
        _cached_target_faces = _app.get(small)

    if not _cached_target_faces:
        overlay = apply_visible(bgr)
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    result = small.copy()
    for face in _cached_target_faces:
        result = _swapper.get(result, face, _source_face, paste_back=True)

    # Scale result back to original resolution
    result = cv2.resize(result, (orig_w, orig_h))
    result = apply_visible(result)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Detection tab
# ---------------------------------------------------------------------------

def run_detect(img_pil):
    if img_pil is None:
        return "Please upload an image."
    bgr = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    r = analyze(bgr)
    pct = r["score"] * 100
    return (
        f"**{r['label']}**\n\n"
        f"Confidence: {pct:.1f}%\n\n"
        f"Method: {r['method']}\n\n"
        f"*Score > 50% = more likely synthetic.*"
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Synthetic Face Demo — Islas Nawaz") as demo:
    gr.Markdown(
        "# Synthetic Face Demo\n"
        "**By Islas Nawaz** — real-time webcam face swap for responsible AI education.\n\n"
        "> All outputs are watermarked. Only use faces you have consent to use."
    )

    with gr.Tab("Live Face Swap"):
        gr.Markdown(
            "**Step 1** — Upload a photo of the face you want to wear.\n\n"
            "**Step 2** — Click *Lock in face*, then enable your webcam below."
        )

        with gr.Row():
            with gr.Column(scale=1):
                source_photo = gr.Image(
                    type="pil",
                    label="Target face photo (whose face do you want?)",
                    sources=["upload"],
                )
                lock_btn = gr.Button("Lock in face", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
                lock_btn.click(set_source, inputs=source_photo, outputs=status)

            with gr.Column(scale=2):
                webcam = gr.Image(
                    label="Your webcam — face will be swapped live",
                    sources=["webcam"],
                    streaming=True,
                )
                output = gr.Image(label="Result (watermarked)", streaming=True)

        webcam.stream(process_frame, inputs=webcam, outputs=output)

    with gr.Tab("Deepfake Detection"):
        gr.Markdown("Upload any image to check whether it appears synthetic.")
        detect_img = gr.Image(type="pil", label="Image to analyse")
        detect_btn = gr.Button("Analyse", variant="primary")
        detect_out = gr.Markdown()
        detect_btn.click(run_detect, inputs=detect_img, outputs=detect_out)

    gr.Markdown("---\n*Created by Islas Nawaz for educational purposes.*")

if __name__ == "__main__":
    demo.queue().launch()
