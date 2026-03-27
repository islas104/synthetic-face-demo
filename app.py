"""
Synthetic Face Demo — Real-time Webcam Face Swap
By Islas Nawaz

Flow:
  1. Upload a photo of the face you want to become
  2. Hit "Lock in face", then enable your webcam
  3. Detection tab lets you analyse any image for deepfake artifacts
"""

import cv2
import numpy as np
import gradio as gr
from pathlib import Path

from core.watermark import apply_visible
from core.detect import analyze

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_app_live = None   # 320px — fast, for webcam frames
_app_hd   = None   # 640px — accurate, for uploaded still photos
_swapper  = None
MODEL_OK  = False
MODEL_ERROR = ""


def _build_providers():
    import onnxruntime as ort
    order = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in ort.get_available_providers()]


def _load_models():
    global _app_live, _app_hd, _swapper, MODEL_OK, MODEL_ERROR
    from insightface.app import FaceAnalysis
    import insightface

    model_dir = Path(__file__).parent / "models"
    providers = _build_providers()
    print(f"Using providers: {providers}")

    _app_live = FaceAnalysis(name="buffalo_l", root=str(model_dir), providers=providers)
    _app_live.prepare(ctx_id=0, det_size=(320, 320))

    _app_hd = FaceAnalysis(name="buffalo_l", root=str(model_dir), providers=providers)
    _app_hd.prepare(ctx_id=0, det_size=(640, 640))

    swap_path = model_dir / "inswapper_128.onnx"
    if not swap_path.exists():
        raise FileNotFoundError("models/inswapper_128.onnx not found.")

    _swapper = insightface.model_zoo.get_model(str(swap_path), download=False,
                                               providers=providers)
    MODEL_OK = True


try:
    _load_models()
except Exception as e:
    MODEL_ERROR = str(e)
    print(f"Model load error: {e}")


# ---------------------------------------------------------------------------
# Colour transfer — match swapped face lighting to the webcam frame
# ---------------------------------------------------------------------------

def _color_transfer(swapped_bgr, target_bgr):
    """Shift swapped image colour stats to match the target frame."""
    src = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst = cv2.cvtColor(target_bgr,  cv2.COLOR_BGR2LAB).astype(np.float32)
    for ch in range(3):
        s_mean, s_std = src[:, :, ch].mean(), src[:, :, ch].std()
        d_mean, d_std = dst[:, :, ch].mean(), dst[:, :, ch].std()
        if s_std < 1e-6:
            continue
        src[:, :, ch] = (src[:, :, ch] - s_mean) * (d_std / s_std) + d_mean
    return cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Source face
# ---------------------------------------------------------------------------

_source_face = None


def _detect_hd(bgr):
    """Multi-scale detection using the HD 640px detector."""
    faces = _app_hd.get(bgr)
    if faces:
        return faces
    for scale in [0.75, 0.5, 0.35]:
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (int(w * scale), int(h * scale)))
        faces = _app_hd.get(small)
        if faces:
            for f in faces:
                f.bbox /= scale
                f.kps  /= scale
            return faces
    return []


def set_source(photo_pil):
    global _source_face
    if photo_pil is None:
        _source_face = None
        return "No photo uploaded."
    if not MODEL_OK:
        return f"Model not loaded: {MODEL_ERROR}"

    bgr = cv2.cvtColor(np.array(photo_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = _detect_hd(bgr)
    if not faces:
        _source_face = None
        return "No face detected in the photo — try a clearer, front-facing image."

    _source_face = sorted(faces,
        key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]),
        reverse=True)[0]
    return "Face locked in. Enable your webcam below."


# ---------------------------------------------------------------------------
# Live frame processing
# ---------------------------------------------------------------------------

_frame_idx   = 0
_cached_faces = None   # last InsightFace detections on the live frame
DETECT_EVERY  = 3      # re-run face detection every N frames
PROCESS_W     = 480    # width to run swap at (lower = faster)


def process_frame(webcam_frame):
    global _frame_idx, _cached_faces

    try:
        if webcam_frame is None:
            return webcam_frame

        # Gradio 6 sends numpy RGB
        if not isinstance(webcam_frame, np.ndarray):
            webcam_frame = np.array(webcam_frame)

        bgr = cv2.cvtColor(webcam_frame, cv2.COLOR_RGB2BGR)

        if not MODEL_OK or _source_face is None:
            return cv2.cvtColor(apply_visible(bgr), cv2.COLOR_BGR2RGB)

        _frame_idx += 1

        # Downscale for processing
        orig_h, orig_w = bgr.shape[:2]
        scale = PROCESS_W / orig_w
        small = cv2.resize(bgr, (PROCESS_W, int(orig_h * scale)))

        # Re-detect faces every DETECT_EVERY frames
        if _frame_idx % DETECT_EVERY == 1 or _cached_faces is None:
            _cached_faces = _app_live.get(small)

        if not _cached_faces:
            return cv2.cvtColor(apply_visible(bgr), cv2.COLOR_BGR2RGB)

        result = small.copy()
        for face in _cached_faces:
            result = _swapper.get(result, face, _source_face, paste_back=True)

        result = _color_transfer(result, small)
        result = cv2.resize(result, (orig_w, orig_h))
        result = apply_visible(result)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"Frame error: {e}")
        return webcam_frame


# ---------------------------------------------------------------------------
# Detection tab
# ---------------------------------------------------------------------------

def run_detect(img_pil):
    if img_pil is None:
        return "Please upload an image."
    bgr = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    r = analyze(bgr)
    return (
        f"**{r['label']}**\n\n"
        f"Confidence: {r['score']*100:.1f}%\n\n"
        f"Method: {r['method']}\n\n"
        "*Score > 50% = more likely synthetic.*"
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
                    label="Whose face do you want?",
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
                    type="numpy",
                )
                output = gr.Image(label="Result (watermarked)", type="numpy")

        webcam.stream(
            process_frame,
            inputs=webcam,
            outputs=output,
            stream_every=0.1,   # ~10 fps
            time_limit=None,
        )

    with gr.Tab("Deepfake Detection"):
        gr.Markdown("Upload any image to check whether it appears synthetic.")
        detect_img = gr.Image(type="pil", label="Image to analyse")
        detect_btn = gr.Button("Analyse", variant="primary")
        detect_out = gr.Markdown()
        detect_btn.click(run_detect, inputs=detect_img, outputs=detect_out)

    gr.Markdown("---\n*Created by Islas Nawaz for educational purposes.*")

if __name__ == "__main__":
    demo.queue().launch()
