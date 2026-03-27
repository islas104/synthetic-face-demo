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
import threading
from pathlib import Path

from core.watermark import apply_visible
from core.detect import analyze

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_app_live = None    # 320px det — fast, for webcam frames
_app_hd = None      # 640px det — accurate, for uploaded still photos
_swapper = None
MODEL_OK = False
MODEL_ERROR = ""


def _build_providers():
    """Prefer CoreML (Apple Silicon / macOS GPU) then fall back to CPU."""
    import onnxruntime as ort
    available = ort.get_available_providers()
    order = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in available]


def _load_models():
    global _app_live, _app_hd, _swapper, MODEL_OK, MODEL_ERROR

    from insightface.app import FaceAnalysis
    import insightface

    model_dir = Path(__file__).parent / "models"
    providers = _build_providers()
    print(f"Using providers: {providers}")

    # Live detector — small grid, fast
    _app_live = FaceAnalysis(name="buffalo_l", root=str(model_dir), providers=providers)
    _app_live.prepare(ctx_id=0, det_size=(320, 320))

    # Still-image detector — full 640 grid, used only for source photo
    _app_hd = FaceAnalysis(name="buffalo_l", root=str(model_dir), providers=providers)
    _app_hd.prepare(ctx_id=0, det_size=(640, 640))

    swap_path = model_dir / "inswapper_128.onnx"
    if not swap_path.exists():
        raise FileNotFoundError(
            "models/inswapper_128.onnx not found.\n"
            "Run: curl -L https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx "
            "-o models/inswapper_128.onnx"
        )
    _swapper = insightface.model_zoo.get_model(str(swap_path), download=False,
                                               providers=providers)
    MODEL_OK = True


try:
    _load_models()
except Exception as e:
    MODEL_ERROR = str(e)
    print(f"Model load error: {e}")


# ---------------------------------------------------------------------------
# Colour transfer (Lab histogram matching for better blending)
# ---------------------------------------------------------------------------

def _color_transfer(src_bgr: np.ndarray, dst_bgr: np.ndarray) -> np.ndarray:
    """
    Transfer the colour statistics of dst onto src so that the swapped
    face matches the lighting/skin tone of the target frame.
    """
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    for ch in range(3):
        src_mean, src_std = src_lab[:, :, ch].mean(), src_lab[:, :, ch].std()
        dst_mean, dst_std = dst_lab[:, :, ch].mean(), dst_lab[:, :, ch].std()
        if src_std < 1e-6:
            continue
        src_lab[:, :, ch] = (src_lab[:, :, ch] - src_mean) * (dst_std / src_std) + dst_mean

    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# IoU face tracker — skip detection when face is stable
# ---------------------------------------------------------------------------

def _iou(a, b) -> float:
    """Intersection-over-Union of two bboxes [x1,y1,x2,y2]."""
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    IOU_THRESHOLD = 0.55   # if overlap > this, reuse cached face
    MAX_MISS = 8           # force re-detect after this many missed frames

    def __init__(self):
        self.cached_faces = None
        self.last_bboxes = []
        self.miss_count = 0
        self.frame_count = 0

    def get_faces(self, frame_bgr):
        self.frame_count += 1
        current_bboxes = self._quick_bboxes(frame_bgr)

        stable = (
            self.cached_faces is not None
            and self.miss_count < self.MAX_MISS
            and len(current_bboxes) == len(self.last_bboxes)
            and all(
                _iou(c, l) > self.IOU_THRESHOLD
                for c, l in zip(current_bboxes, self.last_bboxes)
            )
        )

        if not stable:
            self.cached_faces = _app_live.get(frame_bgr)
            self.last_bboxes = [f.bbox.tolist() for f in self.cached_faces]
            self.miss_count = 0
        else:
            self.miss_count += 1

        return self.cached_faces

    @staticmethod
    def _quick_bboxes(frame_bgr):
        """Fast Haar cascade pre-check to see if face count changed."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = detector.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) == 0:
            return []
        return [[x, y, x + w, y + h] for x, y, w, h in faces]


_tracker = FaceTracker()


# ---------------------------------------------------------------------------
# Threaded frame pipeline
# ---------------------------------------------------------------------------

_source_face = None
_last_result = None          # last successfully processed RGB frame
_pipeline_lock = threading.Lock()
_processing = False          # True while a swap is in progress

PROCESS_WIDTH = 480          # swap resolution — lower = faster


def _swap_worker(bgr_orig):
    """Runs in a background thread. Writes to _last_result when done."""
    global _last_result, _processing

    try:
        orig_h, orig_w = bgr_orig.shape[:2]
        scale = PROCESS_WIDTH / orig_w
        small = cv2.resize(bgr_orig, (PROCESS_WIDTH, int(orig_h * scale)))

        faces = _tracker.get_faces(small)
        if not faces:
            out = apply_visible(bgr_orig)
            with _pipeline_lock:
                _last_result = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            return

        result = small.copy()
        with _pipeline_lock:
            sf = _source_face
        if sf is None:
            return

        for face in faces:
            result = _swapper.get(result, face, sf, paste_back=True)

        # Colour transfer to match target lighting
        result = _color_transfer(result, small)

        result = cv2.resize(result, (orig_w, orig_h))
        result = apply_visible(result)

        with _pipeline_lock:
            _last_result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    finally:
        _processing = False


def process_frame(webcam_frame):
    """
    Called for every webcam frame by Gradio.
    Kicks off a background swap if one isn't already running,
    and immediately returns the last completed result (non-blocking).
    """
    global _processing, _last_result

    if webcam_frame is None:
        return _last_result

    bgr = cv2.cvtColor(webcam_frame, cv2.COLOR_RGB2BGR)

    if not MODEL_OK or _source_face is None:
        out = apply_visible(bgr)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    if not _processing:
        _processing = True
        t = threading.Thread(target=_swap_worker, args=(bgr,), daemon=True)
        t.start()

    with _pipeline_lock:
        result = _last_result

    # On first frame before any result is ready, return the raw feed
    if result is None:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return result


# ---------------------------------------------------------------------------
# Source face setup
# ---------------------------------------------------------------------------

def _detect_with_fallback(bgr):
    # Use the HD detector (640px) for still photos — more reliable
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
                f.kps /= scale
            return faces
    return []


def set_source(photo_pil):
    global _source_face, _last_result

    if photo_pil is None:
        with _pipeline_lock:
            _source_face = None
        return "No photo uploaded."

    if not MODEL_OK:
        return f"Model not loaded: {MODEL_ERROR}"

    bgr = cv2.cvtColor(np.array(photo_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = _detect_with_fallback(bgr)

    if not faces:
        with _pipeline_lock:
            _source_face = None
        return "No face detected. Try a photo with better lighting or a slightly wider crop."

    best = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0]

    with _pipeline_lock:
        _source_face = best
        _last_result = None   # flush cached result so next frame is fresh

    return "Face locked in. Ready — enable your webcam below."


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
