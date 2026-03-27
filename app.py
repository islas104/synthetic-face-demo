"""
Synthetic Face Demo — Real-time Webcam Face Swap
By Islas Nawaz
"""

import cv2
import numpy as np
import gradio as gr
from pathlib import Path

from core.watermark import apply_visible
from core.detect import analyze

# Only the normed_embedding is needed by inswapper — save as plain numpy
SOURCE_FACE_FILE = Path("/tmp/synthetic_demo_source.npy")


class _FaceEmbed:
    """Minimal face-like object carrying just the embedding inswapper needs."""
    def __init__(self, normed_embedding):
        self.normed_embedding = normed_embedding

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_app_live = None
_app_hd   = None
_swapper  = None
MODEL_OK  = False
MODEL_ERROR = ""


def _load_models():
    global _app_live, _app_hd, _swapper, MODEL_OK, MODEL_ERROR
    from insightface.app import FaceAnalysis
    import insightface
    import onnxruntime as ort

    model_dir = Path(__file__).parent / "models"
    det_providers = [p for p in ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                     if p in ort.get_available_providers()]

    # Both detectors use 640 — matches our PROCESS_W so no scale mismatch
    _app_live = FaceAnalysis(name="buffalo_l", root=str(model_dir), providers=det_providers)
    _app_live.prepare(ctx_id=0, det_size=(640, 640))

    _app_hd = FaceAnalysis(name="buffalo_l", root=str(model_dir), providers=det_providers)
    _app_hd.prepare(ctx_id=0, det_size=(640, 640))

    swap_path = model_dir / "inswapper_128.onnx"
    if not swap_path.exists():
        raise FileNotFoundError("models/inswapper_128.onnx not found.")

    # inswapper_128 is incompatible with CoreML — CPU only
    _swapper = insightface.model_zoo.get_model(
        str(swap_path), download=False, providers=["CPUExecutionProvider"]
    )
    MODEL_OK = True


try:
    _load_models()
except Exception as e:
    MODEL_ERROR = str(e)
    print(f"Model load error: {e}")


# ---------------------------------------------------------------------------
# Source face — saved to disk so both Gradio processes can access it
# ---------------------------------------------------------------------------

def _save_source(face):
    np.save(str(SOURCE_FACE_FILE), face.normed_embedding)


_source_cache      = None   # in-memory cache
_source_cache_mtime = None  # mtime when cache was last loaded

def _load_source():
    """Return cached source face, reloading from disk only if the file changed."""
    global _source_cache, _source_cache_mtime
    if not SOURCE_FACE_FILE.exists():
        _source_cache = None
        return None
    mtime = SOURCE_FACE_FILE.stat().st_mtime
    if _source_cache is None or mtime != _source_cache_mtime:
        _source_cache       = _FaceEmbed(np.load(str(SOURCE_FACE_FILE)))
        _source_cache_mtime = mtime
    return _source_cache


def _detect_hd(bgr):
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
    if photo_pil is None:
        SOURCE_FACE_FILE.unlink(missing_ok=True)
        return "No photo uploaded."
    if not MODEL_OK:
        return f"Model not loaded: {MODEL_ERROR}"

    bgr = cv2.cvtColor(np.array(photo_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = _detect_hd(bgr)
    if not faces:
        return "No face detected — try a clearer, front-facing photo."

    best = sorted(faces,
        key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]),
        reverse=True)[0]

    _save_source(best)
    return "✓ Face locked in. Enable your webcam below."


# ---------------------------------------------------------------------------
# Live frame processing
# ---------------------------------------------------------------------------

_frame_idx    = 0
_cached_faces = None   # last SUCCESSFUL detection — never blanked by a single miss
_miss_streak  = 0      # consecutive frames with no detection
DETECT_EVERY  = 3      # attempt detection every N frames
MAX_MISS      = 20     # clear cache only after this many consecutive misses
PROCESS_W     = 640


def _detect_live(small):
    """Detect faces with one scale-down fallback for close-up/large faces."""
    faces = _app_live.get(small)
    if faces:
        return faces
    h, w = small.shape[:2]
    shrunk = cv2.resize(small, (int(w * 0.75), int(h * 0.75)))
    faces = _app_live.get(shrunk)
    for f in faces:
        f.bbox /= 0.75
        f.kps  /= 0.75
    return faces


def process_frame(webcam_frame):
    global _frame_idx, _cached_faces, _miss_streak

    try:
        if webcam_frame is None:
            return webcam_frame

        if not isinstance(webcam_frame, np.ndarray):
            webcam_frame = np.array(webcam_frame)

        bgr = cv2.cvtColor(webcam_frame, cv2.COLOR_RGB2BGR)

        if not MODEL_OK:
            return cv2.cvtColor(apply_visible(bgr), cv2.COLOR_BGR2RGB)

        source_face = _load_source()
        if source_face is None:
            return cv2.cvtColor(apply_visible(bgr), cv2.COLOR_BGR2RGB)

        _frame_idx += 1

        orig_h, orig_w = bgr.shape[:2]
        scale = PROCESS_W / orig_w
        small = cv2.resize(bgr, (PROCESS_W, int(orig_h * scale)))

        # Attempt detection every N frames
        if _frame_idx % DETECT_EVERY == 1 or _cached_faces is None:
            found = _detect_live(small)
            if found:
                _cached_faces = found   # only update cache on success
                _miss_streak  = 0
            else:
                _miss_streak += 1
                if _miss_streak >= MAX_MISS:
                    _cached_faces = None  # truly no face for a while — reset

        # No face found yet at all — pass through raw feed
        if not _cached_faces:
            return cv2.cvtColor(apply_visible(bgr), cv2.COLOR_BGR2RGB)

        result = small.copy()
        for face in _cached_faces:
            result = _swapper.get(result, face, source_face, paste_back=True)

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

CSS = """
:root {
    --primary: #6c63ff;
    --primary-dark: #4b44cc;
    --bg: #0f0f13;
    --surface: #1a1a24;
    --surface2: #22222f;
    --border: #2e2e42;
    --text: #e8e8f0;
    --muted: #888899;
    --success: #22c55e;
    --warning: #f59e0b;
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.app-header p { color: var(--muted); font-size: 0.9rem; margin: 0; }

/* Tabs */
.tab-nav { border-bottom: 1px solid var(--border) !important; }
.tab-nav button {
    color: var(--muted) !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.2rem !important;
}
.tab-nav button.selected {
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
}

/* Cards */
.card {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
}

/* Step badges */
.step-label {
    display: inline-block;
    background: var(--primary);
    color: white;
    border-radius: 50%;
    width: 24px; height: 24px;
    line-height: 24px;
    text-align: center;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 0.5rem;
}

/* Buttons */
button.primary-btn, .gr-button-primary {
    background: var(--primary) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: background 0.2s !important;
}
button.primary-btn:hover, .gr-button-primary:hover {
    background: var(--primary-dark) !important;
}

/* Image panels */
.image-panel .wrap {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    min-height: 300px !important;
}

/* Status box */
.status-box textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--success) !important;
    font-size: 0.85rem !important;
}

/* Disclaimer banner */
.disclaimer {
    background: linear-gradient(135deg, #1e1030, #12122a);
    border: 1px solid #3d2e6e;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.8rem;
    color: #a78bfa;
    text-align: center;
    margin-bottom: 1rem;
}

/* Detection result */
.detect-result {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem;
    min-height: 80px;
}

/* Footer */
.footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.78rem;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid var(--border);
    margin-top: 1.5rem;
}
"""

with gr.Blocks(title="Synthetic Face Demo — Islas Nawaz", css=CSS, theme=gr.themes.Base()) as demo:

    # Header
    gr.HTML("""
        <div class="app-header">
            <h1>Synthetic Face Demo</h1>
            <p>Real-time deepfake demonstration &nbsp;·&nbsp; By <strong>Islas Nawaz</strong></p>
        </div>
    """)

    gr.HTML("""
        <div class="disclaimer">
            All outputs are watermarked &nbsp;|&nbsp;
            Educational use only &nbsp;|&nbsp;
            Only use faces you have explicit consent to use
        </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Live Swap ──────────────────────────────────────────────
        with gr.Tab("Live Face Swap"):
            with gr.Row(equal_height=False):

                # Left panel — source photo
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div style="margin-bottom:0.75rem"><span class="step-label">1</span> <strong>Upload a face photo</strong></div>')
                    source_photo = gr.Image(
                        type="pil",
                        label="Whose face do you want?",
                        sources=["upload"],
                        height=260,
                        elem_classes=["image-panel"],
                    )
                    gr.HTML('<div style="margin:0.75rem 0"><span class="step-label">2</span> <strong>Lock it in</strong></div>')
                    lock_btn = gr.Button("Lock in face", variant="primary", size="lg")
                    status = gr.Textbox(
                        label="",
                        interactive=False,
                        placeholder="Status will appear here...",
                        elem_classes=["status-box"],
                        lines=1,
                    )
                    lock_btn.click(set_source, inputs=source_photo, outputs=status)

                # Right panel — webcam + result
                with gr.Column(scale=2):
                    gr.HTML('<div style="margin-bottom:0.75rem"><span class="step-label">3</span> <strong>Enable your webcam — swap happens live</strong></div>')
                    with gr.Row():
                        webcam = gr.Image(
                            label="Your webcam",
                            sources=["webcam"],
                            streaming=True,
                            type="numpy",
                            height=320,
                            elem_classes=["image-panel"],
                        )
                        output = gr.Image(
                            label="Swapped result",
                            type="numpy",
                            height=320,
                            elem_classes=["image-panel"],
                        )

            webcam.stream(
                process_frame,
                inputs=webcam,
                outputs=output,
                stream_every=0.1,
                time_limit=None,
            )

        # ── Tab 2: Detection ─────────────────────────────────────────────
        with gr.Tab("Deepfake Detection"):
            gr.HTML('<p style="color:var(--muted,#888);margin-bottom:1rem;">Upload any image to analyse whether it looks synthetic.</p>')
            with gr.Row():
                with gr.Column(scale=1):
                    detect_img = gr.Image(
                        type="pil",
                        label="Image to analyse",
                        sources=["upload"],
                        height=320,
                        elem_classes=["image-panel"],
                    )
                    detect_btn = gr.Button("Analyse image", variant="primary", size="lg")
                with gr.Column(scale=1):
                    detect_out = gr.Markdown(
                        value="*Upload an image and click Analyse.*",
                        elem_classes=["detect-result"],
                    )
            detect_btn.click(run_detect, inputs=detect_img, outputs=detect_out)

    gr.HTML('<div class="footer">Created by Islas Nawaz for responsible AI education &nbsp;·&nbsp; All outputs watermarked</div>')

if __name__ == "__main__":
    demo.queue().launch()
