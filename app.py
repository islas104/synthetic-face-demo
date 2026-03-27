"""
Synthetic Face Demo — Real-time Webcam Face Swap
By Islas Nawaz
"""

import cv2
import numpy as np
import gradio as gr
from pathlib import Path
from scipy.spatial import Delaunay

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
_last_swapped = None   # last successfully rendered swap — returned during gaps
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


def _update_detection(small):
    """Run detection every N frames; update cache only on success."""
    global _cached_faces, _miss_streak
    if _frame_idx % DETECT_EVERY != 1 and _cached_faces is not None:
        return
    found = _detect_live(small)
    if found:
        _cached_faces = found
        _miss_streak  = 0
    else:
        _miss_streak += 1
        if _miss_streak >= MAX_MISS:
            _cached_faces = None


# ---------------------------------------------------------------------------
# Expression warping
# ---------------------------------------------------------------------------

def _smile_deltas(dst, lm, cx, fh, mid_y):
    for i, (x, y) in enumerate(lm):
        if y > mid_y:
            # Lift mouth corners strongly, pull centre of lower face up
            x_norm = (x - cx) / (fh * 0.5 + 1e-6)
            dst[i, 1] -= fh * 0.22 * (x_norm ** 2)
        elif y > mid_y - fh * 0.15:
            # Cheeks puff outward slightly
            dst[i, 0] += np.sign(x - cx) * fh * 0.04


def _angry_deltas(dst, lm, cx, fh, mid_y):
    for i, (x, y) in enumerate(lm):
        if y < mid_y:
            x_off = x - cx
            if abs(x_off) < fh * 0.18:        # inner brow → down hard
                dst[i, 1] += fh * 0.18
                dst[i, 0] += np.sign(x_off) * fh * 0.06
            else:                              # outer brow → up
                dst[i, 1] -= fh * 0.06
        else:
            x_norm = (x - cx) / (fh * 0.5 + 1e-6)
            dst[i, 1] += fh * 0.10 * x_norm ** 2   # mouth corners down


def _surprised_deltas(dst, lm, _cx, fh, mid_y):
    for i, (_x, y) in enumerate(lm):
        if y < mid_y:
            t = (mid_y - y) / (fh * 0.55)
            dst[i, 1] -= fh * 0.22 * min(t, 1.0)   # brows shoot up
        elif y > mid_y + fh * 0.20:
            t = (y - (mid_y + fh * 0.20)) / (fh * 0.2)
            dst[i, 1] += fh * 0.18 * min(t, 1.0)   # mouth drops open


def _wink_deltas(dst, lm, cx, fh, mid_y):
    for i, (x, y) in enumerate(lm):
        if y < mid_y and x > cx:               # right eye region → close
            dst[i, 1] += fh * 0.14
        elif y > mid_y and x < cx - fh * 0.2:  # left cheek → lift into smile
            dst[i, 1] -= fh * 0.06


_EXPR_FN = {
    "smile":     _smile_deltas,
    "angry":     _angry_deltas,
    "surprised": _surprised_deltas,
    "wink":      _wink_deltas,
}


def _compute_expression_deltas(lm, expression):
    """Return dst landmarks for the given expression using relative face geometry."""
    lm  = lm.astype(np.float64)
    dst = lm.copy()
    fn  = _EXPR_FN.get(expression)
    if fn is None:
        return dst
    cx    = (lm[:, 0].min() + lm[:, 0].max()) / 2
    fh    = lm[:, 1].max() - lm[:, 1].min()
    mid_y = lm[:, 1].min() + fh * 0.55
    fn(dst, lm, cx, fh, mid_y)
    return dst


def _piecewise_affine_warp(img, src_pts, dst_pts):
    """Warp img using piecewise affine transform on a Delaunay triangulation."""
    h, w = img.shape[:2]
    border = np.array([
        [0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
        [w // 2, 0], [0, h // 2], [w - 1, h // 2], [w // 2, h - 1],
    ], dtype=np.float64)

    src_all = np.vstack([src_pts, border])
    dst_all = np.vstack([dst_pts, border])

    tri = Delaunay(dst_all)
    result = img.copy()

    for simplex in tri.simplices:
        s = src_all[simplex].astype(np.float32)
        d = dst_all[simplex].astype(np.float32)
        M = cv2.getAffineTransform(d, s)          # dst→src (inverse warp)
        warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, d.astype(np.int32), 255)
        result[mask > 0] = warped[mask > 0]

    return result


def _apply_expression(img_bgr, face, expression):
    """Apply expression warp to img_bgr using face.landmark_2d_106."""
    if expression == "neutral":
        return img_bgr
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return img_bgr
    dst = _compute_expression_deltas(lm, expression)
    return _piecewise_affine_warp(img_bgr, lm, dst)


_EXPR_LABEL = {
    "neutral": "😐 Neutral", "smile": "😄 Smile",
    "angry": "😠 Angry", "surprised": "😮 Surprised", "wink": "😉 Wink",
}


def _draw_expr_label(img_bgr, expression):
    """Draw a small expression badge in the top-right corner."""
    if expression == "neutral":
        return img_bgr
    label = _EXPR_LABEL.get(expression, expression)
    _, w = img_bgr.shape[:2]
    fs = max(0.5, w / 800)
    th = max(1, int(fs * 2))
    (tw, txh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = 8
    x0, y0 = w - tw - pad * 2, 8
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x0 - pad, y0), (w - pad, y0 + txh + pad * 2), (40, 20, 80), -1)
    cv2.addWeighted(overlay, 0.75, img_bgr, 0.25, 0, img_bgr)
    cv2.putText(img_bgr, label, (x0, y0 + txh + pad // 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 180, 255), th, cv2.LINE_AA)
    return img_bgr


def _run_swap(small, orig_w, orig_h, source_face, expression):
    """Swap, apply expression warp, scale up, watermark, return RGB."""
    global _last_swapped
    result = small.copy()
    for face in _cached_faces:
        result = _swapper.get(result, face, source_face, paste_back=True)
        result = _apply_expression(result, face, expression)
    result = cv2.resize(result, (orig_w, orig_h))
    result = _draw_expr_label(result, expression)
    result = apply_visible(result)
    _last_swapped = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return _last_swapped


def process_frame(webcam_frame, expression):
    global _frame_idx

    try:
        if webcam_frame is None:
            return gr.skip()

        if not isinstance(webcam_frame, np.ndarray):
            webcam_frame = np.array(webcam_frame)

        bgr = cv2.cvtColor(webcam_frame, cv2.COLOR_RGB2BGR)

        source_face = _load_source()
        if not MODEL_OK or source_face is None:
            return gr.skip()

        _frame_idx += 1
        orig_h, orig_w = bgr.shape[:2]
        small = cv2.resize(bgr, (PROCESS_W, int(orig_h * PROCESS_W / orig_w)))

        _update_detection(small)

        if not _cached_faces:
            return _last_swapped if _last_swapped is not None else gr.skip()

        return _run_swap(small, orig_w, orig_h, source_face, expression)

    except Exception as e:
        print(f"Frame error: {e}")
        return _last_swapped if _last_swapped is not None else gr.skip()


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
    --primary: #7c3aed;
    --primary-light: #a78bfa;
    --primary-dark: #5b21b6;
    --bg: #0a0a0f;
    --surface: #13131e;
    --surface2: #1c1c2e;
    --surface3: #252538;
    --border: #2a2a45;
    --text: #e8e8f0;
    --muted: #7070a0;
    --success: #10b981;
}

body, .gradio-container { background: var(--bg) !important; color: var(--text) !important; }

/* ── Header ── */
.app-header {
    text-align: center; padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border); margin-bottom: 1.5rem;
}
.app-header h1 {
    font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #7c3aed, #c084fc, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.app-header p { color: var(--muted); font-size: 0.95rem; margin: 0; }

/* ── Disclaimer ── */
.disclaimer {
    background: linear-gradient(135deg, #1a0a3e, #0f102a);
    border: 1px solid #4c1d95; border-radius: 10px;
    padding: 0.7rem 1.2rem; font-size: 0.82rem;
    color: #c4b5fd; text-align: center; margin-bottom: 1.5rem;
    letter-spacing: 0.01em;
}

/* ── Step card ── */
.step-card {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}
.step-header {
    display: flex; align-items: center; gap: 0.6rem;
    margin-bottom: 1rem; font-size: 0.95rem; font-weight: 600;
}
.step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #7c3aed, #5b21b6);
    color: white; border-radius: 50%;
    width: 28px; height: 28px; font-size: 0.75rem; font-weight: 800;
    flex-shrink: 0; box-shadow: 0 2px 8px rgba(124,58,237,0.4);
}

/* ── Image panels ── */
.image-panel > div { border-radius: 12px !important; overflow: hidden !important; }
.image-panel .wrap, .image-panel > .wrap {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important; min-height: 280px !important;
}

/* ── Status box ── */
.status-box { margin-top: 0.5rem !important; }
.status-box textarea {
    background: var(--surface2) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--success) !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
}

/* ── Expression buttons ── */
.expr-row { display: flex !important; gap: 8px !important; margin-top: 0.5rem !important; }
.expr-btn {
    flex: 1 !important;
    background: var(--surface3) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important; padding: 0.6rem 0.2rem !important;
    font-size: 1.3rem !important; cursor: pointer !important;
    transition: all 0.15s ease !important; display: flex !important;
    flex-direction: column !important; align-items: center !important;
    gap: 2px !important;
}
.expr-btn:hover {
    border-color: var(--primary-light) !important;
    background: #1e1040 !important; transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.3) !important;
}
.expr-label {
    font-size: 0.6rem !important; color: var(--muted) !important;
    font-weight: 500 !important; text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Lock button ── */
.lock-btn button {
    width: 100% !important; background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 1rem !important;
    padding: 0.75rem !important; letter-spacing: 0.02em !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important;
    transition: all 0.2s !important;
}
.lock-btn button:hover { box-shadow: 0 6px 28px rgba(124,58,237,0.55) !important; }

/* ── Detection result ── */
.detect-result {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; min-height: 100px;
}

/* ── Footer ── */
.footer {
    text-align: center; color: var(--muted); font-size: 0.78rem;
    padding: 1.5rem 0 0.5rem; border-top: 1px solid var(--border); margin-top: 2rem;
}
"""

with gr.Blocks(title="Synthetic Face Demo — Islas Nawaz") as demo:

    gr.HTML("""
        <div class="app-header">
            <h1>Synthetic Face Demo</h1>
            <p>Real-time deepfake demonstration &nbsp;·&nbsp; By <strong>Islas Nawaz</strong></p>
        </div>
        <div class="disclaimer">
            All outputs are watermarked &nbsp;|&nbsp; Educational use only &nbsp;|&nbsp;
            Only use faces you have explicit consent to use
        </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Live Swap ─────────────────────────────────────────────
        with gr.Tab("Live Face Swap"):
            with gr.Row(equal_height=False):

                # ── Left sidebar ──────────────────────────────────────────
                with gr.Column(scale=1, min_width=270):

                    gr.HTML('<div class="step-header"><span class="step-badge">1</span> Upload a face photo</div>')
                    source_photo = gr.Image(
                        type="pil", label="Target face",
                        sources=["upload"], height=240,
                        elem_classes=["image-panel"],
                    )

                    gr.HTML('<div class="step-header" style="margin-top:1.2rem"><span class="step-badge">2</span> Lock it in</div>')
                    lock_btn = gr.Button("Lock in face", variant="primary", elem_classes=["lock-btn"])
                    status = gr.Textbox(
                        label="Status", interactive=False,
                        placeholder="Upload a photo and click Lock in face...",
                        elem_classes=["status-box"], lines=1,
                    )
                    lock_btn.click(set_source, inputs=source_photo, outputs=status)

                    gr.HTML('<div class="step-header" style="margin-top:1.2rem"><span class="step-badge">3</span> Strike a pose</div>')
                    expr_state = gr.State("neutral")
                    gr.HTML("""
                        <div style="display:flex;gap:8px;margin-bottom:0.25rem">
                            <div id="eb-wrap" style="display:contents"></div>
                        </div>
                    """)
                    with gr.Row(elem_classes=["expr-row"]):
                        btn_neutral   = gr.Button("😐\nNeutral",   elem_classes=["expr-btn"])
                        btn_smile     = gr.Button("😄\nSmile",     elem_classes=["expr-btn"])
                        btn_angry     = gr.Button("😠\nAngry",     elem_classes=["expr-btn"])
                        btn_surprised = gr.Button("😮\nSurprised", elem_classes=["expr-btn"])
                        btn_wink      = gr.Button("😉\nWink",      elem_classes=["expr-btn"])

                    btn_neutral.click(  lambda: "neutral",   outputs=expr_state)
                    btn_smile.click(    lambda: "smile",     outputs=expr_state)
                    btn_angry.click(    lambda: "angry",     outputs=expr_state)
                    btn_surprised.click(lambda: "surprised", outputs=expr_state)
                    btn_wink.click(     lambda: "wink",      outputs=expr_state)

                # ── Right panel: webcam + result ──────────────────────────
                with gr.Column(scale=2):
                    gr.HTML('<div class="step-header"><span class="step-badge">4</span> Enable your webcam — swap happens live</div>')
                    with gr.Row():
                        webcam = gr.Image(
                            label="Your webcam",
                            sources=["webcam"], streaming=True,
                            type="numpy", height=340,
                            elem_classes=["image-panel"],
                        )
                        output = gr.Image(
                            label="Swapped result",
                            type="numpy", height=340,
                            elem_classes=["image-panel"],
                        )

            webcam.stream(
                process_frame,
                inputs=[webcam, expr_state],
                outputs=output,
                stream_every=0.1,
                time_limit=None,
            )

        # ── Tab 2: Detection ─────────────────────────────────────────────
        with gr.Tab("Deepfake Detection"):
            gr.HTML('<p style="color:var(--muted,#888);margin-bottom:1.2rem;font-size:0.95rem;">Upload any image to analyse whether it looks synthetic.</p>')
            with gr.Row():
                with gr.Column(scale=1):
                    detect_img = gr.Image(
                        type="pil", label="Image to analyse",
                        sources=["upload"], height=340,
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
    demo.queue().launch(css=CSS, theme=gr.themes.Base())
