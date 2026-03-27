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
EXPRESSION_FILE  = Path("/tmp/synthetic_demo_expression.txt")


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


PROCESS_W = 640


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


def _read_expression():
    """Read the current expression from disk (works across Gradio worker processes)."""
    try:
        return EXPRESSION_FILE.read_text().strip()
    except Exception:
        return "neutral"


def _apply_expression(img_bgr, face, expression=None):
    """Apply expression warp to img_bgr using face.landmark_2d_106."""
    if expression is None:
        expression = _read_expression()
    if expression == "neutral":
        return img_bgr, expression
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None:
        return img_bgr, expression
    dst = _compute_expression_deltas(lm, expression)
    return _piecewise_affine_warp(img_bgr, lm, dst), expression


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


def _swap_photo(photo_rgb, expression):
    """
    Core swap pipeline for a single photo (no streaming).
    photo_rgb: numpy RGB image from webcam capture.
    expression: string e.g. "smile".
    Returns (result_rgb, status_str).
    """
    if not MODEL_OK:
        return None, f"Model error: {MODEL_ERROR}"
    source_face = _load_source()
    if source_face is None:
        return None, "No face locked in — upload a photo and click Lock in face first."

    bgr = cv2.cvtColor(photo_rgb, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = bgr.shape[:2]
    small = cv2.resize(bgr, (PROCESS_W, int(orig_h * PROCESS_W / orig_w)))

    faces = _detect_hd(small)
    if not faces:
        return None, "No face detected in your photo — try better lighting."

    result = small.copy()
    final_expr = expression
    for face in faces:
        result = _swapper.get(result, face, source_face, paste_back=True)
        result, final_expr = _apply_expression(result, face, expression)

    result = cv2.resize(result, (orig_w, orig_h))
    result = _draw_expr_label(result, final_expr)
    result = apply_visible(result)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB), "Done!"


def capture_and_swap(photo, expression):
    """Called by the Capture & Swap button."""
    if photo is None:
        return None, None, "Enable your webcam and capture a photo first."
    result, msg = _swap_photo(np.array(photo), expression)
    stored = photo if result is not None else None
    return result, stored, msg


def reswap(stored_photo, expression):
    """Re-apply swap with a different expression on the stored photo."""
    if stored_photo is None:
        return None, "Capture a photo first."
    result, msg = _swap_photo(np.array(stored_photo), expression)
    return result, msg


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
* { box-sizing: border-box; }
body, .gradio-container, .gradio-container > .main {
    background: #080810 !important;
    color: #ddddf0 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
footer { display: none !important; }
.tab-nav { border-bottom: 1px solid #1e1e32 !important; background: transparent !important; }
.tab-nav button { color: #606080 !important; font-weight: 500 !important; border-radius: 6px 6px 0 0 !important; padding: 0.6rem 1.4rem !important; }
.tab-nav button.selected { color: #a78bfa !important; border-bottom: 2px solid #7c3aed !important; background: transparent !important; }
button { background: #12121e !important; border: 1px solid #252540 !important; color: #ddddf0 !important; border-radius: 8px !important; transition: all 0.15s !important; }
button:hover { background: #1c1c30 !important; border-color: #7c3aed !important; }
/* primary */
button.primary { background: linear-gradient(135deg,#7c3aed,#4f28b8) !important; border: none !important; color: #fff !important; font-weight: 700 !important; box-shadow: 0 4px 18px rgba(124,58,237,0.45) !important; }
button.primary:hover { background: linear-gradient(135deg,#8b4cf0,#5f32c8) !important; box-shadow: 0 6px 26px rgba(124,58,237,0.6) !important; }
/* image panels */
.vid-panel > div, .img-panel > div { border-radius: 10px !important; overflow: hidden !important; }
/* textboxes */
textarea { background: #12121e !important; border: 1px solid #252540 !important; color: #ddddf0 !important; border-radius: 8px !important; }
/* detect result */
.detect-result { background: #12121e; border: 1px solid #252540; border-radius: 12px; padding: 1.5rem; min-height: 120px; }
/* pose toolbar */
#pose-bar { background: #0e0e1c; border: 1px solid #1e1e32; border-radius: 14px; padding: 1rem 1.2rem; margin-top: 1rem; }
#pose-bar button { min-height: 72px !important; border-radius: 10px !important; font-size: 1.5rem !important; background: #14142a !important; border: 1.5px solid #252545 !important; }
#pose-bar button:hover { background: #1e1840 !important; border-color: #a78bfa !important; transform: translateY(-2px) !important; box-shadow: 0 6px 18px rgba(124,58,237,0.3) !important; }
"""

UI_STYLES = """
<style>
.sfd-wrap { text-align:center; padding:2rem 1rem 1rem; border-bottom:1px solid #1a1a2e; margin-bottom:1.2rem; }
.sfd-wrap h1 { font-size:2.1rem; font-weight:800; letter-spacing:-0.5px; margin:0 0 0.25rem;
    background:linear-gradient(130deg,#7c3aed 0%,#c084fc 55%,#38bdf8 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.sfd-wrap p { color:#606080; font-size:0.88rem; margin:0 0 0.8rem; }
.sfd-pills { display:inline-flex; gap:0.6rem; flex-wrap:wrap; justify-content:center; }
.sfd-pill { background:#12122a; border:1px solid #2a2a50; border-radius:20px; padding:0.25rem 0.75rem;
    font-size:0.72rem; color:#8888c0; letter-spacing:0.05em; text-transform:uppercase; }
.ctrl-label { display:flex; align-items:center; gap:0.5rem; font-size:0.82rem; font-weight:600;
    color:#9090c0; text-transform:uppercase; letter-spacing:0.06em; margin:1rem 0 0.4rem; }
.ctrl-label:first-child { margin-top:0; }
.ctrl-num { display:inline-flex; align-items:center; justify-content:center; width:20px; height:20px;
    border-radius:50%; font-size:0.65rem; font-weight:800;
    background:linear-gradient(135deg,#7c3aed,#5b21b6); color:#fff; flex-shrink:0; }
.vid-label { font-size:0.75rem; font-weight:600; color:#606080; text-transform:uppercase;
    letter-spacing:0.07em; margin-bottom:0.4rem; }
.pose-label { font-size:0.72rem; color:#606080; text-align:center; margin-top:0.3rem;
    text-transform:uppercase; letter-spacing:0.04em; }
</style>
"""

with gr.Blocks(title="Synthetic Face Demo — Islas Nawaz") as demo:

    gr.HTML(UI_STYLES + """
    <div class="sfd-wrap">
        <h1>Synthetic Face Demo</h1>
        <p>Real-time deepfake pipeline &nbsp;&middot;&nbsp; By <strong style="color:#c4b5fd">Islas Nawaz</strong></p>
        <div class="sfd-pills">
            <span class="sfd-pill">&#x2713; Watermarked</span>
            <span class="sfd-pill">&#x2713; Educational only</span>
            <span class="sfd-pill">&#x2713; Consent required</span>
        </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Live Swap ─────────────────────────────────────────────
        with gr.Tab("Live Face Swap"):
            with gr.Row(equal_height=False):

                # ── Narrow controls sidebar ───────────────────────────────
                with gr.Column(scale=1, min_width=240):
                    gr.HTML('<div class="ctrl-label"><span class="ctrl-num">1</span> Target face</div>')
                    source_photo = gr.Image(
                        type="pil", label="", show_label=False,
                        sources=["upload"], height=210,
                        elem_classes=["img-panel"],
                    )
                    gr.HTML('<div class="ctrl-label"><span class="ctrl-num">2</span> Lock it in</div>')
                    lock_btn = gr.Button("🔒  Lock in face", variant="primary")
                    status = gr.Textbox(
                        label="", interactive=False, show_label=False,
                        placeholder="Upload a photo, then click Lock in face...",
                        lines=1,
                    )
                    lock_btn.click(set_source, inputs=source_photo, outputs=status)

                # ── Main video area ───────────────────────────────────────
                with gr.Column(scale=3):
                    expr_state    = gr.State("neutral")
                    stored_photo  = gr.State(None)

                    with gr.Row():
                        with gr.Column():
                            gr.HTML('<div class="vid-label">📷 Your webcam</div>')
                            webcam = gr.Image(
                                label="", show_label=False,
                                sources=["webcam"], type="pil",
                                height=400, elem_classes=["vid-panel"],
                            )
                            capture_btn = gr.Button(
                                "📸  Capture & Swap", variant="primary", size="lg"
                            )
                        with gr.Column():
                            gr.HTML('<div class="vid-label">✨ Swapped result</div>')
                            output = gr.Image(
                                label="", show_label=False,
                                type="numpy", height=400,
                                elem_classes=["vid-panel"],
                            )
                            swap_status = gr.Textbox(
                                label="", show_label=False, interactive=False,
                                placeholder="Result will appear here...", lines=1,
                            )

                    capture_btn.click(
                        capture_and_swap,
                        inputs=[webcam, expr_state],
                        outputs=[output, stored_photo, swap_status],
                    )

            # ── Pose toolbar — full width below videos ────────────────────
            gr.HTML('<div class="ctrl-label" style="margin-top:1rem"><span class="ctrl-num">3</span> Strike a pose (re-applies to captured photo)</div>')
            with gr.Row(elem_id="pose-bar"):
                btn_neutral   = gr.Button("😐\nNeutral")
                btn_smile     = gr.Button("😄\nSmile")
                btn_angry     = gr.Button("😠\nAngry")
                btn_surprised = gr.Button("😮\nSurprised")
                btn_wink      = gr.Button("😉\nWink")

            for btn, expr in [
                (btn_neutral, "neutral"), (btn_smile, "smile"),
                (btn_angry, "angry"), (btn_surprised, "surprised"),
                (btn_wink, "wink"),
            ]:
                btn.click(
                    fn=reswap,
                    inputs=[stored_photo, gr.State(expr)],
                    outputs=[output, swap_status],
                )

        # ── Tab 2: Detection ─────────────────────────────────────────────
        with gr.Tab("Deepfake Detection"):
            gr.HTML('<p style="color:#606080;margin:0.5rem 0 1rem;font-size:0.88rem;">Upload any image to check whether it looks AI-generated.</p>')
            with gr.Row():
                with gr.Column(scale=1):
                    detect_img = gr.Image(
                        type="pil", label="Image to analyse",
                        sources=["upload"], height=380,
                        elem_classes=["img-panel"],
                    )
                    detect_btn = gr.Button("🔍  Analyse image", variant="primary", size="lg")
                with gr.Column(scale=1):
                    detect_out = gr.Markdown(
                        value="*Upload an image and click Analyse.*",
                        elem_classes=["detect-result"],
                    )
            detect_btn.click(run_detect, inputs=detect_img, outputs=detect_out)

    gr.HTML('<div style="text-align:center;color:#30304a;font-size:0.74rem;padding:1.2rem 0 0.4rem;border-top:1px solid #141426;margin-top:1.5rem">Islas Nawaz &nbsp;·&nbsp; Responsible AI Education &nbsp;·&nbsp; All outputs watermarked</div>')

if __name__ == "__main__":
    demo.queue().launch(css=CSS, theme=gr.themes.Base())
