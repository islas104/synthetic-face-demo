"""
Lip-sync pipeline: text → speech → talking-face video using Wav2Lip.

Model weights required: models/wav2lip.pth
Download from: https://github.com/Rudrabha/Wav2Lip (see Releases)
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import librosa

# ─── Audio constants (must match Wav2Lip training) ───────────────────────────
_SR          = 16000
_N_FFT       = 800
_HOP         = 200
_WIN         = 800
_N_MELS      = 80
_REF_DB      = 20
_MIN_DB      = -100
_PREEMPH     = 0.97
_MAX_ABS     = 4.0

# ─── Inference constants ─────────────────────────────────────────────────────
_FPS          = 25
_MEL_STEP     = 16   # mel frames per inference step
_IMG          = 96   # face crop size (px)
_BATCH        = 128  # frames per forward pass


# ─── Model Architecture ───────────────────────────────────────────────────────

class _C(nn.Module):
    """Conv2d + BN + ReLU, with optional residual."""
    def __init__(self, ci, co, k, s, p, residual=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(ci, co, k, stride=s, padding=p),
            nn.BatchNorm2d(co),
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class _CT(nn.Module):
    """ConvTranspose2d + BN + ReLU."""
    def __init__(self, ci, co, k, s, p, op=0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(ci, co, k, stride=s, padding=p, output_padding=op),
            nn.BatchNorm2d(co),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv_block(x))


class Wav2LipModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Face encoder: (B, 6, 96, 96) → (B, 512, 1, 1)
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(_C(6, 16, 7, 1, 3)),                              # → 96x96
            nn.Sequential(_C(16, 32, 3, 2, 1),                              # → 48x48
                          _C(32, 32, 3, 1, 1, residual=True),
                          _C(32, 32, 3, 1, 1, residual=True)),
            nn.Sequential(_C(32, 64, 3, 2, 1),                              # → 24x24
                          _C(64, 64, 3, 1, 1, residual=True),
                          _C(64, 64, 3, 1, 1, residual=True),
                          _C(64, 64, 3, 1, 1, residual=True)),
            nn.Sequential(_C(64, 128, 3, 2, 1),                             # → 12x12
                          _C(128, 128, 3, 1, 1, residual=True),
                          _C(128, 128, 3, 1, 1, residual=True)),
            nn.Sequential(_C(128, 256, 3, 2, 1),                            # → 6x6
                          _C(256, 256, 3, 1, 1, residual=True),
                          _C(256, 256, 3, 1, 1, residual=True)),
            nn.Sequential(_C(256, 512, 3, 2, 1),                            # → 3x3
                          _C(512, 512, 3, 1, 1, residual=True)),
            nn.Sequential(_C(512, 512, 3, 1, 0),                            # → 1x1
                          _C(512, 512, 1, 1, 0)),
        ])

        # Audio encoder: (B, 1, 80, 16) → (B, 512, 1, 1)
        self.audio_encoder = nn.Sequential(
            _C(1,   32,  3, 1,      1),
            _C(32,  32,  3, 1,      1, residual=True),
            _C(32,  32,  3, 1,      1, residual=True),
            _C(32,  64,  3, (3, 1), 1),                # H: 80→27
            _C(64,  64,  3, 1,      1, residual=True),
            _C(64,  64,  3, 1,      1, residual=True),
            _C(64,  128, 3, 3,      1),                 # H: 27→9, W: 16→6
            _C(128, 128, 3, 1,      1, residual=True),
            _C(128, 128, 3, 1,      1, residual=True),
            _C(128, 256, 3, (3, 2), 1),                # H: 9→3, W: 6→3
            _C(256, 256, 3, 1,      1, residual=True),
            _C(256, 512, 3, 1,      0),                 # → 1x1
            _C(512, 512, 1, 1,      0),
        )

        # Face decoder: skip-connections from encoder, (B,512,1,1) → (B,3,96,96)
        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(_C(512,  512, 1, 1, 0)),                          # 1x1 + feat[6]=512 → 1024
            nn.Sequential(_CT(1024, 512, 3, 1, 0),                          # 1→3
                          _C(512, 512, 3, 1, 1, residual=True)),            # + feat[5]=512 → 1024
            nn.Sequential(_CT(1024, 512, 3, 2, 1, op=1),                    # 3→6
                          _C(512, 512, 3, 1, 1, residual=True),
                          _C(512, 512, 3, 1, 1, residual=True)),            # + feat[4]=256 → 768
            nn.Sequential(_CT(768,  384, 3, 2, 1, op=1),                    # 6→12
                          _C(384, 384, 3, 1, 1, residual=True),
                          _C(384, 384, 3, 1, 1, residual=True)),            # + feat[3]=128 → 512
            nn.Sequential(_CT(512,  256, 3, 2, 1, op=1),                    # 12→24
                          _C(256, 256, 3, 1, 1, residual=True),
                          _C(256, 256, 3, 1, 1, residual=True)),            # + feat[2]=64  → 320
            nn.Sequential(_CT(320,  128, 3, 2, 1, op=1),                    # 24→48
                          _C(128, 128, 3, 1, 1, residual=True),
                          _C(128, 128, 3, 1, 1, residual=True)),            # + feat[1]=32  → 160
            nn.Sequential(_CT(160,   64, 3, 2, 1, op=1),                    # 48→96
                          _C(64,  64,  3, 1, 1, residual=True),
                          _C(64,  64,  3, 1, 1, residual=True)),            # + feat[0]=16  → 80
        ])

        self.output_block = nn.Sequential(
            _C(80, 32, 3, 1, 1),
            nn.Conv2d(32, 3, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, audio_seq, face_seq):
        # audio_seq: (B, 1, 80, 16)   face_seq: (B, 6, 96, 96)
        audio_emb = self.audio_encoder(audio_seq)   # (B, 512, 1, 1)

        feats = []
        x = face_seq
        for block in self.face_encoder_blocks:
            x = block(x)
            feats.append(x)

        x = audio_emb
        for block in self.face_decoder_blocks:
            x = block(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception:
                pass
            feats.pop()

        return self.output_block(x)  # (B, 3, 96, 96)


# ─── Globals ──────────────────────────────────────────────────────────────────

_model  = None
_device = None


def is_available() -> bool:
    """True if wav2lip.pth exists in models/."""
    return (Path(__file__).parent.parent / "models" / "wav2lip.pth").exists()


def _load():
    global _model, _device
    if _model is not None:
        return True
    weights = Path(__file__).parent.parent / "models" / "wav2lip.pth"
    if not weights.exists():
        return False
    # Force CPU — TorchScript GAN checkpoint was compiled for CUDA/CPU, not MPS
    _device = "cpu"
    try:
        # TorchScript archive (wav2lip_gan.pth distributed as compiled model)
        _model = torch.jit.load(str(weights), map_location=_device).eval()
    except Exception:
        # Regular state-dict checkpoint
        ckpt  = torch.load(str(weights), map_location=_device, weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        m = Wav2LipModel()
        m.load_state_dict(state)
        _model = m.to(_device).eval()
    return True


# ─── Audio ───────────────────────────────────────────────────────────────────

def _mel(wav_path: str) -> np.ndarray:
    wav, _ = librosa.load(wav_path, sr=_SR)
    wav = scipy.signal.lfilter([1, -_PREEMPH], [1], wav)
    D = librosa.stft(wav, n_fft=_N_FFT, hop_length=_HOP, win_length=_WIN)
    mel_basis = librosa.filters.mel(sr=_SR, n_fft=_N_FFT, n_mels=_N_MELS, norm=None)
    amp = np.dot(mel_basis, np.abs(D))
    # Wav2Lip was trained on audio where STFT amplitudes are in the 100-10000 range
    # (matching raw PCM scale). Librosa normalises to [-1,1], so we rescale here.
    amp = amp * 32768.0
    db  = _MIN_DB + 20 * np.log10(np.maximum(1e-5, amp))
    S   = db - _REF_DB
    return np.clip(2 * _MAX_ABS * ((S - _MIN_DB) / -_MIN_DB) - _MAX_ABS,
                   -_MAX_ABS, _MAX_ABS)   # (80, T)


def _tts(text: str, lang: str = "en") -> str:
    from gtts import gTTS
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        path = f.name
    gTTS(text=text, lang=lang).save(path)
    return path


# ─── Face helpers ─────────────────────────────────────────────────────────────

def _face_bbox(bgr: np.ndarray, face_app) -> tuple | None:
    faces = face_app.get(bgr)
    if not faces:
        return None
    best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    h, w = bgr.shape[:2]
    x1, y1 = max(0, int(best.bbox[0])), max(0, int(best.bbox[1]))
    x2, y2 = min(w, int(best.bbox[2])), min(h, int(best.bbox[3]))
    return y1, y2, x1, x2


def _face_tensor(crop_bgr: np.ndarray) -> np.ndarray:
    """Return (6, 96, 96) float32 tensor: [masked_face | reference_face]."""
    face = cv2.resize(crop_bgr, (_IMG, _IMG)).astype(np.float32) / 255.0
    masked = face.copy()
    masked[_IMG // 2:] = 0.0                          # zero lower half (mouth)
    combined = np.concatenate([masked, face], axis=2) # (96, 96, 6) BGR
    return combined.transpose(2, 0, 1)                 # (6, 96, 96)


# ─── Public API ───────────────────────────────────────────────────────────────

def lipsync(face_rgb: np.ndarray, text: str, face_app,
            lang: str = "en") -> tuple[str | None, str]:
    """
    Generate a talking-face video.

    Parameters
    ----------
    face_rgb  : RGB numpy image (H, W, 3)
    text      : words to speak
    face_app  : InsightFace FaceAnalysis instance for face detection
    lang      : BCP-47 language code for gTTS (default 'en')

    Returns
    -------
    (video_path | None, status_message)
    """
    if not _load():
        return None, (
            "Wav2Lip model not found. "
            "Download wav2lip.pth from the Wav2Lip GitHub releases "
            "and place it in the models/ directory."
        )

    bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

    coords = _face_bbox(bgr, face_app)
    if coords is None:
        return None, "No face detected in the image — try the Capture & Swap step first."
    y1, y2, x1, x2 = coords
    crop = bgr[y1:y2, x1:x2]

    # TTS → mel spectrogram
    try:
        audio_path = _tts(text, lang)
    except Exception as e:
        return None, f"Text-to-speech failed (internet required): {e}"
    try:
        mel = _mel(audio_path)                       # (80, T)
    except Exception as e:
        return None, f"Audio processing failed (ffmpeg may be needed): {e}"

    # Frame count from audio length
    mel_per_frame = _SR / _FPS / _HOP               # ≈ 3.2 mel frames per video frame
    num_frames    = max(1, int(mel.shape[1] / mel_per_frame))

    face_t = _face_tensor(crop)                      # (6, 96, 96)

    img_batch, mel_batch, out_frames = [], [], []

    for i in range(num_frames):
        ms = int(i * mel_per_frame)
        me = ms + _MEL_STEP
        chunk = mel[:, ms:me] if me <= mel.shape[1] else \
                np.pad(mel[:, ms:], ((0, 0), (0, me - mel.shape[1])))
        img_batch.append(face_t)
        mel_batch.append(chunk)

        if len(img_batch) == _BATCH or i == num_frames - 1:
            img_t = torch.FloatTensor(np.array(img_batch)).to(_device)   # (B, 6, 96, 96)
            mel_t = torch.FloatTensor(np.array(mel_batch)                 # (B, 80, 16)
                                      ).unsqueeze(1).to(_device)          # (B, 1, 80, 16)
            with torch.no_grad():
                pred = _model(mel_t, img_t).cpu().numpy()                 # (B, 3, 96, 96)

            pred = (pred.transpose(0, 2, 3, 1) * 255).astype(np.uint8)   # (B, 96, 96, 3)
            fw, fh = x2 - x1, y2 - y1
            for p in pred:
                frame = bgr.copy()
                frame[y1:y2, x1:x2] = cv2.resize(p, (fw, fh))
                out_frames.append(frame)

            img_batch, mel_batch = [], []

    # Apply visible watermark text on each frame
    from core.watermark import apply_visible
    out_frames = [apply_visible(f) for f in out_frames]

    # Write MP4
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = f.name
    h, w = bgr.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), _FPS, (w, h))
    for f in out_frames:
        writer.write(f)
    writer.release()

    return out_path, f"Done — {num_frames} frames at {_FPS} fps."
