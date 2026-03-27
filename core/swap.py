"""
Face swap pipeline using InsightFace.

Usage:
    from core.swap import FaceSwapper
    swapper = FaceSwapper()
    result = swapper.swap(source_img, target_img)
"""

import cv2
import numpy as np
from pathlib import Path

# InsightFace is loaded lazily so the module imports cleanly even before
# model weights are downloaded.
_app = None
_swapper = None

MODEL_DIR = Path(__file__).parent.parent / "models"


def _load_models():
    global _app, _swapper
    if _app is not None:
        return

    import insightface
    from insightface.app import FaceAnalysis

    _app = FaceAnalysis(name="buffalo_l", root=str(MODEL_DIR))
    _app.prepare(ctx_id=0, det_size=(640, 640))

    swap_model = MODEL_DIR / "inswapper_128.onnx"
    if not swap_model.exists():
        raise FileNotFoundError(
            f"Swap model not found at {swap_model}.\n"
            "Download inswapper_128.onnx from the InsightFace model zoo and place it in models/."
        )
    _swapper = insightface.model_zoo.get_model(str(swap_model), download=False)


class FaceSwapper:
    def __init__(self):
        _load_models()

    def _detect(self, img_bgr: np.ndarray):
        """Return list of detected faces sorted by area (largest first)."""
        faces = _app.get(img_bgr)
        return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

    def swap(self, source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
        """
        Swap the largest face from source onto every face in target.

        Args:
            source_bgr: Source image (the face to copy from).
            target_bgr: Target image (where the face will be placed).

        Returns:
            Result image as BGR numpy array.
        """
        source_faces = self._detect(source_bgr)
        if not source_faces:
            raise ValueError("No face detected in the source image.")

        target_faces = self._detect(target_bgr)
        if not target_faces:
            raise ValueError("No face detected in the target image.")

        source_face = source_faces[0]
        result = target_bgr.copy()

        for face in target_faces:
            result = _swapper.get(result, face, source_face, paste_back=True)

        return result
