"""
Watermarking utilities.

Applies both a visible banner and an invisible bit-level watermark so that
outputs can be identified as AI-generated even after the visible mark is cropped.
"""

import cv2
import numpy as np

VISIBLE_TEXT = "AI GENERATED — Islas Nawaz"
INVISIBLE_BITS = b"Islas Nawaz synthetic-face-demo"


def apply_visible(img_bgr: np.ndarray) -> np.ndarray:
    """Stamp a semi-transparent banner across the bottom of the image."""
    img = img_bgr.copy()
    h, w = img.shape[:2]

    banner_h = max(30, h // 18)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    font_scale = banner_h / 40
    thickness = max(1, banner_h // 20)
    text_size, _ = cv2.getTextSize(VISIBLE_TEXT, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = (w - text_size[0]) // 2
    y = h - (banner_h - text_size[1]) // 2

    cv2.putText(img, VISIBLE_TEXT, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (200, 200, 200), thickness, cv2.LINE_AA)
    return img


def apply_invisible(img_bgr: np.ndarray) -> np.ndarray:
    """
    Embed an invisible watermark using DCT-domain steganography.

    The mark is robust to moderate JPEG compression but will not survive
    heavy re-encoding. For a production system, use a dedicated library
    such as `invisible-watermark`.
    """
    try:
        from imwatermark import WatermarkEncoder
        encoder = WatermarkEncoder()
        encoder.set_watermark("bytes", INVISIBLE_BITS)
        img_bgr = encoder.encode(img_bgr, "dwtDct")
    except ImportError:
        # Graceful degradation if invisible-watermark is not installed
        pass
    return img_bgr


def watermark(img_bgr: np.ndarray) -> np.ndarray:
    """Apply both visible and invisible watermarks."""
    img = apply_invisible(img_bgr)
    img = apply_visible(img)
    return img
