"""Face detection + alignment for arbitrary (unaligned) images and webcam
frames.

The trained model (src/models/mobilefacenet.py) only ever saw pre-aligned
112x112 CASIA-WebFace crops (see docs/DATASET.md) -- a raw photo or webcam
frame isn't aligned like that, so this module finds a face, crops it to a
square with some margin, and resizes to 112x112 before anything is handed
to the embedding model. Uses MediaPipe's lightweight BlazeFace detector
(224KB model, no GPU required, auto-downloaded on first use -- same
convenience as torchvision's pretrained-weight caching).

This is a best-effort approximation of whatever alignment protocol produced
the original training crops (which isn't documented) -- it won't be
pixel-identical, so expect somewhat lower real-world accuracy on casual
photos/webcam captures than the benchmark numbers in docs/RESULTS_baseline.md,
which were measured on properly pre-aligned data. The demo surfaces the
aligned crop it actually fed to the model, specifically so this gap is
visible rather than hidden.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_MODEL_CACHE_DIR = Path.home() / ".cache" / "facial-recognition-demo"
_MODEL_PATH = _MODEL_CACHE_DIR / "blaze_face_short_range.tflite"


class NoFaceDetected(Exception):
    pass


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading face detector model to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


class FaceAligner:
    def __init__(self, min_confidence: float = 0.5):
        model_path = _ensure_model()
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    def align(self, image: Image.Image, output_size: int = 112, margin: float = 0.35) -> Image.Image:
        """Detects the most prominent face and returns a square `output_size`
        crop centered on it, with `margin` extra context around the raw
        detection box (BlazeFace boxes are tight around eyes/nose/mouth, so
        some margin is needed to include the whole head)."""
        rgb_image = image.convert("RGB")
        rgb = np.array(rgb_image)
        h, w = rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        if not result.detections:
            raise NoFaceDetected("no face detected in image")

        best = max(result.detections, key=lambda d: d.categories[0].score)
        box = best.bounding_box

        cx = box.origin_x + box.width / 2
        cy = box.origin_y + box.height / 2
        side = max(box.width, box.height) * (1 + margin)

        x0 = int(max(0, cx - side / 2))
        y0 = int(max(0, cy - side / 2))
        x1 = int(min(w, cx + side / 2))
        y1 = int(min(h, cy + side / 2))

        crop = rgb_image.crop((x0, y0, x1, y1))
        return crop.resize((output_size, output_size), Image.LANCZOS)

    def close(self):
        self._detector.close()
