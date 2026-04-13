"""
Module 3: Feature Extraction
Converts a face into a 512-dimensional embedding using InsightFace.

Important: always pass the FULL frame to encode(), not a cropped face.
InsightFace runs its own internal detector on the full image.
"""

import numpy as np
import logging
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


class FaceEncoder:
    def __init__(self):
        self._app = FaceAnalysis(providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("FaceEncoder ready (InsightFace).")

    def encode(self, full_frame, hint_box=None):
        """
        Return a 512-D numpy embedding for the most prominent face in the frame.

        Args:
            full_frame : BGR frame from OpenCV (the complete image).
            hint_box   : Optional (x, y, w, h) from MediaPipe to pick the
                         right face when multiple people are present.

        Returns:
            numpy array of shape (512,), or None if no face is found.
        """
        faces = self._app.get(full_frame)

        if not faces:
            logger.debug("No face found by InsightFace.")
            return None

        # If a hint box is given, pick the closest InsightFace detection.
        if hint_box is not None and len(faces) > 1:
            face = self._closest_face(faces, hint_box)
        else:
            face = faces[0]

        return face.embedding  # shape: (512,)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _closest_face(self, faces, hint_box):
        """Return the InsightFace detection nearest to hint_box centre."""
        hx, hy, hw, hh = hint_box
        cx, cy = hx + hw / 2, hy + hh / 2

        best, best_dist = faces[0], float("inf")
        for f in faces:
            bx1, by1, bx2, by2 = f.bbox
            fx, fy = (bx1 + bx2) / 2, (by1 + by2) / 2
            d = (cx - fx) ** 2 + (cy - fy) ** 2
            if d < best_dist:
                best, best_dist = f, d
        return best
