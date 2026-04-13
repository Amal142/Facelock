"""
Module 2: Face Detection
Uses OpenCV's built-in Haar cascade — no extra dependencies required.
"""

import cv2
import logging

logger = logging.getLogger(__name__)

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class FaceDetector:
    def __init__(self):
        self._cascade = cv2.CascadeClassifier(_CASCADE_PATH)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {_CASCADE_PATH}")
        logger.info("FaceDetector ready (OpenCV Haar cascade).")

    def detect(self, frame):
        """
        Return a list of bounding boxes (x, y, w, h) for every face found.
        Returns an empty list when no face is detected.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # improves detection under varied lighting

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        if len(faces) == 0:
            return []

        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def draw_boxes(self, frame, boxes, color=(0, 255, 0), thickness=2):
        """Draw bounding boxes on a frame (in-place)."""
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        return frame
