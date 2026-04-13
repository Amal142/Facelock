"""
Module 1: Camera Acquisition
Controls the webcam using OpenCV.
"""

import cv2
import logging

logger = logging.getLogger(__name__)


class CameraHandler:
    def __init__(self):
        self.cap = None
        self.is_open = False

    def start(self, camera_id=0):
        """Open the webcam stream."""
        self.cap = cv2.VideoCapture(camera_id)
        if self.cap.isOpened():
            self.is_open = True
            logger.info(f"Camera {camera_id} started.")
            return True
        logger.error(f"Could not open camera {camera_id}.")
        return False

    def read(self):
        """Return the current frame, or None on failure."""
        if self.is_open:
            ret, frame = self.cap.read()
            return frame if ret else None
        return None

    def stop(self):
        """Release the webcam."""
        if self.is_open:
            self.cap.release()
            self.is_open = False
            logger.info("Camera released.")
