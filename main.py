"""
main.py — Real-time face authentication.

Usage:
    python main.py

How it works:
    1. The webcam opens.
    2. Each frame is scanned for a face.
    3. Any detected face is encoded and compared against enrolled profiles.
    4. The result (name + similarity score) is drawn on screen.
    5. Press Q to quit.

Run  enroll.py  first to register at least one face profile.
"""

import sys
import time
import cv2
import logging
from modules.camera_handler import CameraHandler
from modules.face_detector import FaceDetector
from modules.face_encoder import FaceEncoder
from modules.authenticator import Authenticator

logging.basicConfig(level=logging.WARNING)

# ── Config ─────────────────────────────────────────────────────────────────── #
WINDOW    = "FaceLock — Authentication  |  Q = quit"
FPS_ALPHA = 0.1   # smoothing for the FPS counter


# ── Drawing helpers ────────────────────────────────────────────────────────── #

def draw_result(frame, boxes, user_id, similarity):
    """
    Draw face boxes and the authentication result on the frame.

    Green  → matched user
    Red    → unknown face
    Orange → face detected, still processing
    """
    for (x, y, w, h) in boxes:
        if user_id:
            color = (0, 200, 50)
            label = f"{user_id}  ({similarity:.2f})"
        elif user_id is None and similarity > 0:
            color = (0, 60, 220)
            label = f"Unknown  ({similarity:.2f})"
        else:
            color = (0, 140, 255)
            label = "Detecting..."

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label background pill
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), color, -1)
        cv2.putText(frame, label, (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return frame


def draw_no_face(frame):
    """Dimmed overlay when no face is in view."""
    cv2.putText(frame, "No face detected", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
    return frame


# ── Main loop ──────────────────────────────────────────────────────────────── #

def main():
    cam      = CameraHandler()
    detector = FaceDetector()
    encoder  = FaceEncoder()
    auth     = Authenticator()

    # Warn if no profiles are registered yet
    users = auth.list_users()
    if not users:
        print("\n[!] No profiles enrolled.  Run  enroll.py  first.\n")
        sys.exit(1)

    print(f"\n=== FaceLock — Live Authentication ===")
    print(f"Enrolled users: {', '.join(users)}")
    print("Press Q to quit.\n")

    if not cam.start():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 800, 520)

    # State carried between frames so the label doesn't flicker
    last_id   = None
    last_sim  = 0.0
    fps       = 0.0
    prev_time = time.time()

    # Encode every N-th frame to keep the UI smooth (encoding is slow on CPU)
    ENCODE_EVERY = 3
    frame_idx    = 0

    while True:
        frame = cam.read()
        if frame is None:
            print("ERROR: Lost camera feed.")
            break

        # ── FPS counter ─────────────────────────────────────────────── #
        now  = time.time()
        fps  = FPS_ALPHA * (1.0 / max(now - prev_time, 1e-6)) + (1 - FPS_ALPHA) * fps
        prev_time = now

        # ── Detect ──────────────────────────────────────────────────── #
        boxes = detector.detect(frame)

        # ── Encode & identify (not every frame) ─────────────────────── #
        if boxes and frame_idx % ENCODE_EVERY == 0:
            emb = encoder.encode(frame, hint_box=boxes[0])
            if emb is not None:
                last_id, last_sim = auth.identify(emb)

        if not boxes:
            last_id, last_sim = None, 0.0   # reset when face leaves frame

        frame_idx += 1

        # ── Draw ────────────────────────────────────────────────────── #
        display = frame.copy()
        if boxes:
            draw_result(display, boxes, last_id, last_sim)
        else:
            draw_no_face(display)
        draw_fps(display, fps)

        cv2.imshow(WINDOW, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    cam.stop()
    print("Bye.")


if __name__ == "__main__":
    main()
