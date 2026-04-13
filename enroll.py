"""
enroll.py — Register a new face profile.

Usage:
    python enroll.py

How it works:
    1. The webcam opens and shows a live preview.
    2. Each time a face is detected, a green box appears.
    3. Press SPACE to capture that frame as a sample.
    4. Capture at least 10 samples (more = better accuracy).
    5. Press Q when done — the script averages all samples and saves the profile.
"""

import sys
import cv2
import logging
from modules.camera_handler import CameraHandler
from modules.face_detector import FaceDetector
from modules.face_encoder import FaceEncoder
from modules.authenticator import Authenticator

logging.basicConfig(level=logging.WARNING)   # suppress debug noise

# ── Constants ──────────────────────────────────────────────────────────────── #
MIN_SAMPLES = 10
WINDOW      = "FaceLock — Enrolment  |  SPACE = capture  |  Q = save & quit"


def draw_hud(frame, boxes, n_captured, min_samples):
    """Overlay status information on the preview frame."""
    # Bounding boxes
    color = (0, 255, 0) if boxes else (0, 100, 255)
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Status bar background
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (30, 30, 30), -1)

    face_txt = f"Face: {'DETECTED' if boxes else 'NOT FOUND'}"
    face_col = (0, 220, 0) if boxes else (0, 80, 255)
    cv2.putText(frame, face_txt, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, face_col, 2)

    progress = f"Samples: {n_captured}/{min_samples}  (SPACE=capture  Q=save)"
    cv2.putText(frame, progress, (230, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    return frame


def main():
    # ── Ask for a name ─────────────────────────────────────────────────────── #
    print("\n=== FaceLock — Enrolment ===")
    user_id = input("Enter the name / ID for this profile: ").strip()
    if not user_id:
        print("Name cannot be empty. Exiting.")
        sys.exit(1)

    # ── Set up modules ─────────────────────────────────────────────────────── #
    cam      = CameraHandler()
    detector = FaceDetector()
    encoder  = FaceEncoder()
    auth     = Authenticator()

    if not cam.start():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    print(f"\nWebcam ready.  Enrolling as: '{user_id}'")
    print(f"→ Look at the camera, then press SPACE to capture a sample.")
    print(f"→ Capture at least {MIN_SAMPLES} samples, then press Q to save.\n")

    embeddings = []

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 800, 520)

    while True:
        frame = cam.read()
        if frame is None:
            print("ERROR: Failed to read frame.")
            break

        boxes = detector.detect(frame)
        display = draw_hud(frame.copy(), boxes, len(embeddings), MIN_SAMPLES)
        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF

        # ── Capture sample ──────────────────────────────────────────────── #
        if key == ord(" "):
            if not boxes:
                print("  [!] No face detected — move closer or improve lighting.")
                continue

            emb = encoder.encode(frame, hint_box=boxes[0])
            if emb is None:
                print("  [!] InsightFace could not encode the face. Try again.")
                continue

            embeddings.append(emb)
            n = len(embeddings)
            print(f"  [✓] Sample {n} captured.")

        # ── Save & exit ─────────────────────────────────────────────────── #
        elif key == ord("q"):
            if len(embeddings) < MIN_SAMPLES:
                print(f"  [!] Need at least {MIN_SAMPLES} samples "
                      f"(have {len(embeddings)}). Keep capturing.")
                continue

            success = auth.enroll(user_id, embeddings)
            if success:
                print(f"\n[✓] Profile '{user_id}' saved "
                      f"({len(embeddings)} samples averaged).")
                print("    You can now run  main.py  to authenticate.\n")
            else:
                print("\n[✗] Enrolment failed — no embeddings collected.")
            break

        # ── ESC — abort ─────────────────────────────────────────────────── #
        elif key == 27:
            print("\nEnrolment cancelled.")
            break

    cv2.destroyAllWindows()
    cam.stop()


if __name__ == "__main__":
    main()
