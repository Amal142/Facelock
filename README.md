# 🔒 FaceLock

Real-time facial recognition for Python. Enrol a face, then authenticate against it live from your webcam.

---

## Quick Start

```bash
# 1 — Install dependencies
pip install -r requirements.txt

# 2 — Enrol your face (run once per person)
python enroll.py

# 3 — Run live authentication
python main.py
```

---

## Project Structure

```
Facelock/
├── modules/
│   ├── camera_handler.py   # Webcam via OpenCV
│   ├── face_detector.py    # Face detection via MediaPipe
│   ├── face_encoder.py     # 512-D embedding via InsightFace
│   ├── database.py         # Pickle-based profile storage
│   ├── authenticator.py    # Cosine-similarity matching
│   └── __init__.py
├── data/                   # Auto-created — stores profiles.pkl
├── enroll.py               # Face registration script
├── main.py                 # Live authentication loop
└── requirements.txt
```

---

## enroll.py

| Key     | Action                        |
|---------|-------------------------------|
| `SPACE` | Capture a sample frame        |
| `Q`     | Save profile & exit           |
| `ESC`   | Abort without saving          |

Capture **at least 10 frames** in varied poses. More samples = better accuracy.

---

## main.py

| Colour | Meaning                             |
|--------|-------------------------------------|
| 🟢 Green  | Recognised — shows name + score  |
| 🔴 Red    | Unknown face                     |
| 🟠 Orange | Face detected, waiting for encode|

Press **Q** to quit.

---

## Tuning the Threshold

In `modules/authenticator.py`, change `DEFAULT_THRESHOLD`:

| Value | Effect                              |
|-------|-------------------------------------|
| 0.50  | Strict — fewer false positives      |
| 0.40  | Balanced (default)                  |
| 0.30  | Lenient — catches more angles/light |

---

## Notes

- Face data is stored in `data/profiles.pkl` (plain pickle, no encryption).  
- To remove a user: `auth.delete_user("name")` or delete the `.pkl` file.  
- InsightFace downloads a ~100 MB model on first run — requires internet once.
