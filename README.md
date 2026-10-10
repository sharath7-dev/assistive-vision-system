# Real-Time Assistive Vision System for the Visually Impaired

A Windows IPCV university project planned to recognize common objects from a
webcam, show labeled bounding boxes, and speak short object announcements.

## Current phase: testable core logic

Requires Python 3.11. No webcam, YOLO model, interface, or speech engine is
implemented yet. The runtime uses only Python's standard library.

- `assistive_vision/detections.py`: immutable `Detection` with a class label,
  confidence in `[0, 1]`, and `(left, top, right, bottom)` pixel coordinates.
  Blank labels, invalid confidence, and invalid boxes are rejected.
- `assistive_vision/announcements.py`: `AnnouncementPolicy` filters detections
  by confidence, deduplicates labels within a frame, and applies a per-label
  cooldown (default: 8 seconds). Different labels have independent cooldowns.
- `tests/`: deterministic pytest tests without camera, model, or audio access.

The policy uses monotonic time by default. Tests supply timestamps so they run
without waiting. Selecting a label immediately starts its cooldown; a future
speech adapter should call it only when speech is enabled and ready to accept
announcements. It tracks object classes, not identities of individual objects.

## Windows setup and tests (PowerShell)

Run from the repository root. These commands are provided for a later setup;
no dependencies have been installed as part of Phase 1.

```powershell
Set-Location "H:\IPCV\New folder\assistive-vision-system"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

If pytest is already installed in your selected Python environment:

```powershell
python -m pytest -q
```

There is no application launch command yet. Later phases will add capture and
YOLO11n inference, speech, then a responsive interface and hardware validation.
The prototype will not measure exact distances or guarantee safe navigation.
Webcam and audio operation have not been tested.
