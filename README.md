# Real-Time Assistive Vision System for the Visually Impaired

Windows / Python 3.11 IPCV university prototype: recognize common objects from
a webcam, display labeled boxes, and speak short announcements. Phase 2 adds
the processing pipeline and a temporary OpenCV preview. A polished Tkinter
interface is reserved for Phase 3.

## Windows setup and launch (PowerShell)

Run from the project folder. Reuse the existing `.venv`; create it only if absent.
All packages are installed locally, without activating the environment.

```powershell
Set-Location "H:\IPCV\New folder\assistive-vision-system"
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m assistive_vision
```

The preview starts capture/detection automatically. With the preview focused,
press **P** to pause/resume detection (camera preview continues), **M** to
mute/unmute, and **Q** or Escape to quit. Closing the preview also quits.
Ctrl+C works in the terminal, including during initial model loading.

```powershell
# Another camera, initially muted
.\.venv\Scripts\python.exe -m assistive_vision --camera 1 --mute
# Existing local pretrained detection weights, including offline operation
.\.venv\Scripts\python.exe -m assistive_vision --weights .\yolo11n.pt
```

YOLO11n is loaded lazily on the first unpaused detection. If the default
`yolo11n.pt` is absent, Ultralytics attempts to download it. Internet access is
needed for that first download; afterward the local weights can be reused.
Unit tests never load actual models, download weights, or access hardware.
Inference explicitly uses **CPU**, confidence threshold 0.5, and image size
416. Use `--confidence 0.5` to configure the same threshold for detection and
speech; the default has not been increased to mask false positives. Scores
are model confidence values and do not guarantee that a label is correct.
PyTorch and related packages are required dependencies of Ultralytics;
installation is substantially larger than the nano model itself.

## How the modules fit together

- `detections.py`: Phase 1 validated, immutable label/confidence/pixel-box data.
- `announcements.py`: Phase 1 per-label cooldown (8 seconds), confidence filter,
  and frame deduplication. An optional label limit supports short utterances
  without consuming cooldowns for labels that were not selected.
- `camera.py`: OpenCV device opening, frame reads, and resource release.
- `detector.py`: lazy YOLO11n adapter returning standard `Detection` instances.
- `drawing.py`: copies frames and adds boxes, dark label backgrounds, and scores.
- `speech.py`: offline pyttsx3 engine on its own thread. It accepts at most one
  utterance at a time, drops submissions while busy, and expires pending speech
  after two seconds measured from the source frame observation, including
  inference time. Every processed frame refreshes valid labels, even while
  speech is busy. Disappeared labels are removed from pending announcements;
  empty detections, pause, and mute discard pending speech. Audio already
  handed to the engine may finish.
- `pipeline.py`: background capture/inference/drawing worker. `start()`,
  `set_paused()`, `stop()`, and `latest()` support a future GUI without putting
  inference on its main thread. Only the latest frame is stored. FPS measures
  processing throughput, not camera hardware FPS or GUI refresh rate.
- `__main__.py`: temporary OpenCV preview and terminal status messages.
- `tests/`: core and mocked adapter/concurrency tests, preserving Phase 1 tests.

GUI clients should call controls from one main thread and read `latest()`
without mutating its frame. `stop()` is nonblocking; `join()` is available for
bounded shutdown waits. Native camera calls, model download/inference, and an
active speech utterance cannot be forcibly interrupted. Workers are daemon
threads as a fallback if a native dependency stalls during exit.

## Errors and manual verification

- Camera opening/reading errors stop the processing worker with an actionable
  status. Check **Windows Settings > Privacy & security > Camera**, enable
  desktop-app access, close other camera apps, and try `--camera 1`.
- Missing dependencies, unavailable weights, or inference errors produce a
  status message and release the camera. Fix the reported problem and restart.
- Speech initialization/playback failure disables announcements while video
  continues. Check Windows output device, volume, and installed speech voices.
- Run commands from this folder to keep downloaded weights here. Ultralytics
  configuration is directed to ignored `.local/ultralytics` unless a
  `YOLO_CONFIG_DIR` was already configured by the user. Model files are ignored.

Before a demo, manually verify camera opening, initial weight download, boxes
and scores on familiar objects, FPS on your laptop, speech without repetition,
pause/resume, mute/unmute, quit, and offline restart with local weights. Also
try an unavailable camera index to confirm the error message. Webcam, actual
model inference, and audio hardware have **not** been verified by unit tests.

This prototype does not measure exact distances or guarantee safe navigation.
Labels describe object classes rather than tracking individual identities.
There is no model training, paid API, database, Docker, or cloud inference.

API references: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/),
[prediction results](https://docs.ultralytics.com/modes/predict/), and
[pyttsx3](https://pyttsx3.readthedocs.io/en/latest/).

## Investigating an unexpected announcement

```powershell
.\.venv\Scripts\python.exe -m assistive_vision --debug --confidence 0.5
```

With `--debug`, the terminal prints at most one concise summary per second:
current labels with their highest confidence, FPS, frame number/age, and live
speech state. Crowded scenes show up to five distinct labels. The last speech
request and its source observation age are included when available; this is
a historical request, not a current detection or confirmation of audible
playback. Summaries sample the latest state and do not record every frame.
Normal mode is quiet apart from important warnings and errors. Initial model
downloads may still show messages from Ultralytics. The matching processed
frame is published before new speech submission.

1. With nobody visible, inspect the preview and terminal for a `person` box.
   If the current model result labels part of the empty scene as a person,
   that is a model false positive; record the preview box, score, scene, and lighting.
2. Enter the scene, then leave. New empty detection results must remove pending
   person announcements. An utterance already started may finish, and camera
   buffering or slow inference can mean the processed scene lags real life.
3. Repeat pause/resume and mute/unmute while leaving. Pending speech should
   not replay on resume/unmute. Cooldown still records selected announcements,
   including ones subsequently cancelled; cancellation does not reset it.
4. Compare the speech request age and current detection output before deciding
   whether a remaining issue is a model label, processing delay, or audio delay.
   Do not increase confidence without checking that evidence first.

The disappearance and expiry fixes are verified with mocks. They do not prove
the cause of any specific webcam observation or eliminate YOLO false positives.
