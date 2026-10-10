# Real-Time Assistive Vision System for the Visually Impaired

Windows / Python 3.11 IPCV university prototype: recognize common objects from
a webcam, display labeled boxes, and speak short announcements. Phase 3 adds
a dark Tkinter/ttk desktop dashboard around the existing background pipeline.
The OpenCV preview remains available using the same processing modules.

## Windows setup and launch (PowerShell)

Run from the project folder. Reuse the existing `.venv`; create it only if absent.
All packages are installed locally, without activating the environment.

```powershell
Set-Location "H:\IPCV\New folder\assistive-vision-system"
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m assistive_vision --gui
```

The GUI starts with the camera off. Select **Start camera**, then use **Stop
camera**, **Pause detection / Resume detection**, **Mute speech / Unmute
speech**, and **Quit application**. The confidence slider changes the detector
and announcement threshold on the next processing frame; it does not reload
the model. Left/Right arrow keys change a focused slider by 0.01.

The dashboard shows an aspect-preserving preview with status outside the
image, measured FPS, actual camera/detection/model states, and up to six
recent object classes from the last ten seconds. The recent list is history,
not a claim that those objects remain in the current scene, and never drives
speech. It has a scrollbar for compact windows. Tab/Shift+Tab navigate controls;
Ctrl+Enter starts/stops, P pauses/resumes, M mutes/unmutes, and Escape or Ctrl+Q
quits. Window closing follows the same shutdown path.

Stop/quit request shutdown and poll both workers without blocking Tkinter.
Starting again is disabled until camera and speech cleanup finish. Cached
model weights/model instance are reused across sessions. A native driver,
download, inference call, or active utterance may take time to return; the
window remains responsive while waiting rather than opening a duplicate session.
The minimum window size preserves readable controls (1000 x 860).

For the original OpenCV preview, omit `--gui`:

```powershell
.\.venv\Scripts\python.exe -m assistive_vision
```

That preview starts capture/detection automatically. With the preview focused,
press **P** to pause/resume detection (camera preview continues), **M** to
mute/unmute, and **Q** or Escape to quit. Closing the preview also quits.
Ctrl+C works in the terminal, including during initial model loading.

```powershell
# Another camera, initially muted
.\.venv\Scripts\python.exe -m assistive_vision --gui --camera 1 --mute
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
  without consuming cooldowns for labels that were not selected. The live
  scheduler observes every frame even while audio is busy, queues at most six
  current classes in arrival order, and serves one concise phrase at a time
  ("Person detected", "Chair detected", "Bottle detected"). Persistent classes
  are announced once per appearance, rather than every cooldown interval.
  Absence for at least one second establishes a new appearance; re-announcement
  also respects the eight-second class cooldown. Brief missed frames therefore
  do not cause chatter. Overflow classes enter as queue space becomes available.
- `camera.py`: OpenCV device opening, frame reads, and resource release.
- `detector.py`: lazy YOLO11n adapter returning standard `Detection` instances.
- `drawing.py`: copies frames and adds boxes, dark label backgrounds, and scores.
- `speech.py`: offline pyttsx3 engine on its own thread. It accepts at most one
  utterance at a time, drops submissions while busy, and expires pending speech
  after two seconds measured from the source frame observation, including
  inference time. Every processed frame refreshes valid labels, even while
  speech is busy. Disappeared labels are removed from pending announcements;
  empty detections, pause, and mute discard pending speech. Audio already
  handed to the engine may finish. A six-class scheduler queue plus one speech
  slot bounds pending work. Engine-handoff/cancellation receipts ensure that
  cancelled or rejected requests do not consume an announcement or cooldown.
  Muting keeps current class context fresh; unmuting serves currently valid,
  unannounced classes, not disappeared history.
- `pipeline.py`: background capture/inference/drawing worker. `start()`,
  `set_paused()`, `stop()`, and `latest()` support the desktop GUI without putting
  inference on its main thread. Only the latest frame is stored. FPS measures
  processing throughput, not camera hardware FPS or GUI refresh rate.
- `desktop_logic.py`: bounded recent-detection history, preview sizing, and
  single-pipeline session controls, tested without Tkinter or hardware.
- `gui.py`: Tkinter/ttk dashboard; all widget/image updates use the main thread
  through `after()`. It reads the existing latest-frame mailbox.
- `__main__.py`: shared launcher for the GUI and original OpenCV preview.
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
.\.venv\Scripts\python.exe -m assistive_vision --gui --debug --confidence 0.5
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
   not replay on resume/unmute. Cooldown starts at actual engine handoff;
   cancelled pending requests do not consume it. Classes already announced
   remain quiet while continuously visible.
4. Compare the speech request age and current detection output before deciding
   whether a remaining issue is a model label, processing delay, or audio delay.
   Do not increase confidence without checking that evidence first.

The disappearance and expiry fixes are verified with mocks. They do not prove
the cause of any specific webcam observation or eliminate YOLO false positives.

### Speech driver and label verification

On the installed Windows pyttsx3 2.99 driver, repeated `runAndWait()` calls
completed the first phrase but cancelled subsequent phrases. Windows speech
now uses one persistent external event loop on the existing speech worker.
Completion failures and a ten-second timeout become visible speech errors.
Three successive phrases were checked through the real driver and application
worker; callback completion does not independently verify audible speaker output.

The current YOLO11n weights contain 80 classes, including `cell phone`, `chair`,
`couch`, and `bed`, but no `pen` class. An actual pen may be missed or incorrectly
classified. Drawing and speech preserve the returned class name; a confidence
threshold cannot guarantee correct classification. The default remains 0.5.

A local webcam check captured 25 frames at that threshold and returned person
and bed detections, without reproducing the reported cell-phone label. Ignored
local evidence is in `.local/verification/` (JSON scores/boxes and annotated
images); no pen-specific confidence or box was established. To verify the
reported case, show the pen clearly with `--gui --debug`, compare the current
box label and confidence with the spoken phrase, and record that frame. Tests
cover unchanged labels, scores and boxes through filtering, drawing and
announcement selection without requiring hardware.

## Phase 3 verification

Automated tests exercise dashboard sizing/history/session logic, confidence
handoff, pipeline states, and shared GUI launch without a physical device or
model downloads. A separate UI-only check initialized/resized the Tkinter
window and displayed a synthetic annotated frame; this is not webcam, actual
YOLO inference, or audio hardware testing.

Before presenting, manually verify live boxes and speech, threshold changes,
repeated start/stop without camera conflicts, pause/resume, mute/unmute,
keyboard focus/shortcuts, window resizing, model-loading and missing-camera
messages, and quit while detection/speech is active. Pillow is the lightweight
Tkinter image bridge and is explicitly declared; it was already installed as
an Ultralytics dependency. Tkinter comes with the normal Windows Python installer.
