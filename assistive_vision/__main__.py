"""Temporary OpenCV preview for Phase 2 manual verification (no Tkinter)."""

import argparse
from math import isfinite
import sys
import time

from .camera import Webcam
from .announcements import AnnouncementPolicy
from .detector import YoloDetector
from .diagnostics import DebugReporter
from .pipeline import VisionPipeline
from .speech import SpeechService


def confidence_value(value: str) -> float:
    confidence = float(value)
    if not isfinite(confidence) or not 0 <= confidence <= 1:
        raise argparse.ArgumentTypeError("confidence must be between 0 and 1")
    return confidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 assistive vision preview")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--confidence", type=confidence_value, default=0.5,
                        help="minimum detector and speech confidence (default: 0.5)")
    parser.add_argument("--debug", action="store_true",
                        help="print one detection/speech summary per second")
    args = parser.parse_args()
    try:
        import cv2
    except ImportError as exc:
        print(f"OpenCV unavailable: {exc}. Install requirements inside .venv.", file=sys.stderr)
        return 1
    pipeline = VisionPipeline(
        Webcam(args.camera), YoloDetector(args.weights, confidence=args.confidence),
        SpeechService(debug=args.debug), AnnouncementPolicy(min_confidence=args.confidence),
    )
    pipeline.speech.set_muted(args.mute)
    paused, muted = False, args.mute
    reporter = DebugReporter(enabled=args.debug)
    pipeline.start()
    last_speech_warning = ""
    shown = False
    try:
        while pipeline.running:
            result = pipeline.latest()
            speech_status = pipeline.speech.status
            if "unavailable" in speech_status and speech_status != last_speech_warning:
                print(speech_status, file=sys.stderr)
                last_speech_warning = speech_status
            reporter.report(result, status=pipeline.status, speech_status=speech_status,
                            last_request=pipeline.speech.last_request)
            if result is not None:
                display = result.frame.copy()
                cv2.putText(display, f"{result.status} | {result.fps:.1f} FPS | "
                            f"{'Muted' if muted else result.speech_status}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Assistive Vision - Phase 2 | P pause | M mute | Q quit", display)
                shown = True
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                break
            if shown and cv2.getWindowProperty(
                "Assistive Vision - Phase 2 | P pause | M mute | Q quit", cv2.WND_PROP_VISIBLE
            ) < 1:
                break
            if key == ord("p"):
                paused = not paused
                pipeline.set_paused(paused)
            elif key == ord("m"):
                muted = not muted
                pipeline.speech.set_muted(muted)
            if result is None:
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Preview unavailable: {exc}", file=sys.stderr)
        return 1
    finally:
        pipeline.stop()
        pipeline.join()
        pipeline.speech.join()
        cv2.destroyAllWindows()
    failed = "unavailable" in pipeline.status or "failed" in pipeline.status
    if failed:
        print(pipeline.status, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
