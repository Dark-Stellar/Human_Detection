"""Real-time person detection and tracking with optional age/expression analysis."""
from __future__ import annotations

import argparse
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
WINDOW_NAME = "Person Detection System"

PERSON_CLASS_ID = 0
DEFAULT_CONF_THRESHOLD = 0.5
CONF_STEP, CONF_MIN, CONF_MAX = 0.05, 0.1, 0.9

TRACK_TIMEOUT_S = 5.0
TRACK_HISTORY = 20
MIN_MATCH_SCORE = 0.3
CENTROID_NORM_PX = 100.0
TRAIL_LENGTH = 10

CONF_BAR_HEIGHT = 8          # thickness of the confidence bar (px)
CONF_BAR_FULL_LABEL_W = 110  # boxes narrower than this get a short label

PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (128, 255, 0), (255, 0, 128),
]

HELP_LINES = [
    "CONTROLS:",
    "Q: Quit | H: HOG | Y: YOLO | A: Haar",
    "T: Trajectory | C: Confidence | S: Stats",
    "R: Reset tracking | +/-: Confidence threshold",
]

METHOD_KEYS = {"h": "hog", "y": "yolo", "a": "haar"}


# --------------------------------------------------------------------------- #
# Data types and geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    w: int
    h: int
    confidence: float

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def center(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


def iou(a: Detection, b: Detection) -> float:
    """Intersection over Union of two detections."""
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Person detectors (common interface)
# --------------------------------------------------------------------------- #
class Detector(ABC):
    name: str
    conf_threshold: float = DEFAULT_CONF_THRESHOLD  # only used by YOLO

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        ...


class YoloDetector(Detector):
    name = "yolo"

    def __init__(self, cfg: Path, weights: Path,
                 nms_threshold: float = 0.4, input_size: Tuple[int, int] = (416, 416)):
        self.net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_layers = self.net.getUnconnectedOutLayersNames()  # computed once
        self.nms_threshold = nms_threshold
        self.input_size = input_size

    @classmethod
    def load(cls, directory: Path = BASE_DIR) -> Optional["YoloDetector"]:
        """Load local YOLOv3 files; return None if unavailable. Never downloads."""
        cfg, weights = directory / "yolov3.cfg", directory / "yolov3.weights"
        if not (cfg.is_file() and weights.is_file()):
            print("Local YOLOv3 files not found; YOLO unavailable.")
            return None
        try:
            detector = cls(cfg, weights)
            print("YOLO model loaded successfully.")
            return detector
        except cv2.error as exc:
            print(f"Error loading YOLO model: {exc}")
            return None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, self.input_size,
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        scale = np.array([width, height, width, height])
        boxes, confidences = [], []
        for output in outputs:
            is_person = output[:, 5:].argmax(axis=1) == PERSON_CLASS_ID
            confident = output[:, 5 + PERSON_CLASS_ID] > self.conf_threshold
            for row in output[is_person & confident]:
                cx, cy, w, h = row[:4] * scale
                boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
                confidences.append(float(row[5 + PERSON_CLASS_ID]))

        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxes(boxes, confidences,
                                self.conf_threshold, self.nms_threshold)
        return [Detection(*boxes[i], confidences[i]) for i in np.array(keep).flatten()]


class HogDetector(Detector):
    name = "hog"

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> List[Detection]:
        boxes, weights = self.hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(32, 32),
            scale=1.05, hitThreshold=2.0, groupThreshold=2)
        weights = np.asarray(weights).reshape(-1)
        return [Detection(int(x), int(y), int(w), int(h), float(np.clip(wt / 2.0, 0, 1)))
                for (x, y, w, h), wt in zip(boxes, weights)]


class HaarDetector(Detector):
    name = "haar"

    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_fullbody.xml")  # loaded once
        if self.cascade.empty():
            raise RuntimeError("Could not load Haar full-body cascade")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        return [Detection(int(x), int(y), int(w), int(h), 0.5) for x, y, w, h in bodies]


def build_detectors() -> Dict[str, Detector]:
    detectors: Dict[str, Detector] = {"hog": HogDetector()}
    try:
        detectors["haar"] = HaarDetector()
    except RuntimeError as exc:
        print(f"Haar detector disabled: {exc}")
    yolo = YoloDetector.load()
    if yolo:
        detectors["yolo"] = yolo
    return detectors


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
@dataclass
class Track:
    id: int
    detection: Detection
    last_seen: float
    history: Deque[Tuple[int, int]] = field(
        default_factory=lambda: deque(maxlen=TRACK_HISTORY))

    def update(self, detection: Detection, now: float) -> None:
        self.detection = detection
        self.last_seen = now
        self.history.append(detection.center)


class Tracker:
    """Greedy one-to-one matching of detections to existing tracks."""

    def __init__(self):
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    @staticmethod
    def _score(a: Detection, b: Detection) -> float:
        distance = math.dist(a.center, b.center) / CENTROID_NORM_PX
        return 0.7 * iou(a, b) + 0.3 * max(0.0, 1.0 - distance)

    def update(self, detections: List[Detection]) -> List[Track]:
        """Return the tracks that were matched or created this frame."""
        now = time.monotonic()

        # Score every (track, detection) pair, best first.
        pairs = sorted(
            ((self._score(t.detection, d), t.id, i)
             for t in self.tracks.values() for i, d in enumerate(detections)),
            reverse=True)

        used_tracks, used_dets, active = set(), set(), []
        for score, track_id, det_idx in pairs:
            if score <= MIN_MATCH_SCORE:
                break
            if track_id in used_tracks or det_idx in used_dets:
                continue
            used_tracks.add(track_id)
            used_dets.add(det_idx)
            track = self.tracks[track_id]
            track.update(detections[det_idx], now)
            active.append(track)

        for i, detection in enumerate(detections):
            if i not in used_dets:
                track = Track(self._next_id, detection, now)
                track.history.append(detection.center)
                self.tracks[track.id] = track
                self._next_id += 1
                active.append(track)

        self.tracks = {tid: t for tid, t in self.tracks.items()
                       if now - t.last_seen <= TRACK_TIMEOUT_S}
        return active


# --------------------------------------------------------------------------- #
# Face attributes (age + expression) in a background thread
# --------------------------------------------------------------------------- #
class FaceAttributes:
    def __init__(self, interval: float = 1.0):
        from deepface import DeepFace  # optional dependency

        print("Loading age and expression models (first run may download weights)...",
              flush=True)
        for model in ("Age", "Emotion"):
            DeepFace.build_model(model, task="facial_attribute")
        print("Face analysis ready.", flush=True)

        self._analyze = DeepFace.analyze
        self.interval = interval
        self.status = "Face analysis: waiting for first result"
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._pending: Optional[Future] = None
        self._started = float("-inf")
        self._result_time = float("-inf")
        self._results: list = []
        self._disabled = False

    def _predict(self, frame: np.ndarray) -> list:
        try:
            result = self._analyze(frame, actions=("age", "emotion"),
                                   detector_backend="opencv", enforce_detection=True,
                                   align=True, silent=True)
        except ValueError as exc:
            if "face could not be detected" in str(exc).lower():
                return []
            raise
        return [result] if isinstance(result, dict) else result

    def _collect_finished(self, now: float) -> None:
        if self._pending is None or not self._pending.done():
            return
        try:
            self._results = self._pending.result()
            self._result_time = now
            self.status = ("Face analysis: active" if self._results
                           else "Face analysis: no clear face")
        except Exception as exc:
            print(f"Face analysis disabled: {exc}")
            self._disabled = True
            self._results = []
            self.status = "Face analysis failed; see terminal"
        self._pending = None

    def update(self, frame: np.ndarray) -> list:
        """Submit at most one snapshot at a time; never queue old frames."""
        now = time.monotonic()
        self._collect_finished(now)

        if (not self._disabled and self._pending is None
                and now - self._started >= self.interval):
            self._started = now
            self._pending = self._pool.submit(self._predict, frame.copy())

        fresh = now - self._result_time <= 2 + self.interval
        return self._results if fresh else []

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
class Stats:
    def __init__(self):
        self.start = time.monotonic()
        self._last_tick = self.start
        self.fps_history: Deque[float] = deque(maxlen=100)
        self.detect_times: Deque[float] = deque(maxlen=50)
        self.total_detections = 0
        self.max_people = 0

    def tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now
        if dt > 0:
            self.fps_history.append(1.0 / dt)

    def record_detection(self, count: int, seconds: float) -> None:
        self.total_detections += count
        self.max_people = max(self.max_people, count)
        self.detect_times.append(seconds)

    @property
    def fps(self) -> float:
        return float(np.mean(self.fps_history)) if self.fps_history else 0.0

    @property
    def runtime(self) -> float:
        return time.monotonic() - self.start

    def panel_lines(self, method: str, conf: float, people: int, tracks: int) -> List[str]:
        detect_ms = np.mean(self.detect_times) * 1000 if self.detect_times else 0.0
        rate = self.total_detections / self.runtime if self.runtime > 0 else 0.0
        return [
            f"FPS: {self.fps:.1f} | Method: {method.upper()} | Conf: {conf:.2f}",
            f"People: {people} | Max: {self.max_people}",
            f"Cumulative detections: {self.total_detections} | Rate: {rate:.1f}/s",
            f"Detection time: {detect_ms:.1f} ms | Tracks: {tracks}",
            f"Runtime: {int(self.runtime)} s",
        ]

    def print_summary(self, method: str) -> None:
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print(f"Total runtime:          {self.runtime:.1f} s")
        print(f"Cumulative detections:  {self.total_detections}")
        print(f"Maximum people:         {self.max_people}")
        print(f"Average FPS:            {self.fps:.1f}")
        print(f"Final method:           {method.upper()}")
        print("=" * 60)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
class Renderer:
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    PANEL_LINE_HEIGHT = 20

    def __init__(self):
        self.show_trajectory = True
        self.show_confidence = True
        self.show_stats = True

    @staticmethod
    def color_for(track_id: int) -> Tuple[int, int, int]:
        return PALETTE[track_id % len(PALETTE)]

    def draw_tracks(self, frame: np.ndarray, tracks: List[Track]) -> None:
        for track in tracks:
            d, color = track.detection, self.color_for(track.id)

            # Top edge is lowered slightly to keep the face area clearer.
            cv2.rectangle(frame, (d.x, d.y + d.h // 10), (d.x + d.w, d.y + d.h), color, 2)
            cv2.circle(frame, d.center, 4, color, -1)

            if self.show_confidence:
                self._draw_confidence_bar(frame, track.id, d, color)
            if self.show_trajectory:
                self._draw_trail(frame, track, color)

    def _draw_confidence_bar(self, frame: np.ndarray, track_id: int,
                             d: Detection, color: Tuple[int, int, int]) -> None:
        """Confidence bar sits on top of the person box and matches its width."""
        frame_w = frame.shape[1]
        x1 = max(0, d.x)
        x2 = min(frame_w - 1, d.x + d.w)
        if x2 <= x1:
            return  # box is completely off-screen

        box_top = d.y + d.h // 10                           # same top edge as drawn box
        bar_bottom = max(box_top, CONF_BAR_HEIGHT + 16)     # keep bar and label on-screen
        bar_top = bar_bottom - CONF_BAR_HEIGHT

        fill_x = x1 + int((x2 - x1) * d.confidence)
        cv2.rectangle(frame, (x1, bar_top), (x2, bar_bottom), (50, 50, 50), -1)
        cv2.rectangle(frame, (x1, bar_top), (fill_x, bar_bottom), color, -1)

        # Shorter label for narrow boxes so the text stays roughly within the box.
        label = (f"ID:{track_id} Conf:{d.confidence:.2f}"
                 if x2 - x1 >= CONF_BAR_FULL_LABEL_W
                 else f"#{track_id} {d.confidence:.2f}")
        cv2.putText(frame, label, (x1, bar_top - 4),
                    self.FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_trail(frame: np.ndarray, track: Track, color: Tuple[int, int, int]) -> None:
        points = list(track.history)[-TRAIL_LENGTH:]
        for i, (start, end) in enumerate(zip(points, points[1:]), start=1):
            fade = i / len(points)  # older segments are darker
            cv2.line(frame, start, end, tuple(int(c * fade) for c in color), 2)

    def draw_faces(self, frame: np.ndarray, faces: list) -> None:
        color = (0, 220, 255)
        for face in faces:
            r = face["region"]
            x, y, w, h = (int(r[k]) for k in ("x", "y", "w", "h"))
            label = f"Age ~{face['age']:.0f} | Expression: {face['dominant_emotion']}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (max(0, x), min(frame.shape[0] - 2, y + h + 16)),
                        self.FONT, 0.5, color, 1, cv2.LINE_AA)

    def draw_footer(self, frame: np.ndarray, face_status: Optional[str]) -> None:
        lines = []
        if face_status:
            lines.append((face_status, (0, 220, 255)))
        lines += [(text, (255, 255, 255)) for text in HELP_LINES]

        y = frame.shape[0] - 10 - 20 * (len(lines) - 1)
        for text, color in lines:
            cv2.putText(frame, text, (10, y), self.FONT, 0.45, color, 1, cv2.LINE_AA)
            y += 20

    def compose(self, frame: np.ndarray, stats_lines: List[str]) -> np.ndarray:
        """Stack the stats panel above the video so no pixels are hidden."""
        if not self.show_stats:
            return frame
        panel = np.zeros((len(stats_lines) * self.PANEL_LINE_HEIGHT + 10,
                          frame.shape[1], 3), dtype=np.uint8)
        for i, text in enumerate(stats_lines):
            cv2.putText(panel, text, (10, 20 + i * self.PANEL_LINE_HEIGHT),
                        self.FONT, 0.5, (255, 255, 255), 1)
        return np.vstack([panel, frame])


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
def open_capture(source) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video source: {source!r}")
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    return cap


class PersonDetectionApp:
    def __init__(self, detectors: Dict[str, Detector],
                 faces: Optional[FaceAttributes] = None):
        self.detectors = detectors
        self.faces = faces
        self.method = "yolo" if "yolo" in detectors else "hog"
        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        self.tracker = Tracker()
        self.stats = Stats()
        self.renderer = Renderer()

    # -- controls ----------------------------------------------------------- #
    def _set_conf(self, value: float) -> None:
        self.conf_threshold = min(CONF_MAX, max(CONF_MIN, value))
        for detector in self.detectors.values():
            detector.conf_threshold = self.conf_threshold
        print(f"Confidence threshold: {self.conf_threshold:.2f}")

    def _toggle(self, attr: str, label: str) -> None:
        value = not getattr(self.renderer, attr)
        setattr(self.renderer, attr, value)
        print(f"{label}: {'ON' if value else 'OFF'}")

    def _handle_key(self, key: int) -> bool:
        """Apply a key press. Returns False when the app should quit."""
        char = chr(key) if 0 <= key < 128 else ""
        if char == "q":
            return False
        if char in METHOD_KEYS:
            method = METHOD_KEYS[char]
            if method in self.detectors:
                self.method = method
                print(f"Switched to {method.upper()} detection")
            else:
                print(f"{method.upper()} is not available")
        elif char == "t":
            self._toggle("show_trajectory", "Trajectory display")
        elif char == "c":
            self._toggle("show_confidence", "Confidence display")
        elif char == "s":
            self._toggle("show_stats", "Statistics display")
        elif char == "r":
            self.tracker.reset()
            print("Tracking reset")
        elif char in ("+", "="):
            self._set_conf(self.conf_threshold + CONF_STEP)
        elif char in ("-", "_"):
            self._set_conf(self.conf_threshold - CONF_STEP)
        return True

    # -- main loop ---------------------------------------------------------- #
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        faces = self.faces.update(frame) if self.faces else []

        started = time.perf_counter()
        detections = self.detectors[self.method].detect(frame)
        self.stats.record_detection(len(detections), time.perf_counter() - started)

        tracks = self.tracker.update(detections)
        self.stats.tick()

        self.renderer.draw_tracks(frame, tracks)
        self.renderer.draw_faces(frame, faces)
        self.renderer.draw_footer(frame, self.faces.status if self.faces else None)

        lines = self.stats.panel_lines(self.method, self.conf_threshold,
                                       len(tracks), len(self.tracker.tracks))
        return self.renderer.compose(frame, lines)

    def run(self, source) -> None:
        cap = open_capture(source)
        print("Methods available:", ", ".join(self.detectors))
        print("Press 'q' to quit.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    if isinstance(source, int):
                        print("Error: failed to capture camera frame.")
                    break
                cv2.imshow(WINDOW_NAME, self._process_frame(frame))
                if not self._handle_key(cv2.waitKey(1) & 0xFF):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        self.stats.print_summary(self.method)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect people in a webcam feed or video file.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    source.add_argument("--video", help="Path to a local video file")
    parser.add_argument("--face-attributes", action=argparse.BooleanOptionalAction,
                        default=True, help="Show estimated age and facial expression")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    faces: Optional[FaceAttributes] = None
    try:
        if args.face_attributes:
            try:
                faces = FaceAttributes()
            except Exception as exc:
                raise SystemExit(f"Face analysis setup failed: {exc}\n"
                                 "Install the face requirements or run with "
                                 "--no-face-attributes.")
        app = PersonDetectionApp(build_detectors(), faces)
        app.run(args.video if args.video is not None else args.camera)
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    finally:
        if faces is not None:
            faces.close()


if __name__ == "__main__":
    main()