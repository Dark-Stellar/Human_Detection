from concurrent.futures import ThreadPoolExecutor
from time import monotonic
import cv2
import numpy as np
import time
from collections import deque, defaultdict
import os
import argparse
from pathlib import Path

class FaceAttributes:
    def __init__(self, interval=1.0):
        from deepface import DeepFace

        print("Loading age and expression models (first run may download weights)...", flush=True)
        for model in ("Age", "Emotion"):
            DeepFace.build_model(model, task="facial_attribute")
        print("Face analysis ready.", flush=True)
        self.analyze = DeepFace.analyze
        self.interval = interval
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.started = float('-inf')
        self.results = []
        self.result_time = float("-inf")
        self.disabled = False
        self.status = "Face analysis: waiting for first result"

    def _predict(self, frame):
        try:
            return self.analyze(frame, actions=('age', 'emotion'),
                                detector_backend='opencv', enforce_detection=True,
                                align=True, silent=True)
        except ValueError as exc:
            if 'face could not be detected' in str(exc).lower():
                return []
            raise

    def update(self, frame):
        """Submit at most one snapshot at a time; never queue old frames."""
        now = monotonic()
        if self.pending is not None and self.pending.done():
            try:
                result = self.pending.result()
                self.result_time = now
                self.results = result
                self.status = "Face analysis: active" if result else "Face analysis: no clear face"
            except Exception as exc:
                print(f'Face analysis disabled: {exc}')
                self.disabled = True
                self.status = "Face analysis failed; see terminal"
                self.results = []
            self.pending = None
        if not self.disabled and self.pending is None and now - self.started >= self.interval:
            self.started = now
            self.pending = self.pool.submit(self._predict, frame.copy())
        return self.results if now - self.result_time <= 2 + self.interval else []

    @staticmethod
    def draw(frame, results):
        for face in results:
            region = face['region']
            x, y, w, h = (int(region[key]) for key in ('x', 'y', 'w', 'h'))
            label = f"Age ~{face['age']:.0f} | Expression: {face['dominant_emotion']}"
            color = (0, 220, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (max(0, x), min(frame.shape[0] - 2, y + h + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


class CompletePersonDetector:
    def __init__(self):
        # Initialize HOG detector (always available)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # YOLO configuration
        self.weights_path = str(Path(__file__).resolve().with_name("yolov3.weights"))
        self.config_path = str(Path(__file__).resolve().with_name("yolov3.cfg"))
        self.names_path = str(Path(__file__).resolve().with_name("coco.names"))

        # Load YOLO model
        self.yolo_net = None
        self.yolo_classes = []
        self.setup_yolo()

        # Detection history for tracking
        self.detection_history = deque(maxlen=30)
        self.person_tracks = defaultdict(lambda: deque(maxlen=20))
        self.next_person_id = 1

        # Performance metrics
        self.fps_history = deque(maxlen=100)
        self.detection_times = deque(maxlen=50)

        # Detection parameters
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.input_size = (416, 416)

        # Detection method
        self.current_method = "yolo" if self.yolo_net else "hog"

        # Statistics
        self.total_detections = 0
        self.max_people_count = 0
        self.start_time = time.time()

        # UI settings
        self.show_trajectory = True
        self.show_confidence = True
        self.show_statistics = True

    def setup_yolo(self):
        """Load local model files without downloading anything."""
        if not all(os.path.isfile(f) for f in (self.weights_path, self.config_path, self.names_path)):
            print("Local YOLOv3 files not found; using HOG.")
            return

        try:
            # Load COCO class names
            with open(self.names_path, 'rt') as f:
                self.yolo_classes = f.read().strip().split('\n')

            # Load YOLO network
            self.yolo_net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
            self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("YOLO model loaded successfully!")

        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Falling back to HOG detector")
            self.yolo_net = None

    def get_output_layers_names(self):
        """Get names of output layers for YOLO"""
        if self.yolo_net is None:
            return []

        layer_names = self.yolo_net.getLayerNames()
        try:
            # OpenCV 4.x
            return [layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
        except:
            # OpenCV 3.x
            return [layer_names[i[0] - 1] for i in self.yolo_net.getUnconnectedOutLayers()]

    def detect_with_yolo(self, frame):
        """Detect objects using YOLOv3"""
        if self.yolo_net is None:
            return []

        height, width = frame.shape[:2]

        # Create blob from input frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, self.input_size, swapRB=True, crop=False)
        self.yolo_net.setInput(blob)

        # Get output layer names and run forward pass
        output_layers = self.get_output_layers_names()
        start_time = time.time()
        outputs = self.yolo_net.forward(output_layers)
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)

        boxes = []
        confidences = []
        class_ids = []

        # Process each detection
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Filter for person class (class_id 0) and confidence
                if class_id == 0 and confidence > self.conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate top-left corner
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression to eliminate redundant boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        final_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                final_detections.append((x, y, w, h, confidence))
                self.total_detections += 1

        return final_detections

    def detect_with_hog(self, frame):
        """Fallback detection using HOG"""
        try:
            boxes, weights = self.hog.detectMultiScale(
                frame,
                winStride=(8, 8),
                padding=(32, 32),
                scale=1.05,
                hitThreshold=2.0,
                groupThreshold=2
            )

            detected_people = []
            if len(boxes) > 0:
                for (x, y, w, h), weight in zip(boxes, np.asarray(weights).reshape(-1)):
                    confidence = float(np.clip(weight / 2.0, 0.0, 1.0))
                    detected_people.append((x, y, w, h, confidence))
                    self.total_detections += 1

            return detected_people
        except Exception as e:
            print(f"HOG detection error: {e}")
            return []

    def detect_with_haar_cascade(self, frame):
        """Alternative detection using Haar Cascade"""
        try:
            # Try to load Haar cascade for full body
            cascade_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            bodies = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )

            detected_people = []
            for (x, y, w, h) in bodies:
                detected_people.append((x, y, w, h, 0.5))  # Default confidence for Haar
                self.total_detections += 1

            return detected_people
        except Exception as e:
            print(f"Haar cascade error: {e}")
            return []

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for tracking"""
        x1, y1, w1, h1 = box1[:4]
        x2, y2, w2, h2 = box2[:4]

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def track_people(self, current_detections):
        """Track people across frames with improved matching"""
        current_time = time.time()
        matched_ids = {}

        for detection in current_detections:
            x, y, w, h, confidence = detection
            centroid = (x + w//2, y + h//2)

            best_match_id = None
            best_score = 0.3  # Minimum score threshold

            for person_id, track in self.person_tracks.items():
                if len(track) > 0:
                    last_detection = track[-1]
                    last_box = last_detection[:4]
                    current_box = (x, y, w, h)

                    # Calculate IoU
                    iou = self.calculate_iou(last_box, current_box)

                    # Calculate centroid distance (normalized by frame size)
                    last_centroid = (last_box[0] + last_box[2]//2, last_box[1] + last_box[3]//2)
                    centroid_distance = np.sqrt((centroid[0] - last_centroid[0])**2 +
                                              (centroid[1] - last_centroid[1])**2) / 100.0

                    # Combined score (higher is better)
                    score = iou * 0.7 + max(0, 1 - centroid_distance) * 0.3

                    if score > best_score:
                        best_score = score
                        best_match_id = person_id

            if best_match_id is not None:
                matched_ids[best_match_id] = detection
                self.person_tracks[best_match_id].append((x, y, w, h, confidence, current_time))
            else:
                # New person detected
                new_id = self.next_person_id
                self.next_person_id += 1
                matched_ids[new_id] = detection
                self.person_tracks[new_id].append((x, y, w, h, confidence, current_time))

        # Clean up old tracks (older than 5 seconds)
        current_time = time.time()
        expired_ids = []
        for person_id, track in self.person_tracks.items():
            if track and current_time - track[-1][5] > 5.0:  # 5 seconds timeout
                expired_ids.append(person_id)

        for expired_id in expired_ids:
            del self.person_tracks[expired_id]

        # Update max people count
        self.max_people_count = max(self.max_people_count, len(matched_ids))

        return matched_ids

    def draw_detections(self, frame, tracked_detections):
        """Draw advanced visualization with tracking information"""
        for person_id, detection in tracked_detections.items():
            x, y, w, h, confidence = detection

            # Generate color based on person ID for consistent tracking
            color = self.get_color_from_id(person_id)

            # Lower the displayed top edge to keep the face area clearer.
            cv2.rectangle(frame, (x, y + h // 10), (x + w, y + h), color, 2)

            # Draw confidence bar
            if self.show_confidence:
                bar_width = 100
                confidence_width = int(confidence * bar_width)
                cv2.rectangle(frame, (x, y - 25), (x + bar_width, y - 5), (50, 50, 50), -1)
                cv2.rectangle(frame, (x, y - 25), (x + confidence_width, y - 5), color, -1)

                # Draw info text
                info_text = f"ID:{person_id} Conf:{confidence:.2f}"
                cv2.putText(frame, info_text, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw centroid
            centroid = (x + w//2, y + h//2)
            cv2.circle(frame, centroid, 4, color, -1)

            # Draw trajectory
            if self.show_trajectory and person_id in self.person_tracks:
                track = self.person_tracks[person_id]
                points = []
                for detection_point in track:
                    px, py, pw, ph, pconf, ptime = detection_point
                    center = (px + pw//2, py + ph//2)
                    points.append(center)

                # Draw trajectory lines (only recent points)
                for i in range(max(1, len(points) - 10), len(points)):
                    if i > 0:
                        cv2.line(frame, points[i-1], points[i], color, 2)

                # Draw trail dots
                for i, point in enumerate(points[-10:]):  # Only last 10 points
                    alpha = i / 10.0  # Fade out older points
                    trail_color = tuple(int(c * alpha) for c in color)
                    cv2.circle(frame, point, 2, trail_color, -1)

    def get_color_from_id(self, person_id):
        """Generate consistent color based on person ID"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128),
            (255, 128, 0), (128, 255, 0), (255, 0, 128)
        ]
        return colors[person_id % len(colors)]

    def calculate_fps(self, start_time):
        """Calculate and return FPS"""
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        self.fps_history.append(fps)
        return np.mean(self.fps_history)

    def draw_statistics(self, frame, fps, detection_count):
        """Draw comprehensive performance statistics on frame"""
        if not self.show_statistics:
            return

        stats_height = 120
        stats_bg = np.zeros((stats_height, frame.shape[1], 3), dtype=np.uint8)

        # Calculate additional statistics
        avg_detection_time = np.mean(self.detection_times) * 1000 if self.detection_times else 0
        runtime = time.time() - self.start_time
        detection_rate = self.total_detections / runtime if runtime > 0 else 0

        stats_text = [
            f"FPS: {fps:.1f} | Method: {self.current_method.upper()}",
            f"People detected: {detection_count} | Max: {self.max_people_count}",
            f"Cumulative detections: {self.total_detections} | Rate: {detection_rate:.1f}/s",
            f"Detection time: {avg_detection_time:.1f}ms | Tracks: {len(self.person_tracks)}",
            f"Runtime: {int(runtime)}s | YOLO: {'Yes' if self.yolo_net else 'No'}"
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(stats_bg, text, (10, 20 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Combine stats with main frame
        visible_height = min(frame.shape[0], stats_bg.shape[0])
        frame[:visible_height] = stats_bg[:visible_height]

    def draw_control_help(self, frame):
        """Draw control instructions on frame"""
        help_text = [
            "CONTROLS:",
            "Q: Quit  |  H: HOG  |  Y: YOLO  |  A: Haar",
            "T: Toggle Trajectory  |  C: Toggle Confidence  |  S: Toggle Stats",
            "R: Reset Tracking  |  +/-: Adjust Confidence"
        ]

        y_start = frame.shape[0] - 80
        for i, text in enumerate(help_text):
            cv2.putText(frame, text, (10, y_start + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def adjust_confidence(self, increase=True):
        """Adjust confidence threshold"""
        if increase:
            self.conf_threshold = min(0.9, self.conf_threshold + 0.05)
        else:
            self.conf_threshold = max(0.1, self.conf_threshold - 0.05)
        print(f"Confidence threshold set to: {self.conf_threshold:.2f}")

    def detect_people_complete(self, source=0, face_attributes=None):
        """Main detection function with all features"""
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open video source: {source!r}")

        try:
            # Set camera resolution for better performance
            if isinstance(source, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

            print("=" * 60)
            print("COMPLETE PERSON DETECTION SYSTEM")
            print("=" * 60)
            print(f"Available detection methods:")
            print(f"  - HOG: Always available")
            print(f"  - YOLO: {'Available' if self.yolo_net else 'Not available'}")
            print(f"  - Haar Cascade: Available")
            print("\nControls:")
            print("  'q' - Quit")
            print("  'h' - Use HOG detection")
            print("  'y' - Use YOLO detection (if available)")
            print("  'a' - Use Haar cascade detection")
            print("  't' - Toggle trajectory display")
            print("  'c' - Toggle confidence bars")
            print("  's' - Toggle statistics panel")
            print("  'r' - Reset tracking")
            print("  '+' - Increase confidence threshold")
            print("  '-' - Decrease confidence threshold")
            print("=" * 60)

            while True:
                start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, int):
                        print("Error: Failed to capture camera frame.")
                    break

                face_results = face_attributes.update(frame) if face_attributes else []

                # Detect people based on selected method
                if self.current_method == "yolo" and self.yolo_net:
                    detections = self.detect_with_yolo(frame)
                elif self.current_method == "haar":
                    detections = self.detect_with_haar_cascade(frame)
                else:
                    detections = self.detect_with_hog(frame)

                # Track people across frames
                tracked_detections = self.track_people(detections)

                # Draw visualizations
                self.draw_detections(frame, tracked_detections)

                # Calculate and display statistics
                fps = self.calculate_fps(start_time)
                self.draw_statistics(frame, fps, len(tracked_detections))

                # Draw control help
                self.draw_control_help(frame)

                # Display method and confidence info
                method_text = f"Method: {self.current_method.upper()} | Conf: {self.conf_threshold:.2f}"
                cv2.putText(frame, method_text, (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if face_attributes:
                    face_attributes.draw(frame, face_results)
                    cv2.putText(frame, face_attributes.status, (10, max(15, frame.shape[0] - 95)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)

                # Display frame
                cv2.imshow('Complete Person Detection System', frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('h'):
                    self.current_method = "hog"
                    print("Switched to HOG detection")
                elif key == ord('y') and self.yolo_net:
                    self.current_method = "yolo"
                    print("Switched to YOLO detection")
                elif key == ord('a'):
                    self.current_method = "haar"
                    print("Switched to Haar cascade detection")
                elif key == ord('t'):
                    self.show_trajectory = not self.show_trajectory
                    print(f"Trajectory display: {'ON' if self.show_trajectory else 'OFF'}")
                elif key == ord('c'):
                    self.show_confidence = not self.show_confidence
                    print(f"Confidence display: {'ON' if self.show_confidence else 'OFF'}")
                elif key == ord('s'):
                    self.show_statistics = not self.show_statistics
                    print(f"Statistics display: {'ON' if self.show_statistics else 'OFF'}")
                elif key == ord('r'):
                    self.person_tracks.clear()
                    self.next_person_id = 1
                    print("Tracking reset")
                elif key == ord('+'):
                    self.adjust_confidence(True)
                elif key == ord('-'):
                    self.adjust_confidence(False)

            # Final statistics
            runtime = time.time() - self.start_time
            print("\n" + "=" * 60)
            print("SESSION SUMMARY:")
            print(f"Total runtime: {runtime:.1f} seconds")
            print(f"Cumulative detections: {self.total_detections}")
            print(f"Maximum people detected: {self.max_people_count}")
            print(f"Average FPS: {(np.mean(self.fps_history) if self.fps_history else 0):.1f}")
            print(f"Final detection method: {self.current_method.upper()}")
            print("=" * 60)

        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Detect people in a webcam feed or video file.")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    inputs.add_argument("--video", help="Path to a local video file")
    parser.add_argument("--face-attributes", action=argparse.BooleanOptionalAction, default=True,
                        help="Show estimated age and facial expression (optional dependencies)")
    args = parser.parse_args()
    detector = CompletePersonDetector()
    faces = None
    try:
        if args.face_attributes:
            try:
                faces = FaceAttributes()
            except Exception as exc:
                parser.exit(1, f"Face analysis setup failed: {exc}\n"
                            "Install requirements-face.txt or run with --no-face-attributes.\n")
        detector.detect_people_complete(args.video if args.video is not None else args.camera, faces)
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    finally:
        if faces is not None:
            faces.close()


if __name__ == "__main__":
    main()