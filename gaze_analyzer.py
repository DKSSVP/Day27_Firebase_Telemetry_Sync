import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = "face_landmarker.task"

ANOMALY_DURATION_SECONDS = 2.0

HORIZONTAL_TOLERANCE = 0.15
VERTICAL_TOLERANCE = 0.12

LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)

LEFT_IRIS_INDICES = (468, 469, 470, 471, 472)
RIGHT_IRIS_INDICES = (473, 474, 475, 476, 477)


class GazeAnalyzer:
    """MediaPipe-based gaze deviation analyzer."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found: {self.model_path}"
            )

        self.mp_vision = mp.tasks.vision

        options = self.mp_vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path)
            ),
            running_mode=self.mp_vision.RunningMode.VIDEO,
            num_faces=1,
        )

        self.landmarker = self.mp_vision.FaceLandmarker.create_from_options(
            options
        )

        self.off_center_start: Optional[float] = None
        self.last_timestamp_ms = 0

    @staticmethod
    def _calculate_eye_vector(
        landmarks: Any,
        corner_indices: Tuple[int, int],
        iris_indices: Tuple[int, ...],
    ) -> Tuple[float, float]:

        corner_a = landmarks[corner_indices[0]]
        corner_b = landmarks[corner_indices[1]]

        iris_points = [landmarks[index] for index in iris_indices]

        iris_x = np.mean([point.x for point in iris_points])
        iris_y = np.mean([point.y for point in iris_points])

        eye_center_x = (corner_a.x + corner_b.x) / 2
        eye_center_y = (corner_a.y + corner_b.y) / 2

        eye_width = abs(corner_a.x - corner_b.x)

        if eye_width < 1e-6:
            return 0.0, 0.0

        horizontal = (iris_x - eye_center_x) / eye_width
        vertical = (iris_y - eye_center_y) / eye_width

        return float(horizontal), float(vertical)

    def _get_gaze_vector(
        self, landmarks: Any
    ) -> Tuple[float, float]:

        left_vector = self._calculate_eye_vector(
            landmarks,
            LEFT_EYE_CORNERS,
            LEFT_IRIS_INDICES,
        )

        right_vector = self._calculate_eye_vector(
            landmarks,
            RIGHT_EYE_CORNERS,
            RIGHT_IRIS_INDICES,
        )

        average_x = (left_vector[0] + right_vector[0]) / 2
        average_y = (left_vector[1] + right_vector[1]) / 2

        return average_x, average_y

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze one webcam frame.

        Returns a normalized gaze result for the central alert engine.
        """

        current_time = time.monotonic()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = int(current_time * 1000)

        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1

        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        if not result.face_landmarks:
            self.off_center_start = None

            return {
                "face_detected": False,
                "triggered": False,
                "duration_seconds": 0.0,
                "vector_shift": {"x": 0.0, "y": 0.0},
                "status": "NO_FACE_DETECTED",
            }

        landmarks = result.face_landmarks[0]

        gaze_x, gaze_y = self._get_gaze_vector(landmarks)

        off_center = (
            abs(gaze_x) > HORIZONTAL_TOLERANCE
            or abs(gaze_y) > VERTICAL_TOLERANCE
        )

        if off_center:
            if self.off_center_start is None:
                self.off_center_start = current_time

            duration = current_time - self.off_center_start

        else:
            self.off_center_start = None
            duration = 0.0

        triggered = (
            off_center
            and duration >= ANOMALY_DURATION_SECONDS
        )

        return {
            "face_detected": True,
            "triggered": triggered,
            "duration_seconds": round(duration, 3),
            "vector_shift": {
                "x": round(gaze_x, 4),
                "y": round(gaze_y, 4),
            },
            "status": (
                "GAZE_DEVIATION"
                if triggered
                else "OFF_CENTER_PENDING"
                if off_center
                else "CENTERED"
            ),
        }

    def close(self) -> None:
        """Release the MediaPipe Face Landmarker."""

        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None


if __name__ == "__main__":
    analyzer = None
    camera = None

    try:
        analyzer = GazeAnalyzer()

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError("Unable to open webcam.")

        print("Gaze Analyzer started. Press Q or ESC to exit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            gaze_data = analyzer.analyze_frame(frame)

            status = gaze_data["status"]

            color = (
                (0, 0, 255)
                if gaze_data["triggered"]
                else (0, 255, 0)
            )

            cv2.putText(
                frame,
                f"Status: {status}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            cv2.putText(
                frame,
                (
                    f"X: {gaze_data['vector_shift']['x']:.3f} "
                    f"Y: {gaze_data['vector_shift']['y']:.3f}"
                ),
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Duration: {gaze_data['duration_seconds']:.2f}s",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Day26 - Gaze Analyzer", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except Exception as error:
        print(f"Gaze Analyzer Error: {error}")

    finally:
        if camera is not None:
            camera.release()

        if analyzer is not None:
            analyzer.close()

        cv2.destroyAllWindows()
        print("Gaze Analyzer stopped. Resources released.")