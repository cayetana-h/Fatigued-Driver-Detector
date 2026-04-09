from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

LEFT_EYE_INDICES = [33, 160, 158, 133]
RIGHT_EYE_INDICES = [362, 385, 387, 263]

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

NOSE_TIP = 1


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _eye_aspect_ratio(landmarks: np.ndarray, indices: list[int]) -> float:
    left, top, bottom, right = [landmarks[i] for i in indices]
    vertical = _distance(top, bottom)
    horizontal = max(_distance(left, right), 1e-6)
    return vertical / horizontal


def _mouth_open_ratio(landmarks: np.ndarray) -> float:
    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]
    vertical = _distance(top, bottom)
    horizontal = max(_distance(left, right), 1e-6)
    return vertical / horizontal


def _head_drop_ratio(landmarks: np.ndarray) -> float:
    nose = landmarks[NOSE_TIP]
    left_eye = landmarks[LEFT_EYE_INDICES[0]]
    right_eye = landmarks[RIGHT_EYE_INDICES[0]]
    eye_center = (left_eye + right_eye) / 2.0
    return float(nose[1] - eye_center[1])


def _to_array(landmarks) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


@dataclass
class FatigueConfig:
    eye_aspect_ratio_threshold: float = 0.20
    consecutive_closed_frames: int = 6
    yawn_ratio_threshold: float = 0.65
    yawn_frames_required: int = 3
    yawn_alert_window_seconds: float = 10.0
    head_drop_threshold: float = 0.035
    head_drop_frames: int = 12
    face_missing_reset_frames: int = 15


class DriverFatigueMonitor:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        config: FatigueConfig | None = None,
    ):
        self.config = config or FatigueConfig()
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.eye_closed_frames = 0
        self.yawn_frames = 0
        self.yawn_count = 0
        self.last_yawn_time: float | None = None
        self.head_drop_frames = 0
        self.face_visible = False
        self.face_missing_frames = 0
        self.last_reason = "normal"
        self.last_metrics: dict[str, float] = {}

    def _reset_fatigue_counters(self) -> None:
        self.eye_closed_frames = 0
        self.yawn_frames = 0
        self.head_drop_frames = 0
        self.last_yawn_time = None

    def update(self, frame: np.ndarray) -> bool:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            self.face_visible = False
            self.face_missing_frames += 1
            self.last_reason = "face_lost"

            if self.face_missing_frames >= self.config.face_missing_reset_frames:
                self._reset_fatigue_counters()

            return False

        self.face_visible = True
        self.face_missing_frames = 0

        landmarks = _to_array(result.multi_face_landmarks[0].landmark)

        left_ratio = _eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
        right_ratio = _eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
        mouth_ratio = _mouth_open_ratio(landmarks)
        head_drop = _head_drop_ratio(landmarks)
        avg_eye_ratio = float((left_ratio + right_ratio) * 0.5)

        self.last_metrics = {
            "eye_ratio": avg_eye_ratio,
            "mouth_ratio": mouth_ratio,
            "head_drop": head_drop,
        }

        if avg_eye_ratio < self.config.eye_aspect_ratio_threshold:
            self.eye_closed_frames += 1
        else:
            self.eye_closed_frames = 0

        if mouth_ratio > self.config.yawn_ratio_threshold:
            self.yawn_frames += 1
        else:
            if self.yawn_frames >= self.config.yawn_frames_required:
                self.yawn_count += 1
                self.last_yawn_time = time.monotonic()
            self.yawn_frames = 0

        if head_drop > self.config.head_drop_threshold:
            self.head_drop_frames += 1
        else:
            self.head_drop_frames = 0

        recent_yawn = (
            self.last_yawn_time is not None
            and (time.monotonic() - self.last_yawn_time) < self.config.yawn_alert_window_seconds
        )

        drowsy = False
        self.last_reason = "normal"

        if self.eye_closed_frames >= self.config.consecutive_closed_frames:
            drowsy = True
            self.last_reason = "eyes_closed"
        elif recent_yawn:
            drowsy = True
            self.last_reason = "recent_yawn"
        elif self.head_drop_frames >= self.config.head_drop_frames:
            drowsy = True
            self.last_reason = "head_drop"

        return drowsy

    def summary(self) -> str:
        if not self.face_visible:
            return f"face lost ({self.face_missing_frames} frames)"

        return (
            f"eyes={self.last_metrics.get('eye_ratio', 0):.2f} "
            f"mouth={self.last_metrics.get('mouth_ratio', 0):.2f} "
            f"head={self.last_metrics.get('head_drop', 0):.3f} "
            f"reason={self.last_reason}"
        )

    def snapshot(self) -> dict[str, float | int | str | bool]:
        return {
            "face_visible": self.face_visible,
            "face_missing_frames": self.face_missing_frames,
            "eye_closed_frames": self.eye_closed_frames,
            "yawn_frames": self.yawn_frames,
            "yawn_count": self.yawn_count,
            "head_drop_frames": self.head_drop_frames,
            "eye_ratio": self.last_metrics.get("eye_ratio", 0.0),
            "mouth_ratio": self.last_metrics.get("mouth_ratio", 0.0),
            "head_drop": self.last_metrics.get("head_drop", 0.0),
            "reason": self.last_reason,
        }

    def close(self) -> None:
        self._face_mesh.close()


def detect_fatigue(frame: np.ndarray) -> bool:
    monitor = DriverFatigueMonitor()
    result = monitor.update(frame)
    monitor.close()
    return result