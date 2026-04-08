from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp

LEFT_EYE_INDICES = [33, 160, 158, 133]
RIGHT_EYE_INDICES = [362, 385, 387, 263]
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
NOSE_TIP = 1

EYE_ASPECT_RATIO_THRESHOLD = 0.20
CONSECUTIVE_CLOSED_FRAMES = 6
YAWN_RATIO_THRESHOLD = 0.65
YAWN_FRAMES_REQUIRED = 3
HEAD_DROP_THRESHOLD = 0.035
HEAD_DROP_FRAMES = 12


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


class DriverFatigueMonitor:
    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
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
        self.head_drop_frames = 0
        self.face_visible = False
        self.last_metrics: dict[str, float] = {}

    def update(self, frame: np.ndarray) -> bool:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            self.face_visible = False
            return False

        self.face_visible = True
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

        if avg_eye_ratio < EYE_ASPECT_RATIO_THRESHOLD:
            self.eye_closed_frames += 1
        else:
            self.eye_closed_frames = 0

        if mouth_ratio > YAWN_RATIO_THRESHOLD:
            self.yawn_frames += 1
        else:
            if self.yawn_frames >= YAWN_FRAMES_REQUIRED:
                self.yawn_count += 1
            self.yawn_frames = 0

        if head_drop > HEAD_DROP_THRESHOLD:
            self.head_drop_frames += 1
        else:
            self.head_drop_frames = 0

        drowsy = (
            self.eye_closed_frames >= CONSECUTIVE_CLOSED_FRAMES
            or self.yawn_count > 0
            or self.head_drop_frames >= HEAD_DROP_FRAMES
        )
        return drowsy

    def summary(self) -> str:
        if not self.face_visible:
            return "face lost"
        return (
            f"eyes={self.last_metrics.get('eye_ratio', 0):.2f} "
            f"mouth={self.last_metrics.get('mouth_ratio', 0):.2f} "
            f"head={self.last_metrics.get('head_drop', 0):.3f}"
        )

    def close(self) -> None:
        self._face_mesh.close()


def detect_fatigue(frame: np.ndarray) -> bool:
    monitor = DriverFatigueMonitor()
    result = monitor.update(frame)
    monitor.close()
    return result
