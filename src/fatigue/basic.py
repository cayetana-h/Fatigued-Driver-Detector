from __future__ import annotations

import collections

import cv2
import mediapipe as mp
import numpy as np

LEFT_EYE_INDICES = [33, 133, 159, 145, 158, 153]
RIGHT_EYE_INDICES = [362, 263, 386, 374, 385, 380]

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

NOSE_TIP = 1

EYE_ASPECT_RATIO_THRESHOLD = 0.08
CONSECUTIVE_CLOSED_FRAMES = 24
EYE_ALERT_HOLD_FRAMES = 60
EYE_WINDOW_FRAMES = 24
EYE_WINDOW_CLOSED_FRAMES = 20
SEVERE_EYE_ASPECT_RATIO_THRESHOLD = 0.05
SEVERE_EYE_WINDOW_FRAMES = 12
SEVERE_EYE_WINDOW_CLOSED_FRAMES = 10

YAWN_RATIO_THRESHOLD = 0.30
YAWN_FRAMES_REQUIRED = 4
YAWN_ALERT_WINDOW_FRAMES = 90

HEAD_DROP_THRESHOLD = 0.035
HEAD_DROP_FRAMES = 12
HEAD_DROP_STANDALONE_ALERT = False
MAX_EYE_YAW_RATIO = 0.30


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _eye_aspect_ratio(landmarks: np.ndarray, indices: list[int]) -> float:
    left, right, top_1, bottom_1, top_2, bottom_2 = [landmarks[i] for i in indices]
    vertical = _distance(top_1, bottom_1) + _distance(top_2, bottom_2)
    horizontal = max(2.0 * _distance(left, right), 1e-6)
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


def _face_yaw_ratio(landmarks: np.ndarray) -> float:
    nose = landmarks[NOSE_TIP]
    left_eye = landmarks[LEFT_EYE_INDICES[0]]
    right_eye = landmarks[RIGHT_EYE_INDICES[0]]
    eye_width = max(abs(float(right_eye[0] - left_eye[0])), 1e-6)
    eye_center_x = float((left_eye[0] + right_eye[0]) * 0.5)
    return abs(float(nose[0]) - eye_center_x) / eye_width


def _to_array(landmarks) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


class DriverFatigueMonitor:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.eye_closed_frames = 0
        self.eye_closed_history: collections.deque[int] = collections.deque(
            maxlen=EYE_WINDOW_FRAMES
        )
        self.severe_eye_closed_history: collections.deque[int] = collections.deque(
            maxlen=SEVERE_EYE_WINDOW_FRAMES
        )
        self.recent_eye_closure_frames = 0
        self.yawn_frames = 0
        self.yawn_count = 0
        self.recent_yawn_frames = 0
        self.head_drop_frames = 0
        self.face_visible = False
        self.last_metrics: dict[str, float] = {}
        self.last_reasons: list[str] = []

    def update(self, frame: np.ndarray) -> bool:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            self.face_visible = False
            self.eye_closed_frames = 0
            self.eye_closed_history.append(0)
            self.severe_eye_closed_history.append(0)
            return False

        self.face_visible = True

        landmarks = _to_array(result.multi_face_landmarks[0].landmark)

        left_ratio = _eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
        right_ratio = _eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
        mouth_ratio = _mouth_open_ratio(landmarks)
        head_drop = _head_drop_ratio(landmarks)
        yaw_ratio = _face_yaw_ratio(landmarks)

        avg_eye_ratio = float((left_ratio + right_ratio) * 0.5)

        self.last_metrics = {
            "eye_ratio": avg_eye_ratio,
            "mouth_ratio": mouth_ratio,
            "head_drop": head_drop,
            "yaw_ratio": yaw_ratio,
        }

        eye_measurement_reliable = yaw_ratio <= MAX_EYE_YAW_RATIO
        eye_closed = (
            avg_eye_ratio < EYE_ASPECT_RATIO_THRESHOLD
            and eye_measurement_reliable
        )
        severe_eye_closed = (
            avg_eye_ratio < SEVERE_EYE_ASPECT_RATIO_THRESHOLD
            and eye_measurement_reliable
        )

        self.eye_closed_history.append(int(eye_closed))
        self.severe_eye_closed_history.append(int(severe_eye_closed))

        if eye_closed:
            self.eye_closed_frames += 1
        else:
            if self.recent_eye_closure_frames > 0:
                self.recent_eye_closure_frames -= 1
            self.eye_closed_frames = 0

        if mouth_ratio > YAWN_RATIO_THRESHOLD:
            self.yawn_frames += 1
        else:
            if self.yawn_frames >= YAWN_FRAMES_REQUIRED:
                self.yawn_count += 1
                self.recent_yawn_frames = YAWN_ALERT_WINDOW_FRAMES
            elif self.recent_yawn_frames > 0:
                self.recent_yawn_frames -= 1
            self.yawn_frames = 0

        if head_drop > HEAD_DROP_THRESHOLD:
            self.head_drop_frames += 1
        else:
            self.head_drop_frames = 0

        active_yawn = self.yawn_frames >= YAWN_FRAMES_REQUIRED
        recent_yawn = self.recent_yawn_frames > 0
        rolling_eye_closure = (
            len(self.eye_closed_history) >= EYE_WINDOW_CLOSED_FRAMES
            and sum(self.eye_closed_history) >= EYE_WINDOW_CLOSED_FRAMES
        )
        severe_rolling_eye_closure = (
            len(self.severe_eye_closed_history) >= SEVERE_EYE_WINDOW_CLOSED_FRAMES
            and sum(self.severe_eye_closed_history) >= SEVERE_EYE_WINDOW_CLOSED_FRAMES
        )
        active_eye_closure = (
            self.eye_closed_frames >= CONSECUTIVE_CLOSED_FRAMES
            or rolling_eye_closure
            or severe_rolling_eye_closure
        )

        if active_eye_closure:
            self.recent_eye_closure_frames = EYE_ALERT_HOLD_FRAMES

        recent_eye_closure = self.recent_eye_closure_frames > 0

        self.last_reasons = []
        if active_eye_closure or recent_eye_closure:
            self.last_reasons.append("eyes")
        if active_yawn or recent_yawn:
            self.last_reasons.append("yawn")
        if HEAD_DROP_STANDALONE_ALERT and self.head_drop_frames >= HEAD_DROP_FRAMES:
            self.last_reasons.append("head")

        drowsy = bool(self.last_reasons)

        return drowsy

    def summary(self) -> str:
        if not self.face_visible:
            return "face lost"

        return (
            f"eyes={self.last_metrics.get('eye_ratio', 0):.2f} "
            f"mouth={self.last_metrics.get('mouth_ratio', 0):.2f} "
            f"head={self.last_metrics.get('head_drop', 0):.3f} "
            f"yaw={self.last_metrics.get('yaw_ratio', 0):.2f} "
            f"eye_frames={self.eye_closed_frames} "
            f"eye_window={sum(self.eye_closed_history)}/{len(self.eye_closed_history)} "
            f"severe_eye_window={sum(self.severe_eye_closed_history)}/{len(self.severe_eye_closed_history)} "
            f"recent_eye_frames={self.recent_eye_closure_frames} "
            f"yawn_frames={self.yawn_frames} "
            f"recent_yawn_frames={self.recent_yawn_frames} "
            f"head_frames={self.head_drop_frames} "
            f"reasons={','.join(self.last_reasons) or 'none'}"
        )

    def close(self) -> None:
        self._face_mesh.close()


def detect_fatigue(frame: np.ndarray) -> bool:
    monitor = DriverFatigueMonitor()
    result = monitor.update(frame)
    monitor.close()
    return result
