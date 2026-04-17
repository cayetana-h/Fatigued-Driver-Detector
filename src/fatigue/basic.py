from __future__ import annotations

import cv2
import numpy as np

# Eye-closure proxy: no eyes detected for several consecutive frames.
EYE_CLOSED_FRAMES_THRESHOLD = 10
# Head-drop proxy: face center drifts below calibrated baseline.
HEAD_DROP_PIXELS_THRESHOLD = 25
HEAD_DROP_FRAMES_THRESHOLD = 10
# Optional mouth-open proxy via smile cascade in lower face area.
MOUTH_OPEN_FRAMES_THRESHOLD = 8
# Stabilization: require persistence before switching overall status.
DROWSY_PERSISTENCE_FRAMES = 3
ALERT_PERSISTENCE_FRAMES = 6


class DriverFatigueMonitor:
    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        _ = (min_detection_confidence, min_tracking_confidence)
        self._face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._eyes = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )
        self._smile = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )

        self.eye_closed_frames = 0
        self.mouth_open_frames = 0
        self.head_drop_frames = 0
        self.face_visible = False
        self._baseline_face_y: float | None = None
        self._is_drowsy = False
        self._drowsy_votes = 0
        self._alert_votes = 0
        self.last_metrics: dict[str, float] = {}

    def update(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            self.face_visible = False
            self._alert_votes += 1
            self._drowsy_votes = 0
            if self._alert_votes >= ALERT_PERSISTENCE_FRAMES:
                self._is_drowsy = False
            return False

        self.face_visible = True
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y : y + h, x : x + w]

        if self._baseline_face_y is None:
            self._baseline_face_y = float(y + h * 0.5)

        eyes_roi = face_roi[0 : int(h * 0.55), :]
        eyes = self._eyes.detectMultiScale(eyes_roi, scaleFactor=1.1, minNeighbors=4, minSize=(18, 18))
        eyes_visible = len(eyes) >= 1

        mouth_roi = face_roi[int(h * 0.55) : h, :]
        mouth = self._smile.detectMultiScale(mouth_roi, scaleFactor=1.5, minNeighbors=20, minSize=(25, 20))
        mouth_open = len(mouth) > 0

        head_drop = float((y + h * 0.5) - self._baseline_face_y)

        self.last_metrics = {
            "eyes_visible": float(eyes_visible),
            "mouth_open": float(mouth_open),
            "head_drop": head_drop,
        }

        if not eyes_visible:
            self.eye_closed_frames += 1
        else:
            self.eye_closed_frames = 0

        if mouth_open:
            self.mouth_open_frames += 1
        else:
            self.mouth_open_frames = 0

        if head_drop > HEAD_DROP_PIXELS_THRESHOLD:
            self.head_drop_frames += 1
        else:
            self.head_drop_frames = 0

        raw_drowsy = (
            self.eye_closed_frames >= EYE_CLOSED_FRAMES_THRESHOLD
            or self.mouth_open_frames >= MOUTH_OPEN_FRAMES_THRESHOLD
            or self.head_drop_frames >= HEAD_DROP_FRAMES_THRESHOLD
        )

        if raw_drowsy:
            self._drowsy_votes += 1
            self._alert_votes = 0
            if self._drowsy_votes >= DROWSY_PERSISTENCE_FRAMES:
                self._is_drowsy = True
        else:
            self._alert_votes += 1
            self._drowsy_votes = 0
            if self._alert_votes >= ALERT_PERSISTENCE_FRAMES:
                self._is_drowsy = False

        return self._is_drowsy

    def summary(self) -> str:
        if not self.face_visible:
            return "face lost"
        return (
            f"eyes={int(self.last_metrics.get('eyes_visible', 0))} "
            f"mouth={int(self.last_metrics.get('mouth_open', 0))} "
            f"head_drop={self.last_metrics.get('head_drop', 0):.1f}px"
        )

    def close(self) -> None:
        return None


def detect_fatigue(frame: np.ndarray) -> bool:
    monitor = DriverFatigueMonitor()
    result = monitor.update(frame)
    monitor.close()
    return result
