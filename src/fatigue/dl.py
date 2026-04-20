from __future__ import annotations

import collections
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EyeStateCNN(nn.Module):
    """Lightweight CNN: input is a 64x64 grayscale eye crop, output is P(closed)."""

    INPUT_SIZE = 64

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Eye crop helpers
# ---------------------------------------------------------------------------

# Wider set of landmarks gives a more stable bounding box
_LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144, 163, 7]
_RIGHT_EYE_INDICES = [362, 385, 387, 263, 380, 373, 390, 249]

_EYE_PAD = 0.35  # fractional padding around the bounding box


def _crop_eye(
    frame: np.ndarray,
    landmarks: np.ndarray,
    indices: list[int],
) -> np.ndarray | None:
    h, w = frame.shape[:2]
    pts = landmarks[indices, :2]          # (N, 2) normalised xy
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    bw, bh = x_max - x_min, y_max - y_min
    x_min = max(0.0, x_min - bw * _EYE_PAD)
    x_max = min(1.0, x_max + bw * _EYE_PAD)
    y_min = max(0.0, y_min - bh * _EYE_PAD)
    y_max = min(1.0, y_max + bh * _EYE_PAD)

    x1, y1, x2, y2 = int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (EyeStateCNN.INPUT_SIZE, EyeStateCNN.INPUT_SIZE))


def _to_tensor(eye_crop: np.ndarray) -> torch.Tensor:
    arr = eye_crop.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


# ---------------------------------------------------------------------------
# PERCLOS parameters
# ---------------------------------------------------------------------------

PERCLOS_WINDOW_FRAMES   = 300   # ~10 s at 30 fps
PERCLOS_THRESHOLD       = 0.15  # 15 % closed over the window → drowsy (standard)
SUSTAINED_CLOSED_FRAMES = 24    # ~0.8 s continuous closure → immediate alert
CLOSED_PROB_THRESHOLD   = 0.5   # CNN score above this → eye is closed


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class DLFatigueMonitor:
    """
    Deep-learning fatigue monitor.

    Replaces the hand-crafted EAR threshold with a trained CNN that classifies
    each eye crop as open or closed, then applies the PERCLOS metric.

    Interface is identical to DriverFatigueMonitor so main.py can swap between
    them with minimal changes.
    """

    def __init__(self, model_path: str | Path = "models/eye_classifier.pt") -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                "Run scripts/train_eye_classifier.py first."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = EyeStateCNN().to(self.device)
        self._model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self._model.eval()

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._closed_history: collections.deque[int] = collections.deque(
            maxlen=PERCLOS_WINDOW_FRAMES
        )
        self._sustained_closed = 0

        self.face_visible     = False
        self.last_perclos     = 0.0
        self.last_left_prob   = 0.0
        self.last_right_prob  = 0.0

    def update(self, frame: np.ndarray) -> bool:
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            self.face_visible = False
            return False

        self.face_visible = True
        lm = np.array(
            [[l.x, l.y, l.z] for l in result.multi_face_landmarks[0].landmark],
            dtype=np.float32,
        )

        left_crop  = _crop_eye(frame, lm, _LEFT_EYE_INDICES)
        right_crop = _crop_eye(frame, lm, _RIGHT_EYE_INDICES)

        if left_crop is None or right_crop is None:
            return False

        with torch.no_grad():
            self.last_left_prob  = float(self._model(_to_tensor(left_crop).to(self.device)))
            self.last_right_prob = float(self._model(_to_tensor(right_crop).to(self.device)))

        avg_prob  = (self.last_left_prob + self.last_right_prob) / 2.0
        is_closed = int(avg_prob >= CLOSED_PROB_THRESHOLD)

        self._closed_history.append(is_closed)

        if is_closed:
            self._sustained_closed += 1
        else:
            self._sustained_closed = 0

        n = len(self._closed_history)
        self.last_perclos = sum(self._closed_history) / n if n else 0.0

        sustained_alert = self._sustained_closed >= SUSTAINED_CLOSED_FRAMES
        perclos_alert   = (n == PERCLOS_WINDOW_FRAMES and self.last_perclos >= PERCLOS_THRESHOLD)

        return sustained_alert or perclos_alert

    def summary(self) -> str:
        if not self.face_visible:
            return "face lost"
        return (
            f"L={self.last_left_prob:.2f} R={self.last_right_prob:.2f} "
            f"PERCLOS={self.last_perclos:.1%}"
        )

    def close(self) -> None:
        self._face_mesh.close()
