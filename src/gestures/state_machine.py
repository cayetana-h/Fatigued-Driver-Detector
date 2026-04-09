from __future__ import annotations

import logging
import time
from enum import Enum, auto
from typing import Optional

import numpy as np

from src.gestures.classifier import classify_from_landmarks

log = logging.getLogger(__name__)

SEQUENCE = ["open_palm", "peace", "fist"]
TIME_WINDOW = 8.0
HOLD_FRAMES = 8
COOLDOWN = 2.0
DEFAULT_MODEL_PATH = "models/hand_landmarker.task"


class _S(Enum):
    IDLE = auto()
    COLLECTING = auto()
    COOLDOWN = auto()
    DONE = auto()


class GestureGate:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self.mp = mp
        self._frame_idx = 0

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._reset()

    @property
    def activated(self) -> bool:
        return self._state == _S.DONE

    @property
    def step(self) -> int:
        return self._step

    @property
    def time_remaining(self) -> Optional[float]:
        if self._state != _S.COLLECTING or self._t0 is None:
            return None
        return max(0.0, TIME_WINDOW - (time.monotonic() - self._t0))

    def update(self, rgb_frame: np.ndarray) -> bool:
        now = time.monotonic()

        if self._state == _S.DONE:
            return False

        if self._state == _S.COOLDOWN:
            if now - self._cooldown_start >= COOLDOWN:
                self._reset()
            return False

        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(self._frame_idx * (1000 / 30))
        self._frame_idx += 1
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        gesture = None
        if result.hand_landmarks:
            pts = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
            gesture = classify_from_landmarks(pts)

        return self.process_gesture(gesture, now)

    def process_gesture(self, gesture: Optional[str], now: Optional[float] = None) -> bool:
        if now is None:
            now = time.monotonic()
        return self._advance(gesture, now)

    def close(self) -> None:
        self._landmarker.close()

    def _reset(self) -> None:
        self._state = _S.IDLE
        self._step = 0
        self._t0 = None
        self._hold_count = 0
        self._last_gesture = None
        self._cooldown_start = 0.0

    def _timeout(self) -> None:
        log.debug("Sequence timed out")
        self._state = _S.COOLDOWN
        self._cooldown_start = time.monotonic()
        self._step = 0
        self._hold_count = 0
        self._last_gesture = None

    def _advance(self, gesture: Optional[str], now: float) -> bool:
        if self._state == _S.COLLECTING and now - self._t0 > TIME_WINDOW:
            self._timeout()
            return False

        target = SEQUENCE[self._step]

        if gesture == target:
            self._hold_count = self._hold_count + 1 if gesture == self._last_gesture else 1
            self._last_gesture = gesture

            if self._hold_count >= HOLD_FRAMES:
                if self._state == _S.IDLE:
                    self._state = _S.COLLECTING
                    self._t0 = now

                self._step += 1
                self._hold_count = 0
                self._last_gesture = None
                log.debug("Accepted step %d: %s", self._step, target)

                if self._step == len(SEQUENCE):
                    self._state = _S.DONE
                    return True
        else:
            if gesture != self._last_gesture:
                self._hold_count = 0
                self._last_gesture = gesture

        return False  