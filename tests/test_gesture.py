import numpy as np

from src.gestures.classifier import classify_from_landmarks


def _make_hand(
    thumb_tip_x: float,
    index_tip_y: float,
    middle_tip_y: float,
    ring_tip_y: float,
    pinky_tip_y: float,
) -> np.ndarray:
    pts = np.zeros((21, 3), dtype=np.float32)

    # Wrist
    pts[0] = [0.5, 0.9, 0.0]

    # Thumb
    pts[2] = [0.42, 0.65, 0.0]
    pts[3] = [0.35, 0.55, 0.0]
    pts[4] = [thumb_tip_x, 0.45, 0.0]

    # Index
    pts[6] = [0.45, 0.60, 0.0]
    pts[8] = [0.45, index_tip_y, 0.0]

    # Middle
    pts[10] = [0.50, 0.62, 0.0]
    pts[12] = [0.50, middle_tip_y, 0.0]

    # Ring
    pts[14] = [0.55, 0.66, 0.0]
    pts[16] = [0.55, ring_tip_y, 0.0]

    # Pinky
    pts[18] = [0.60, 0.70, 0.0]
    pts[20] = [0.60, pinky_tip_y, 0.0]

    return pts


def test_classifies_open_palm():
    pts = _make_hand(
        thumb_tip_x=0.22,
        index_tip_y=0.28,
        middle_tip_y=0.24,
        ring_tip_y=0.30,
        pinky_tip_y=0.34,
    )
    assert classify_from_landmarks(pts) == "open_palm"


def test_classifies_peace():
    pts = _make_hand(
        thumb_tip_x=0.40,
        index_tip_y=0.28,
        middle_tip_y=0.24,
        ring_tip_y=0.80,
        pinky_tip_y=0.82,
    )
    assert classify_from_landmarks(pts) == "peace"


def test_classifies_fist():
    pts = _make_hand(
        thumb_tip_x=0.44,
        index_tip_y=0.82,
        middle_tip_y=0.84,
        ring_tip_y=0.85,
        pinky_tip_y=0.86,
    )
    assert classify_from_landmarks(pts) == "fist"