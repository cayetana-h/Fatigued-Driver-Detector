import numpy as np
from src.gestures.classifier import classify_from_landmarks


def make_landmarks(index_up=False, middle_up=False, ring_up=False, pinky_up=False, thumb_out=False):
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[:, 1] = 0.5
    pts[:, 0] = 0.5

    finger_data = [
        (8, 6, index_up),
        (12, 10, middle_up),
        (16, 14, ring_up),
        (20, 18, pinky_up),
    ]
    for tip, pip, up in finger_data:
        pts[pip, 1] = 0.5
        pts[tip, 1] = 0.3 if up else 0.6

    pts[2, 0] = 0.4
    pts[4, 0] = 0.8 if thumb_out else 0.45
    return pts


def test_classify_peace():
    landmarks = make_landmarks(index_up=True, middle_up=True, ring_up=False, pinky_up=False)
    assert classify_from_landmarks(landmarks) == "peace"


def test_classify_fist():
    landmarks = make_landmarks(index_up=False, middle_up=False, ring_up=False, pinky_up=False)
    assert classify_from_landmarks(landmarks) == "fist"


def test_classify_open_palm():
    landmarks = make_landmarks(index_up=True, middle_up=True, ring_up=True, pinky_up=True, thumb_out=True)
    assert classify_from_landmarks(landmarks) == "open_palm"


def test_classify_unknown():
    landmarks = make_landmarks(index_up=True, middle_up=False, ring_up=False, pinky_up=False, thumb_out=False)
    assert classify_from_landmarks(landmarks) is None