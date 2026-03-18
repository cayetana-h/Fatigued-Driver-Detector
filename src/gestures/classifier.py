from __future__ import annotations
import numpy as np
from typing import Optional


"""
This module contains the logic for classifying hand gestures based on the positions of the landmarks.

gestures to recognize:
- peace: index and middle up, ring and pinky down
- fist: all fingers down
- open palm: all fingers up and thumb out
"""


def _finger_up(pts: np.ndarray, tip: int, pip: int) -> bool:
    """checks if the tip of a finger is above its pip joint"""
    return pts[tip, 1] < pts[pip, 1]

def _thumb_out(pts: np.ndarray) -> bool:
    """
    calculating distance between thumb tip and index pip, and compare 
    to distance between index pip and wrist
    if the thumb is out, the first distance should be larger than the second one
    """
    return abs(pts[4, 0] - pts[2, 0]) > 0.06

def classify_from_landmarks(pts: np.ndarray) -> Optional[str]:
    """classifies the hand gesture based on the positions of the landmarks"""
    index  = _finger_up(pts, 8,  6) 
    middle = _finger_up(pts, 12, 10)
    ring   = _finger_up(pts, 16, 14)
    pinky  = _finger_up(pts, 20, 18)
    thumb  = _thumb_out(pts)

    if index and middle and not ring and not pinky:
        return "peace"
    if not index and not middle and not ring and not pinky:
        return "fist"
    if index and middle and ring and pinky and thumb:
        return "open_palm"
    return None