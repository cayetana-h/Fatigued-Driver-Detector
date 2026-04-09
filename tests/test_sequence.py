import time

from src.gestures.state_machine import GestureGate, SEQUENCE


def _make_logic_only_gate() -> GestureGate:
    """
    Build a GestureGate instance without running __init__,
    so tests don't load the MediaPipe model.
    """
    gate = GestureGate.__new__(GestureGate)
    gate._reset()
    return gate


def test_gate_activates_after_correct_sequence():
    gate = _make_logic_only_gate()
    now = time.monotonic()

    assert not gate.activated
    assert gate.step == 0

    for gesture in SEQUENCE:
        for _ in range(8):
            activated = gate.process_gesture(gesture, now=now)
            now += 0.05

        if gesture != SEQUENCE[-1]:
            assert not activated

    assert gate.activated


def test_gate_rejects_wrong_first_gesture():
    gate = _make_logic_only_gate()
    now = time.monotonic()

    assert not gate.process_gesture("peace", now=now)
    assert gate.step == 0