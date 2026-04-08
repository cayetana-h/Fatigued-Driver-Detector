import time
from src.gestures.state_machine import GestureGate, SEQUENCE


def test_gate_activates_after_correct_sequence():
    gate = GestureGate()
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
    gate.close()


def test_gate_rejects_wrong_first_gesture():
    gate = GestureGate()
    now = time.monotonic()

    assert not gate.process_gesture("peace", now=now)
    assert gate.step == 0
    gate.close()