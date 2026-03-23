import time

_start = time.time()

def detect_fatigue(frame):
    # fake fatigue after 10 seconds
    return (time.time() - _start) > 10