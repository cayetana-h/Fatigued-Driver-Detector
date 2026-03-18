import cv2
import time   
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.gestures.state_machine import GestureGate, SEQUENCE

ICONS = {"peace": "V", "fist": "[[]]", "open_palm": "◯"}

gate             = GestureGate()
cap              = cv2.VideoCapture(0)
show_until       = 0.0   

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    rgb            = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    just_activated = gate.update(rgb)

    if just_activated:
        show_until = time.monotonic() + 2.0  

    if time.monotonic() < show_until:
        cv2.putText(frame, "ACTIVATED!", (80, 220),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, (80, 200, 80), 3)

    for i, g in enumerate(SEQUENCE):
        color = (80, 200, 80)  if i < gate.step  else \
                (0,  180, 220) if i == gate.step  else \
                (100, 100, 100)
        cv2.putText(frame, ICONS[g], (40 + i * 140, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 2)

    remaining = gate.time_remaining
    if remaining is not None:
        cv2.putText(frame, f"{remaining:.1f}s remaining", (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 220), 1)

    cv2.imshow("Sequence test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
gate.close()
cv2.destroyAllWindows()