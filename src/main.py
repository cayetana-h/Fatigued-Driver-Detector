import cv2
import time
from src.fatigue.basic import DriverFatigueMonitor
from src.gestures.state_machine import GestureGate


def main():
    state = "inactive"
    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    gate = GestureGate()
    fatigue_monitor = DriverFatigueMonitor()

    last_print = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        activated = gate.update(rgb)

        if state == "inactive" and activated:
            state = "active"
            print("system activated")

        status_text = "inactive"
        detail_text = ""

        if state == "active":
            fatigue = fatigue_monitor.update(frame)
            status_text = "drowsy" if fatigue else "alert"
            detail_text = fatigue_monitor.summary()

            if time.time() - last_print > 2:
                print(f"driver is {status_text} | {detail_text}")
                last_print = time.time()

        cv2.putText(frame, f"state: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if state == "active" else (0, 0, 255), 2)
        cv2.putText(frame, f"step: {gate.step}/3", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)

        if gate.time_remaining is not None:
            cv2.putText(frame, f"time left: {gate.time_remaining:.1f}s", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2)

        if state == "active":
            cv2.putText(frame, f"status: {status_text}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if status_text == "alert" else (0, 0, 255), 2)
            cv2.putText(frame, detail_text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (200, 200, 50), 2)

        cv2.imshow("driver monitor", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    fatigue_monitor.close()
    gate.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
