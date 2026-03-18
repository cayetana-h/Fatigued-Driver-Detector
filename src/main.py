import cv2
import time
from src.gestures.state_machine import GestureGate


def detect_fatigue(frame):
    # placeholder for now (always alert)
    return False


def main():
    state = "inactive"

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    gate = GestureGate()

    last_print = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # update gesture system every frame
        activated = gate.update(rgb)

        # activation phase
        if state == "inactive":
            if activated:
                state = "active"
                print("system activated")

        # fatigue phase
        status_text = "inactive"

        if state == "active":
            fatigue = detect_fatigue(frame)

            if fatigue:
                status_text = "drowsy"
            else:
                status_text = "alert"

            # print occasionally instead of spamming
            if time.time() - last_print > 2:
                print(f"driver is {status_text}")
                last_print = time.time()

        # draw state
        cv2.putText(
            frame,
            f"state: {state}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if state == "active" else (0, 0, 255),
            2
        )

        # draw step progress
        cv2.putText(
            frame,
            f"step: {gate.step}/3",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        # draw time remaining (only during sequence)
        if gate.time_remaining is not None:
            cv2.putText(
                frame,
                f"time left: {gate.time_remaining:.1f}s",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 100, 100),
                2
            )

        # draw fatigue status
        if state == "active":
            cv2.putText(
                frame,
                f"status: {status_text}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if status_text == "alert" else (0, 0, 255),
                2
            )

        cv2.imshow("driver monitor", frame)

        # press esc to exit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    gate.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()