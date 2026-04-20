import argparse
import time

import cv2

from src.fatigue.basic import DriverFatigueMonitor
from src.gestures.state_machine import GestureGate


def _make_monitors(mode: str):
    """Return the fatigue monitor(s) to run based on --mode."""
    classical = DriverFatigueMonitor() if mode in ("classical", "both") else None

    dl = None
    if mode in ("dl", "both"):
        from src.fatigue.dl import DLFatigueMonitor
        dl = DLFatigueMonitor()

    return classical, dl


def _drowsy_label(classical_result, dl_result, mode: str) -> tuple[bool, str]:
    """Combine results and return (is_drowsy, display_label)."""
    if mode == "classical":
        return classical_result, "classical"
    if mode == "dl":
        return dl_result, "dl"
    # both — flag drowsy if either pipeline agrees
    return (classical_result or dl_result), "both"


def main(mode: str = "classical") -> None:
    state = "inactive"
    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
    gate = GestureGate()

    classical_monitor, dl_monitor = _make_monitors(mode)

    last_print = 0.0
    classical_drowsy = False
    dl_drowsy = False

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
        detail_lines: list[str] = []

        if state == "active":
            if classical_monitor is not None:
                classical_drowsy = classical_monitor.update(frame)
                detail_lines.append(f"[classical] {classical_monitor.summary()}")

            if dl_monitor is not None:
                dl_drowsy = dl_monitor.update(frame)
                detail_lines.append(f"[dl]        {dl_monitor.summary()}")

            is_drowsy, _ = _drowsy_label(classical_drowsy, dl_drowsy, mode)
            status_text = "drowsy" if is_drowsy else "alert"

            if time.time() - last_print > 2:
                print(f"driver is {status_text}")
                for line in detail_lines:
                    print(" ", line)
                last_print = time.time()

        # --- overlay ---
        status_color = (0, 255, 0) if state == "active" else (0, 0, 255)
        cv2.putText(frame, f"state: {state}  mode: {mode}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"step: {gate.step}/3", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if gate.time_remaining is not None:
            cv2.putText(frame, f"time left: {gate.time_remaining:.1f}s", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)

        if state == "active":
            alert_color = (0, 0, 255) if status_text == "drowsy" else (0, 255, 0)
            cv2.putText(frame, f"status: {status_text}", (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, alert_color, 2)

            y = 180
            for line in detail_lines:
                cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 200, 50), 1)
                y += 28

        cv2.imshow("driver monitor", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if classical_monitor is not None:
        classical_monitor.close()
    if dl_monitor is not None:
        dl_monitor.close()
    gate.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fatigued Driver Detector")
    parser.add_argument(
        "--mode",
        choices=["classical", "dl", "both"],
        default="classical",
        help="Which fatigue pipeline to run (default: classical)",
    )
    args = parser.parse_args()
    main(mode=args.mode)
