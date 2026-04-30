from __future__ import annotations

import argparse
import cv2

from src.fatigue.basic import DriverFatigueMonitor
from src.fatigue.dl import DLFatigueMonitor
from src.gestures.state_machine import GestureGate


def _make_monitors(mode: str):
    classical = DriverFatigueMonitor() if mode in ("classical", "both") else None
    dl = DLFatigueMonitor() if mode in ("dl", "both") else None
    return classical, dl


def main(mode: str, video_path: str | None = None):
    # Select video source
    if video_path:
        print(f"[INFO] Using video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
    else:
        print("[INFO] Using webcam")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    gate = GestureGate()
    classical_monitor, dl_monitor = _make_monitors(mode)

    system_active = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video or cannot read frame")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Gesture activation ---
        if not system_active:
            activated = gate.update(rgb)

            if activated:
                system_active = True
                print("[INFO] System activated!")

            cv2.putText(
                frame,
                f"STATE: INACTIVE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # --- Fatigue detection ---
        else:
            drowsy_classical = False
            drowsy_dl = False

            if classical_monitor:
                drowsy_classical = classical_monitor.update(frame)

            if dl_monitor:
                drowsy_dl = dl_monitor.update(frame)

            # Combine results
            if mode == "classical":
                drowsy = drowsy_classical
            elif mode == "dl":
                drowsy = drowsy_dl
            else:
                drowsy = drowsy_classical or drowsy_dl

            status = "DROWSY" if drowsy else "ALERT"

            cv2.putText(
                frame,
                f"STATE: ACTIVE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"STATUS: {status}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255) if drowsy else (0, 255, 0),
                2,
            )

        # --- Show frame ---
        cv2.imshow("Driver Fatigue Monitor", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["classical", "dl", "both"],
        default="classical",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        help="Path to video file instead of webcam",
    )

    args = parser.parse_args()

    main(mode=args.mode, video_path=args.video_path)