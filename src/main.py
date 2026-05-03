from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.fatigue.basic import DriverFatigueMonitor
from src.fatigue.dl import DLFatigueMonitor
from src.gestures.state_machine import GestureGate


def _make_monitors(mode: str):
    classical = DriverFatigueMonitor() if mode in ("classical", "both") else None
    dl = DLFatigueMonitor() if mode in ("dl", "both") else None
    return classical, dl


def _put_lines(frame, lines: list[str], start_y: int = 120) -> None:
    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (20, start_y + idx * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )


def main(
    mode: str,
    video_path: str | None = None,
    debug: bool = False,
    start_active: bool = False,
    no_display: bool = False,
    log_every: int = 30,
    output_path: str | None = None,
):
    # Select video source
    if video_path:
        print(f"[INFO] Using video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
    else:
        print("[INFO] Using webcam")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open video source")

    writer = None
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output video writer: {output_path}")

    gate = None if start_active else GestureGate()
    classical_monitor, dl_monitor = _make_monitors(mode)

    system_active = start_active
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video or cannot read frame")
            break
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Gesture activation ---
        if not system_active:
            if gate is None:
                raise RuntimeError("Gesture gate is unavailable while system is inactive.")

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

            if debug:
                lines = []
                if classical_monitor:
                    lines.append(
                        f"classical={drowsy_classical} {classical_monitor.summary()}"
                    )
                if dl_monitor:
                    lines.append(f"dl={drowsy_dl} {dl_monitor.summary()}")
                _put_lines(frame, lines)

                if frame_idx % max(log_every, 1) == 0:
                    print(
                        f"[DEBUG] frame={frame_idx} status={status} "
                        + " | ".join(lines)
                    )

        if writer:
            writer.write(frame)

        # --- Show frame ---
        if no_display:
            continue

        cv2.imshow("Driver Fatigue Monitor", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    if writer:
        writer.release()
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show and print detector metrics while running.",
    )
    parser.add_argument(
        "--start-active",
        action="store_true",
        help="Skip gesture activation and start fatigue detection immediately.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process the video without opening an OpenCV window.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=30,
        help="Print debug metrics every N frames when --debug is enabled.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Optional path to save the annotated output video.",
    )

    args = parser.parse_args()

    main(
        mode=args.mode,
        video_path=args.video_path,
        debug=args.debug,
        start_active=args.start_active,
        no_display=args.no_display,
        log_every=args.log_every,
        output_path=args.output_path,
    )
