from __future__ import annotations

import argparse
import sys
import time

import cv2

from src.fatigue.basic import DriverFatigueMonitor, FatigueConfig
from src.gestures.state_machine import GestureGate
from src.utils.event_logger import CsvEventLogger


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gesture-activated driver fatigue monitor")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera-index", type=int, default=0, help="Webcam index to use")
    source.add_argument("--video-path", type=str, help="Optional recorded video path")

    parser.add_argument("--gesture-model-path", type=str, default="models/hand_landmarker.task")
    parser.add_argument("--log-path", type=str, help="Optional CSV log output path")
    parser.add_argument("--no-display", action="store_true", help="Run without OpenCV window")

    parser.add_argument("--eye-threshold", type=float, default=0.20)
    parser.add_argument("--eye-frames", type=int, default=6)
    parser.add_argument("--yawn-threshold", type=float, default=0.65)
    parser.add_argument("--yawn-frames", type=int, default=3)
    parser.add_argument("--yawn-window", type=float, default=10.0)
    parser.add_argument("--head-threshold", type=float, default=0.035)
    parser.add_argument("--head-frames", type=int, default=12)
    parser.add_argument("--face-reset-frames", type=int, default=15)

    return parser


def open_capture(args: argparse.Namespace) -> tuple[cv2.VideoCapture, str]:
    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
        source_name = args.video_path
    else:
        backend = cv2.CAP_ANY
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            backend = cv2.CAP_AVFOUNDATION
        cap = cv2.VideoCapture(args.camera_index, backend)
        source_name = f"camera:{args.camera_index}"

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source_name}")

    return cap, source_name


def main() -> None:
    args = build_arg_parser().parse_args()

    cap, source_name = open_capture(args)

    gate = GestureGate(model_path=args.gesture_model_path)
    fatigue_monitor = DriverFatigueMonitor(
        config=FatigueConfig(
            eye_aspect_ratio_threshold=args.eye_threshold,
            consecutive_closed_frames=args.eye_frames,
            yawn_ratio_threshold=args.yawn_threshold,
            yawn_frames_required=args.yawn_frames,
            yawn_alert_window_seconds=args.yawn_window,
            head_drop_threshold=args.head_threshold,
            head_drop_frames=args.head_frames,
            face_missing_reset_frames=args.face_reset_frames,
        )
    )

    logger = CsvEventLogger(args.log_path) if args.log_path else None

    state = "inactive"
    previous_status = "inactive"
    previous_face_visible = True
    last_print = 0.0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            activated = gate.update(rgb)

            if state == "inactive" and activated:
                state = "active"
                print("system activated")
                if logger:
                    logger.log(
                        frame_idx=frame_idx,
                        source=source_name,
                        state=state,
                        status="activated",
                        event="activation",
                        detail="gesture sequence completed",
                    )

            status_text = "inactive"
            detail_text = ""

            if state == "active":
                fatigue = fatigue_monitor.update(frame)
                status_text = "drowsy" if fatigue else "alert"
                detail_text = fatigue_monitor.summary()
                snapshot = fatigue_monitor.snapshot()

                if fatigue and previous_status != "drowsy":
                    print("fatigue event started")
                    if logger:
                        logger.log(
                            frame_idx=frame_idx,
                            source=source_name,
                            state=state,
                            status=status_text,
                            event="fatigue_start",
                            detail=detail_text,
                            snapshot=snapshot,
                        )

                if not fatigue and previous_status == "drowsy":
                    if logger:
                        logger.log(
                            frame_idx=frame_idx,
                            source=source_name,
                            state=state,
                            status=status_text,
                            event="fatigue_end",
                            detail=detail_text,
                            snapshot=snapshot,
                        )

                if previous_face_visible and not snapshot["face_visible"]:
                    if logger:
                        logger.log(
                            frame_idx=frame_idx,
                            source=source_name,
                            state=state,
                            status=status_text,
                            event="face_lost",
                            detail=detail_text,
                            snapshot=snapshot,
                        )

                if not previous_face_visible and snapshot["face_visible"]:
                    if logger:
                        logger.log(
                            frame_idx=frame_idx,
                            source=source_name,
                            state=state,
                            status=status_text,
                            event="face_found",
                            detail=detail_text,
                            snapshot=snapshot,
                        )

                previous_face_visible = bool(snapshot["face_visible"])
            else:
                previous_face_visible = True

            if time.time() - last_print > 2:
                print(f"source={source_name} | state={state} | status={status_text} | {detail_text}")
                last_print = time.time()

            if not args.no_display:
                cv2.putText(
                    frame,
                    f"state: {state}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if state == "active" else (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"step: {gate.step}/3",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )

                if gate.time_remaining is not None:
                    cv2.putText(
                        frame,
                        f"time left: {gate.time_remaining:.1f}s",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 100, 100),
                        2,
                    )

                if state == "active":
                    cv2.putText(
                        frame,
                        f"status: {status_text}",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0) if status_text == "alert" else (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        detail_text,
                        (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (200, 200, 50),
                        2,
                    )

                cv2.imshow("driver monitor", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            previous_status = status_text
            frame_idx += 1

    finally:
        fatigue_monitor.close()
        gate.close()
        cap.release()
        if logger:
            logger.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()