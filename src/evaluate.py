from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2

from src.fatigue.basic import DriverFatigueMonitor, FatigueConfig
from src.gestures.state_machine import GestureGate
from src.utils.event_logger import CsvEventLogger


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline evaluation for driver fatigue monitor")
    parser.add_argument("--video-path", required=True, help="Recorded video to benchmark")
    parser.add_argument("--output-json", default="artifacts/eval_summary.json")
    parser.add_argument("--log-path", default="artifacts/eval_events.csv")
    parser.add_argument(
        "--labels-csv",
        help="Optional CSV with columns: frame_idx,is_drowsy for frame-level metrics",
    )
    parser.add_argument(
        "--use-gate",
        action="store_true",
        help="Require gesture activation instead of starting in active mode",
    )
    parser.add_argument("--gesture-model-path", type=str, default="models/hand_landmarker.task")

    parser.add_argument("--eye-threshold", type=float, default=0.20)
    parser.add_argument("--eye-frames", type=int, default=6)
    parser.add_argument("--yawn-threshold", type=float, default=0.65)
    parser.add_argument("--yawn-frames", type=int, default=3)
    parser.add_argument("--yawn-window", type=float, default=10.0)
    parser.add_argument("--head-threshold", type=float, default=0.035)
    parser.add_argument("--head-frames", type=int, default=12)
    parser.add_argument("--face-reset-frames", type=int, default=15)

    return parser


def load_labels(path: str | None) -> dict[int, int]:
    if not path:
        return {}

    labels: dict[int, int] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            frame_idx = int(row["frame_idx"])
            is_drowsy = int(row["is_drowsy"])
            labels[frame_idx] = is_drowsy
    return labels


def compute_binary_metrics(preds: list[int], truths: list[int]) -> dict[str, float]:
    tp = sum(1 for p, t in zip(preds, truths) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(preds, truths) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(preds, truths) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(preds, truths) if p == 0 and t == 1)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(preds) if preds else 0.0

    return {
        "num_labeled_frames": len(preds),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    labels = load_labels(args.labels_csv)

    video_path = Path(args.video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    gate = None
    if args.use_gate:
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

    logger = CsvEventLogger(args.log_path)

    frame_idx = 0
    active = not args.use_gate
    activation_frame = 0 if active else None
    fatigue_frames = 0
    face_lost_frames = 0
    fatigue_events = 0
    previous_drowsy = False

    preds: list[int] = []
    truths: list[int] = []

    t0 = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if gate is not None and not active:
                if gate.update(rgb):
                    active = True
                    activation_frame = frame_idx
                    logger.log(
                        frame_idx=frame_idx,
                        source=str(video_path),
                        state="active",
                        status="activated",
                        event="activation",
                        detail="gesture sequence completed during evaluation",
                    )

            drowsy = False
            snapshot = {}

            if active:
                drowsy = fatigue_monitor.update(frame)
                snapshot = fatigue_monitor.snapshot()

                if not snapshot["face_visible"]:
                    face_lost_frames += 1

                if drowsy:
                    fatigue_frames += 1

                if drowsy and not previous_drowsy:
                    fatigue_events += 1
                    logger.log(
                        frame_idx=frame_idx,
                        source=str(video_path),
                        state="active",
                        status="drowsy",
                        event="fatigue_start",
                        detail=fatigue_monitor.summary(),
                        snapshot=snapshot,
                    )

                if not drowsy and previous_drowsy:
                    logger.log(
                        frame_idx=frame_idx,
                        source=str(video_path),
                        state="active",
                        status="alert",
                        event="fatigue_end",
                        detail=fatigue_monitor.summary(),
                        snapshot=snapshot,
                    )

                if frame_idx in labels:
                    preds.append(int(drowsy))
                    truths.append(int(labels[frame_idx]))

            previous_drowsy = drowsy
            frame_idx += 1

    finally:
        elapsed = time.perf_counter() - t0
        fps = frame_idx / elapsed if elapsed > 0 else 0.0
        duration_s = frame_idx / fps if fps > 0 else 0.0

        summary = {
            "video_path": str(video_path),
            "total_frames": frame_idx,
            "elapsed_wall_time_s": elapsed,
            "throughput_fps": fps,
            "approx_processed_duration_s": duration_s,
            "used_gate": bool(args.use_gate),
            "activation_frame": activation_frame,
            "fatigue_frames": fatigue_frames,
            "fatigue_ratio": (fatigue_frames / frame_idx) if frame_idx else 0.0,
            "face_lost_frames": face_lost_frames,
            "fatigue_events": fatigue_events,
        }

        if preds and truths:
            summary["classification_metrics"] = compute_binary_metrics(preds, truths)

        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        logger.close()
        fatigue_monitor.close()
        if gate is not None:
            gate.close()
        cap.release()

        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()