from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from src.fatigue.basic import DriverFatigueMonitor


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def infer_label_from_name(video_path: Path) -> str:
	name = video_path.stem.lower()
	if "drowsy" in name:
		return "drowsy"
	if "alert" in name:
		return "alert"
	return "unknown"


def safe_div(num: float, den: float) -> float:
	return num / den if den else 0.0


def classify_video(drowsy_ratio: float, threshold: float) -> str:
	return "drowsy" if drowsy_ratio >= threshold else "alert"


def evaluate_video(video_path: Path, sample_every: int, ratio_threshold: float) -> dict[str, object]:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"Could not open video: {video_path}")

	fps = cap.get(cv2.CAP_PROP_FPS)
	fps = fps if fps and fps > 0 else 30.0

	monitor = DriverFatigueMonitor()

	frame_idx = 0
	processed_frames = 0
	drowsy_frames = 0
	first_drowsy_frame: int | None = None
	processing_times_ms: list[float] = []

	while True:
		ok, frame = cap.read()
		if not ok:
			break

		if frame_idx % sample_every != 0:
			frame_idx += 1
			continue

		start = time.perf_counter()
		is_drowsy = monitor.update(frame)
		elapsed_ms = (time.perf_counter() - start) * 1000.0

		processed_frames += 1
		processing_times_ms.append(elapsed_ms)
		if is_drowsy:
			drowsy_frames += 1
			if first_drowsy_frame is None:
				first_drowsy_frame = frame_idx

		frame_idx += 1

	cap.release()
	monitor.close()

	drowsy_ratio = safe_div(drowsy_frames, processed_frames)
	predicted_label = classify_video(drowsy_ratio, ratio_threshold)
	inferred_label = infer_label_from_name(video_path)

	return {
		"video": str(video_path),
		"fps": round(fps, 3),
		"processed_frames": processed_frames,
		"drowsy_frames": drowsy_frames,
		"drowsy_ratio": round(drowsy_ratio, 4),
		"first_drowsy_sec": round(first_drowsy_frame / fps, 3) if first_drowsy_frame is not None else "",
		"avg_processing_ms": round(safe_div(sum(processing_times_ms), len(processing_times_ms)), 3),
		"inferred_label": inferred_label,
		"predicted_label": predicted_label,
	}


def compute_classification_metrics(rows: list[dict[str, object]]) -> dict[str, float] | None:
	labeled = [r for r in rows if r["inferred_label"] in {"alert", "drowsy"}]
	if not labeled:
		return None

	tp = sum(1 for r in labeled if r["inferred_label"] == "drowsy" and r["predicted_label"] == "drowsy")
	tn = sum(1 for r in labeled if r["inferred_label"] == "alert" and r["predicted_label"] == "alert")
	fp = sum(1 for r in labeled if r["inferred_label"] == "alert" and r["predicted_label"] == "drowsy")
	fn = sum(1 for r in labeled if r["inferred_label"] == "drowsy" and r["predicted_label"] == "alert")

	accuracy = safe_div(tp + tn, len(labeled))
	precision = safe_div(tp, tp + fp)
	recall = safe_div(tp, tp + fn)
	f1 = safe_div(2 * precision * recall, precision + recall)

	return {
		"samples": float(len(labeled)),
		"tp": float(tp),
		"tn": float(tn),
		"fp": float(fp),
		"fn": float(fn),
		"accuracy": round(accuracy, 4),
		"precision": round(precision, 4),
		"recall": round(recall, 4),
		"f1": round(f1, 4),
	}


def discover_videos(data_dir: Path) -> list[Path]:
	videos = [p for p in sorted(data_dir.rglob("*")) if p.suffix.lower() in VIDEO_EXTENSIONS]
	return videos


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"video",
		"fps",
		"processed_frames",
		"drowsy_frames",
		"drowsy_ratio",
		"first_drowsy_sec",
		"avg_processing_ms",
		"inferred_label",
		"predicted_label",
	]
	with output_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate fatigue detection on recorded videos.")
	parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Directory containing test videos")
	parser.add_argument(
		"--output-csv",
		type=Path,
		default=Path("outputs/evaluation/fatigue_eval.csv"),
		help="Output CSV path",
	)
	parser.add_argument(
		"--sample-every",
		type=int,
		default=2,
		help="Process every Nth frame to speed up offline evaluation",
	)
	parser.add_argument(
		"--ratio-threshold",
		type=float,
		default=0.25,
		help="Classify a video as drowsy if drowsy_ratio >= threshold",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.sample_every < 1:
		raise ValueError("--sample-every must be >= 1")

	if not args.data_dir.exists():
		raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

	videos = discover_videos(args.data_dir)
	if not videos:
		raise FileNotFoundError(f"No videos found under: {args.data_dir}")

	rows: list[dict[str, object]] = []
	for video_path in videos:
		print(f"Evaluating {video_path} ...")
		rows.append(evaluate_video(video_path, args.sample_every, args.ratio_threshold))

	write_csv(args.output_csv, rows)
	print(f"Saved per-video results to: {args.output_csv}")

	metrics = compute_classification_metrics(rows)
	if metrics is None:
		print("No labels inferred from filenames. Add 'alert' or 'drowsy' in filenames for aggregate metrics.")
		return

	print("Aggregate metrics (from filename labels):")
	print(
		"samples={samples:.0f} tp={tp:.0f} tn={tn:.0f} fp={fp:.0f} fn={fn:.0f} "
		"acc={accuracy:.4f} prec={precision:.4f} rec={recall:.4f} f1={f1:.4f}".format(**metrics)
	)


if __name__ == "__main__":
	main()
