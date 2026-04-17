# Fatigued Driver Detector

## Project overview
This repository implements a gesture-based activation system for driver fatigue monitoring.
The system remains inactive until the driver performs a sequence of three hand gestures, then uses visual fatigue cues to identify driver drowsiness.

## What the project implements
- Gesture sequence activation: `open_palm`, `peace`, `fist` within a time window
- Fatigue cues based on eye visibility, mouth-open proxy, and head drop
- Live webcam pipeline with visualization and activation state

## Dataset protocol note
- Course-compliant setup can use a public dataset for model development/training and a separate team-collected dataset for testing.
- This repository currently focuses on inference and real-time testing; if required by your instructor, include your team testing clips/images under `data/raw` and document the split in your report.

Recommended structure for testing videos:
```text
data/raw/
	test/
		alert_driver01.mp4
		alert_driver02.mov
		drowsy_driver01.mp4
		drowsy_driver02.mov
```

The evaluator infers labels from filenames containing `alert` or `drowsy`.

## Setup
1. Create and activate a Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Run
From the repository root:
```bash
python3 -m src.main
```

## Usage
1. Show the gestures in order: `open_palm` → `peace` → `fist`
2. Complete the sequence within about 8 seconds
3. After activation, the dashboard will evaluate drowsiness using face cues
4. Press `Esc` to exit

## Testing
Run the unit tests using the installed Python environment:
```bash
python3 -m pytest
```

## Offline evaluation on your own test dataset
Run reproducible evaluation on recorded team videos:
```bash
python3 -m scripts.evaluate --data-dir data/raw/test --output-csv outputs/evaluation/fatigue_eval.csv
```

Optional tuning flags:
```bash
python3 -m scripts.evaluate --data-dir data/raw/test --sample-every 2 --ratio-threshold 0.25
```

Outputs:
- Per-video CSV with frame-level summary (`drowsy_ratio`, `first_drowsy_sec`, `avg_processing_ms`).
- Aggregate classification metrics (accuracy/precision/recall/F1) when labels are available in filenames.
