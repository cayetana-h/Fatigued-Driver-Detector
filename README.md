# Fatigued Driver Detector

## Project overview
This repository implements a gesture-based activation system for driver fatigue monitoring.
The system remains inactive until the driver performs a sequence of three hand gestures, then uses visual cues from a face mesh to identify driver drowsiness.

## What the project implements
- Gesture sequence activation: `open_palm`, `peace`, `fist` within a time window
- Fatigue cues based on eye closure, yawning, and head drop
- Live webcam pipeline with visualization and activation state

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
python src/main.py
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
