# Fatigued Driver Detector

## Project overview
This repository implements a gesture-based activation system for driver fatigue monitoring.

The system remains inactive until the driver performs the gesture sequence:

`open_palm -> peace -> fist`

After activation, the system uses face landmarks to detect fatigue based on:
- prolonged eye closure
- yawning
- head drop

## Features
- Gesture-sequence activation
- Real-time webcam pipeline
- Fatigue monitoring with face mesh
- CSV event logging
- CLI configuration for reproducibility
- Offline evaluation on recorded video

## Setup
Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt