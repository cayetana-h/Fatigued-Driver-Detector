import cv2
import numpy as np
import sys
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.gestures.classifier import classify_from_landmarks

MODEL_PATH = "models/hand_landmarker.task" 

# hand landmarker setup
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
# the hand landmarker detects hand landmarks in the video feed (21 points per hand) and returns their normalized positions
landmarker = mp_vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
print("Show gestures in front of the camera. Q to quit.")
frame_idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(frame_idx * (1000 / 30))   
    frame_idx += 1

    result  = landmarker.detect_for_video(mp_image, timestamp_ms)
    gesture = None

    # if a hand is detected, we get the normalized positions of the landmarks and classify the gesture
    if result.hand_landmarks:
        lm  = result.hand_landmarks[0]   
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        gesture = classify_from_landmarks(pts)

        h, w = frame.shape[:2]
        for p in lm:
            cx, cy = int(p.x * w), int(p.y * h)
            cv2.circle(frame, (cx, cy), 3, (80, 200, 80), -1)

    label = gesture or "---"
    color = (80, 200, 80) if gesture else (160, 160, 160)
    cv2.putText(frame, label, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 2)

    cv2.imshow("Gesture test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()