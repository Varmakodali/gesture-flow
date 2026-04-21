import cv2
import numpy as np
from collections import deque, Counter
from flask import Flask, render_template, Response, jsonify
import sys
import os
import pickle
import atexit
import signal
import time
import threading

# Force stable Protobuf behavior
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ---- Global camera reference for cleanup ----
active_camera = None

# ---- Shared prediction state (thread-safe) ----
prediction_lock = threading.Lock()
current_prediction = {
    "label": "",
    "confidence": 0.0,
    "sentence": ""
}


def cleanup_camera():
    """Release camera on exit to prevent it from staying locked."""
    global active_camera
    if active_camera is not None and active_camera.isOpened():
        active_camera.release()
        print("[CLEANUP] Camera released successfully.")
    active_camera = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n[INFO] Shutting down...")
    cleanup_camera()
    sys.exit(0)


# Register cleanup handlers
atexit.register(cleanup_camera)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, template_folder='../templates')

# ---- Paths ----
LANDMARK_MODEL_PATH = r"d:/My_projects/sign_language/models/landmark_model.pkl"
LANDMARK_LABELS_PATH = r"d:/My_projects/sign_language/models/landmark_labels.pkl"

# ---- MediaPipe Loading ----
USE_MEDIAPIPE = False
hands_detector = None
mp_hands_module = None
mp_draw_module = None

try:
    import mediapipe as mp
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
        mp_hands_module = mp.solutions.hands
        mp_draw_module = mp.solutions.drawing_utils
        hands_detector = mp_hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        USE_MEDIAPIPE = True
        print("[OK] MediaPipe Hands loaded successfully!")
    else:
        print("[WARN] MediaPipe 'solutions' not available.")
except Exception as e:
    print(f"[WARN] MediaPipe not available: {e}")

# ---- Load Landmark Model ----
landmark_model = None
landmark_labels = None

try:
    with open(LANDMARK_MODEL_PATH, 'rb') as f:
        landmark_model = pickle.load(f)
    with open(LANDMARK_LABELS_PATH, 'rb') as f:
        landmark_labels = pickle.load(f)
    print(f"[OK] Landmark model loaded! ({len(landmark_labels)} classes)")
except Exception as e:
    print(f"[WARN] Landmark model not found: {e}")
    print("[INFO] Run: python extract_landmarks.py && python train_landmarks.py")


def predict_from_landmarks(hand_landmarks):
    """Predict ASL letter from MediaPipe hand landmarks directly."""
    if landmark_model is None or landmark_labels is None:
        return None, 0.0

    # Extract features: normalize by position AND scale
    wrist = hand_landmarks.landmark[0]
    
    # Calculate hand size: max distance from wrist to any fingertip
    tip_indices = [4, 8, 12, 16, 20]
    max_dist = 0.001  # avoid division by zero
    for tip_idx in tip_indices:
        tip = hand_landmarks.landmark[tip_idx]
        dist = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2) ** 0.5
        max_dist = max(max_dist, dist)
    
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([
            (lm.x - wrist.x) / max_dist,
            (lm.y - wrist.y) / max_dist,
            (lm.z - wrist.z) / max_dist
        ])

    features = np.array(features).reshape(1, -1)

    # Predict with probability
    prediction = landmark_model.predict(features)[0]
    probabilities = landmark_model.predict_proba(features)[0]
    confidence = np.max(probabilities)

    return prediction, confidence


def generate_frames():
    global active_camera

    # Release any previously locked camera
    if active_camera is not None and active_camera.isOpened():
        active_camera.release()
        time.sleep(0.5)

    # Try to open camera with retries
    cap = None
    for attempt in range(3):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            break
        print(f"[WARN] Camera open attempt {attempt + 1}/3 failed. Retrying in 2s...")
        cap.release()
        time.sleep(2)

    if cap is None or not cap.isOpened():
        print("[ERROR] Could not open camera after 3 attempts.")
        print("[TIP]  Make sure no other app is using the camera.")
        print("[TIP]  Try closing other Python processes: taskkill /F /IM python.exe")
        return

    # Track camera globally for cleanup on exit
    active_camera = cap

    # Set standard resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[OK] Video stream started. Open http://127.0.0.1:5000 in your browser.")

    # Prediction smoothing
    prediction_history = deque(maxlen=5)
    stable_label = ""
    stable_confidence = 0.0
    no_hand_count = 0
    frame_count = 0
    fps_start_time = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_count += 1
            
            # Skip every other frame for better performance
            if frame_count % 2 != 0:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue

            if USE_MEDIAPIPE and hands_detector is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands_detector.process(rgb_frame)

                hand_found = False
                if results.multi_hand_landmarks:
                    hand_found = True
                    for hand_landmarks in results.multi_hand_landmarks:
                        # ---- PREDICT FROM LANDMARKS (not image!) ----
                        try:
                            label, confidence = predict_from_landmarks(hand_landmarks)
                            if label and confidence > 0.3:
                                prediction_history.append(label)
                                no_hand_count = 0
                                # Log only high-confidence predictions to reduce spam
                                if confidence > 0.6:
                                    print(f"[PREDICT] {label} ({confidence*100:.0f}%)")
                        except Exception as e:
                            print(f"[ERROR] Prediction failed: {e}")

                        # ---- Get bounding box for display ----
                        x_max, y_max = 0, 0
                        x_min, y_min = w, h
                        for lm in hand_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            x_min, x_max = min(x_min, x), max(x_max, x)
                            y_min, y_max = min(y_min, y), max(y_max, y)

                        offset = 30
                        x_min_box = max(0, x_min - offset)
                        y_min_box = max(0, y_min - offset)
                        x_max_box = min(w, x_max + offset)
                        y_max_box = min(h, y_max + offset)

                        # Draw skeleton (red dots, white connections)
                        mp_draw_module.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands_module.HAND_CONNECTIONS,
                            mp_draw_module.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=6),
                            mp_draw_module.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                        )

                        # Draw green bounding box
                        cv2.rectangle(frame, (x_min_box, y_min_box), (x_max_box, y_max_box), (0, 255, 0), 3)

                if not hand_found:
                    no_hand_count += 1
                    if no_hand_count == 15:
                        print("[INFO] No hand detected for 15 frames, clearing predictions")
                    if no_hand_count > 10:
                        prediction_history.clear()
                        stable_label = ""
                        stable_confidence = 0.0

            # ---- Majority Vote ----
            if len(prediction_history) >= 3:
                vote = Counter(prediction_history).most_common(1)[0]
                stable_label = vote[0]
                stable_confidence = vote[1] / len(prediction_history)

            # ---- Update shared prediction state ----
            with prediction_lock:
                current_prediction["label"] = stable_label
                current_prediction["confidence"] = round(stable_confidence * 100, 1)

            # ---- Overlay: Prediction text ----
            if stable_label and stable_confidence > 0.5:
                # Shadow
                cv2.putText(frame, stable_label, (32, 202), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 6)
                # Main text
                cv2.putText(frame, stable_label, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 50, 0), 4)
                # Stability
                stab_text = f"Stability: {stable_confidence * 100:.0f}%"
                cv2.putText(frame, stab_text, (32, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            
            # ---- FPS Counter ----
            elapsed = time.time() - fps_start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_text, (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        active_camera = None
        print("[CLEANUP] Camera released.")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/prediction')
def prediction():
    with prediction_lock:
        data = dict(current_prediction)
    return jsonify(data)


def run_flask():
    print("=" * 50)
    print("  Sign Language Recognition Server")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)


if __name__ == "__main__":
    run_flask()
