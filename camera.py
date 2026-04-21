
import cv2
import time
from realtime.predictor import SignPredictor

def run_camera():
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    try:
        predictor = SignPredictor()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        # Define ROI (Region of Interest) for hand
        # Assuming user places hand in the green box
        roi_top = 50
        roi_right = 350
        roi_bottom = 350
        roi_left = 50
        
        # Draw ROI
        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
        
        roi = frame[roi_top:roi_bottom, roi_left:roi_right]
        
        # Predict
        try:
            label, confidence = predictor.predict(roi)
            label_text = f"{label} ({confidence:.2f})"
        except Exception as e:
            label_text = "..."
            
        cv2.putText(frame, label_text, (roi_left, roi_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('ASL Realtime Predictor', frame)
        
        c = cv2.waitKey(1)
        if c == 27: # Esc to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()
