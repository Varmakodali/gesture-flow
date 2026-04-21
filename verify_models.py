import pickle
import tensorflow as tf
import os

print("=" * 60)
print("SIGN LANGUAGE SYSTEM - MODEL VERIFICATION")
print("=" * 60)

# Check landmark model
landmark_model_path = r"d:/My_projects/sign_language/models/landmark_model.pkl"
landmark_labels_path = r"d:/My_projects/sign_language/models/landmark_labels.pkl"

print("\n1. Checking Landmark Model...")
if os.path.exists(landmark_model_path):
    try:
        with open(landmark_model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"   ✓ Landmark model loaded successfully")
        print(f"   Type: {type(model).__name__}")
    except Exception as e:
        print(f"   ✗ Error loading landmark model: {e}")
else:
    print(f"   ✗ Landmark model not found at {landmark_model_path}")

print("\n2. Checking Landmark Labels...")
if os.path.exists(landmark_labels_path):
    try:
        with open(landmark_labels_path, 'rb') as f:
            labels = pickle.load(f)
        print(f"   ✓ Landmark labels loaded successfully")
        print(f"   Classes: {len(labels)} - {labels}")
    except Exception as e:
        print(f"   ✗ Error loading landmark labels: {e}")
else:
    print(f"   ✗ Landmark labels not found at {landmark_labels_path}")

# Check ASL model
asl_model_path = r"d:/My_projects/sign_language/models/asl_model.h5"

print("\n3. Checking ASL Model...")
if os.path.exists(asl_model_path):
    try:
        model = tf.keras.models.load_model(asl_model_path)
        print(f"   ✓ ASL model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"   ✗ Error loading ASL model: {e}")
else:
    print(f"   ✗ ASL model not found at {asl_model_path}")

# Check MediaPipe
print("\n4. Checking MediaPipe...")
try:
    import mediapipe as mp
    print(f"   ✓ MediaPipe installed (version {mp.__version__})")
    
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print(f"   ✓ MediaPipe Hands detector created successfully")
    else:
        print(f"   ✗ MediaPipe solutions.hands not available")
except Exception as e:
    print(f"   ✗ MediaPipe error: {e}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
