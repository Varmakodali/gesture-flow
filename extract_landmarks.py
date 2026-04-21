"""
Extract hand landmarks from ASL training images using MediaPipe.
Landmarks are normalized by POSITION and SCALE for consistency
between training images and webcam input.
"""
import os
import cv2
import numpy as np
import mediapipe as mp
import csv

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

DATA_DIR = r"d:/sign_language/data/asl_alphabet_test/asl_alphabet_train/asl_alphabet_train"
OUTPUT_CSV = r"d:/sign_language/models/landmark_data.csv"

SAMPLES_PER_CLASS = 300  # More samples = better accuracy

def normalize_landmarks(hand_landmarks_list):
    """
    Normalize landmarks to be position AND scale invariant.
    1. Subtract wrist position (position invariant)
    2. Divide by hand size (scale invariant)
    """
    wrist = hand_landmarks_list[0]
    
    # Calculate hand size: max distance from wrist to any finger tip
    tip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
    max_dist = 0.001  # avoid division by zero
    for tip_idx in tip_indices:
        tip = hand_landmarks_list[tip_idx]
        dist = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2) ** 0.5
        max_dist = max(max_dist, dist)
    
    features = []
    for lm in hand_landmarks_list:
        features.extend([
            (lm.x - wrist.x) / max_dist,
            (lm.y - wrist.y) / max_dist,
            (lm.z - wrist.z) / max_dist
        ])
    return features


def extract_landmarks():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    classes = sorted(os.listdir(DATA_DIR))
    print(f"Found {len(classes)} classes: {classes}")

    rows = []
    total_processed = 0
    total_detected = 0

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = os.listdir(class_dir)
        sample_count = min(SAMPLES_PER_CLASS, len(images))
        sampled = np.random.choice(images, sample_count, replace=False)
        
        detected_count = 0

        for img_name in sampled:
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            total_processed += 1
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                features = normalize_landmarks(hand.landmark)
                features.append(class_name)
                rows.append(features)
                detected_count += 1
                total_detected += 1

                # Also add FLIPPED version for robustness
                # (webcam is mirrored, training images may not be)
                img_flipped = cv2.flip(img, 1)
                rgb_flipped = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB)
                results_flipped = hands.process(rgb_flipped)
                if results_flipped.multi_hand_landmarks:
                    hand_flipped = results_flipped.multi_hand_landmarks[0]
                    features_flipped = normalize_landmarks(hand_flipped.landmark)
                    features_flipped.append(class_name)
                    rows.append(features_flipped)
                    total_detected += 1

        print(f"  [{class_idx+1}/{len(classes)}] {class_name}: {detected_count}/{sample_count} hands detected")

    hands.close()

    # Write to CSV
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n{'='*50}")
    print(f"  Landmark extraction complete!")
    print(f"  Processed: {total_processed} images")
    print(f"  Samples saved: {len(rows)} (including flipped)")
    print(f"  Saved to: {OUTPUT_CSV}")
    print(f"{'='*50}")

if __name__ == "__main__":
    extract_landmarks()
