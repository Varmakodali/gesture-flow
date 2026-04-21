
import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = r"d:/My_projects/sign_language/models/asl_model.h5"

# Must match the training image size
IMG_SIZE = (128, 128)

class SignPredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        self.model = tf.keras.models.load_model(model_path)
        # Labels must match alphabetical folder order from flow_from_directory
        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                       'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
                       'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
                       
    def predict(self, image):
        # Preprocess: match training pipeline
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_normalized = img_resized / 255.0
        img_reshaped = np.reshape(img_normalized, (1, IMG_SIZE[0], IMG_SIZE[1], 3))
        
        prediction = self.model.predict(img_reshaped, verbose=0)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return self.labels[class_index], confidence
