# GestureFlow: AI-Powered Sign Language Recognition & System Controller

GestureFlow is a high-performance, real-time sign language recognition system that uses computer vision and deep learning to translate American Sign Language (ASL) into digital text and system commands. It features a premium Flask-based web dashboard for live monitoring and interaction.

## 🚀 Key Features

- **Real-time Recognition:** Low-latency hand tracking and gesture classification using MediaPipe and TensorFlow.
- **Dual Prediction Engine:**
  - **Landmark-based:** Fast and lightweight prediction using hand skeleton landmarks.
  - **MobileNetV2 Transfer Learning:** High-accuracy image classification for complex signs.
- **Web Dashboard:** Interactive dark-themed UI built with Flask, providing a live video feed with overlayed predictions.
- **System Automation:** Map specific signs to system actions like launching applications, volume control, and more.
- **Performance Optimized:** Multi-threaded frame processing and prediction smoothing for a seamless experience.

## 🛠️ Tech Stack

- **Core:** Python 3.x
- **Computer Vision:** OpenCV, MediaPipe
- **Machine Learning:** TensorFlow, Scikit-learn
- **Web Framework:** Flask
- **Aesthetics:** Vanilla CSS (Modern Dark Mode)

## 📁 Project Structure

```text
sign_language/
├── data/               # Dataset for training
├── models/             # ML models and web server logic
│   ├── app.py          # Flask application server
│   ├── train.py        # Model training script (MobileNetV2)
│   └── ...             # Trained model files (.pkl, .h5)
├── realtime/           # Real-time camera and prediction modules
├── templates/          # HTML templates for the dashboard
├── run.py              # Main entry point for the application
├── verify_models.py    # System health check script
└── requirements.txt    # Project dependencies
```

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/gesture-flow.git
   cd gesture-flow
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/scripts/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

Run the project using the `run.py` script:

### **1. Real-time Camera Feed**
Launch the standalone camera recognition window:
```bash
python run.py realtime
```

### **2. Web Dashboard**
Start the Flask server and open `http://127.0.0.1:5000` in your browser:
```bash
python run.py web
```

### **3. Train the Model**
Update the model with your own data:
```bash
python run.py train
```

### **4. Verify Setup**
Check if all models and dependencies are working correctly:
```bash
python verify_models.py
```

## 🤝 Contributing

Contributions are welcome! If you have suggestions for new features or improvements, please open an issue or submit a pull request.

---
*Created with ❤️ for accessible technology.*
