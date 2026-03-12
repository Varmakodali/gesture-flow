# gesture-flow 🤟

A real-time **American Sign Language (ASL) letter detection system** built with deep learning and computer vision. The system recognizes hand gestures through a live camera feed and translates them into text — helping bridge the communication gap for the hearing-impaired community.

---

## Demo

> Real-time hand tracking → Letter prediction → Text output via Flask web dashboard

---

## Features

- Real-time ASL alphabet detection using a webcam
- Hand tracking and landmark extraction
- Deep learning model for letter classification
- Flask-based web dashboard to display predictions live
- Lightweight and runs on a standard laptop

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| OpenCV | Real-time video capture and image processing |
| MediaPipe | Hand landmark detection and tracking |
| TensorFlow / Keras | Deep learning model training and inference |
| Flask | Web dashboard for live output |
| NumPy | Data processing |

---

## How It Works

1. **Camera Input** — The webcam captures a live video stream frame by frame
2. **Hand Detection** — MediaPipe detects the hand and extracts 21 key landmark points
3. **Preprocessing** — Landmark coordinates are normalized and fed into the model
4. **Prediction** — A trained CNN classifies the hand gesture as an ASL letter (A–Z)
5. **Output** — The predicted letter is displayed on the Flask web dashboard in real time

---

## Project Structure

```
gesture-flow/
│
├── model/
│   └── asl_model.h5          # Trained deep learning model
│
├── static/
│   └── style.css             # Dashboard styling
│
├── templates/
│   └── index.html            # Flask web dashboard UI
│
├── app.py                    # Main Flask application
├── train.py                  # Model training script
├── utils.py                  # Helper functions
└── requirements.txt          # Python dependencies
```

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/Varmakodali/gesture-flow.git
cd gesture-flow

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then open your browser and go to `http://localhost:5000`

---

## Model Details

- Architecture: Convolutional Neural Network (CNN)
- Input: 21 hand landmark coordinates (x, y, z)
- Output: 26 classes (A–Z ASL letters)
- Training data: Custom dataset of hand gesture images

---

## Real-World Impact

This project is designed to assist the **hearing-impaired and speech-impaired community** by enabling real-time sign language interpretation. It can be extended to full word and sentence recognition in future versions.

---

## Author

**Nikesh Varma Kodali**
B.Tech – Computer Science & Engineering
Ramachandra College of Engineering, Eluru, Andhra Pradesh, India
GitHub: [@Varmakodali](https://github.com/Varmakodali)

---

## License

This project is open source and available under the [MIT License](LICENSE).
