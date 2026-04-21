
import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [train|realtime]")
        return

    command = sys.argv[1]

    if command == "train":
        from models.train import train_model
        print("Starting training...")
        train_model()
    elif command == "realtime":
        from realtime.camera import run_camera
        print("Starting realtime predictor...")
        run_camera()
    elif command == "web":
        from models.app import run_flask
        print("Starting Flask web server...")
        run_flask()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python run.py [train|realtime|web]")

if __name__ == "__main__":
    main()
