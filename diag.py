import os
import sys
import importlib.util

print(f"Python Version: {sys.version}")
print(f"Sys Path: {sys.path}")

try:
    import mediapipe as mp
    print(f"MediaPipe Path: {mp.__file__}")
    print(f"MediaPipe Dir: {os.path.dirname(mp.__file__)}")
    print(f"MediaPipe Attributes: {dir(mp)}")
    
    pkg_dir = os.path.dirname(mp.__file__)
    print(f"\nFiles in MediaPipe Dir: {os.listdir(pkg_dir)}")
    
    if os.path.exists(os.path.join(pkg_dir, 'python')):
        print(f"Python Subdir Files: {os.listdir(os.path.join(pkg_dir, 'python'))}")
        if os.path.exists(os.path.join(pkg_dir, 'python', 'solutions')):
            print(f"Solutions Subdir Files: {os.listdir(os.path.join(pkg_dir, 'python', 'solutions'))}")
            
except Exception as e:
    print(f"Error during diagnostic: {e}")

try:
    import google.protobuf
    print(f"\nProtobuf Version: {google.protobuf.__version__ if hasattr(google.protobuf, '__version__') else 'unknown'}")
    print(f"Protobuf Path: {google.protobuf.__file__}")
except Exception as e:
    print(f"Protobuf Error: {e}")
