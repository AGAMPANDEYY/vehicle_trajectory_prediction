"""
🎥 Video FPS: 30.0
📦 Total Frames: 9025
⏱️ Duration: 300.83 seconds

"""

import cv2

# Load video
video_path = r'D:\vehicle_trajectory_prediction\traffic-analysis-detection-tracking\data\0212.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"🎥 Video FPS: {fps}")
    print(f"frame shape: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"📦 Total Frames: {frame_count}")
    print(f"⏱️ Duration: {duration:.2f} seconds")

# Release video
cap.release()
