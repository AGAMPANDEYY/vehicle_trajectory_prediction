import numpy as np
import os
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the dataset
dataset = np.load(os.path.join(BASE_DIR, "data", "sdtatt_data.npy"), allow_pickle=True)

# Load the video
video_path = r"C:\Agam\Work\vehicle_trajectory_prediction\traffic-analysis-detection-tracking\data\0212_cropped.mp4"
cap = cv2.VideoCapture(video_path)

print(f"Total samples: {len(dataset)}")

# Analyze first 5 samples
for i, sample in enumerate(dataset[:5]):
    print(f"\n--- Sample {i+1} ---")
    print(f"TV ID: {sample['tv_id']}")
    print(f"Center Frame: {sample['center_frame']}")
    print(f"tv_hist shape: {sample['tv_hist'].shape}")

    # 1. Seek to the center frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample['center_frame'])
    ret, frame = cap.read()
    if not ret:
        print("  ▶ Failed to read frame", sample['center_frame'])
        continue

    # 2. Prepare the history points
    #    If your coords are floats, convert to ints
    hist_pts = sample['tv_hist'].astype(np.int32)  # shape (TH, 2)
    #    Reshape for polylines: OpenCV wants shape (num_points,1,2)
    hist_pts = hist_pts.reshape(-1, 1, 2)

    # 3. Draw the trajectory
    #    - Blue line, thickness=2
    cv2.polylines(frame, [hist_pts], isClosed=False, color=(255, 0, 0), thickness=2)

    # 5. Show the frame
    cv2.imshow(f"Sample {i+1} Trajectory Overlay", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Release after the loop
cap.release()
