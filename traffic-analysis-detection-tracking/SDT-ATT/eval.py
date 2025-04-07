#python traffic-analysis-detection-tracking/SDT-ATT/eval.py

import torch
from torch.utils.data import DataLoader
from model import SDTATTModel
from dataloader import SDTATTDataset 
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm 
import os 

# ---- Config ----
BATCH_SIZE = 1
INPUT_DIM = 2
HIDDEN_DIM = 64
NUM_NEIGHBORS = 5  # Change according to your dataset Curreent numpy from datasets.py has 5 neighbors
FUTURE_LEN = 30

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N_DATA_PATH = os.path.join(BASE_DIR, "data", "sdtatt_data.npy")  # update if needed

NUM_SAMPLES_TO_PREDICT=5

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Dataset ----
dataset = SDTATTDataset(N_DATA_PATH)

print("Dataset details:")
print(f"Number of samples: {len(dataset)}")
print(f"Sample keys: {dataset[0].keys()}")  # Check keys in the first sample
print(f"Sample shape: {dataset[0]['tv_hist'].shape}, {dataset[0]['nv_sp'].shape}, {dataset[0]['nv_dp'].shape}")

frame_id = 2345 # Set to None if you want to search by track_id only
track_id = None  # Set to None if you want to search by frame_id only
sample = dataset.get_sample_by_frame_and_track(frame_id, track_id)
# ---- Prepare Model ----
model = SDTATTModel(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_neighbors=NUM_NEIGHBORS,
    future_len=FUTURE_LEN
).to(device)

model.eval()

# ---- Move data to device ----
tv_hist = sample['tv_hist'].unsqueeze(0).to(device)  # shape: [1, T, 2]
nv_sp = sample['nv_sp'].unsqueeze(0).to(device)      # shape: [1, N, T, 2]
nv_dp = sample['nv_dp'].unsqueeze(0).to(device)      # shape: [1, N, T, 2]

# ---- Inference ----
with torch.no_grad():
    output = model(tv_hist, nv_sp, nv_dp)  # shape: [1, future_len, 2]

# Assume you have access to the last known position (e.g., from tv_hist)
last_pos = tv_hist[0, -1]  # Shape: [2], last frame of history
pred_rel = output[0]       # Shape: [30, 2] from model

#
print("Past vehcile Trajectory:")
print(tv_hist.cpu().numpy())  # [T, 2]

# Convert relative to absolute
pred_abs = torch.cumsum(pred_rel, dim=0) + last_pos  # [30, 2]
print("Absolute Predicted Future Trajectory:")
print(pred_abs.cpu().numpy())  # [30, 2]    


import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
VIDEO_PATH = os.path.join(PARENT_DIR, "data", "0212.mp4")  # update if needed

# ---- Load Video and Get Properties ----
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video resolution: {width}x{height}, FPS: {fps}, Total frames: {frame_count}")
cap.release()

# ---- Draw prediction on the first frame ----
video = cv2.VideoCapture(VIDEO_PATH)
ret, frame = video.read()

print("Image shape:", frame.shape)  # Should be (H, W, 3)
print("Prediction coordinates (first 5):", pred_abs.numpy()[:5])

if not ret:
    print("Error: Could not read the first frame.")
    exit()

for (x, y) in pred_abs.numpy().astype(int):
    if 0 <= x < width and 0 <= y < height:
        cv2.circle(frame, (x, y), radius=2, color=(0, 0, 255), thickness=-1)


display_frame = cv2.resize(frame, (1024, 540)) 
cv2.imshow(f"Prediction on Sample Frame No", display_frame)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()
video.release()

# ---- Write Output Video with Prediction Overlay ----
video = cv2.VideoCapture(VIDEO_PATH)
out = cv2.VideoWriter(r"C:\Agam\Work\vehicle_trajectory_prediction\traffic-analysis-detection-tracking\SDT-ATT\data\output_SDTATT.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

start_frame = 1
frame_idx = 0

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print("Starting video overlay and tqdm...")


with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if start_frame <= frame_idx < start_frame + FUTURE_LEN:
            idx = frame_idx - start_frame
            x, y = pred_abs[idx].numpy().astype(int)
            frame = cv2.circle(frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

video.release()
out.release()
print("Video saved as output_SDTATT.mp4")