#python traffic-analysis-detection-tracking/SDT-ATT/eval.py

import torch
from torch.utils.data import DataLoader
from model import SDTATTModel
from dataloader import SDTATTDataset 
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm 

# ---- Config ----
BATCH_SIZE = 1
INPUT_DIM = 2
HIDDEN_DIM = 64
NUM_NEIGHBORS = 5  # Change according to your dataset Curreent numpy from datasets.py has 5 neighbors
FUTURE_LEN = 30
N_DATA_PATH = r"C:\Agam\Work\vehicle_trajectory_prediction\traffic-analysis-detection-tracking\SDT-ATT\data\sdtatt_data.npy"  # update if needed

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Dataset ----
dataset = SDTATTDataset(N_DATA_PATH)
sample = dataset[0]

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

# Convert relative to absolute
pred_abs = torch.cumsum(pred_rel, dim=0) + last_pos  # [30, 2]
print("Absolute Predicted Future Trajectory:")
print(pred_abs.cpu().numpy())  # [30, 2]    



import cv2

# Load the video
VIDEO_PATH = r"C:\Agam\Work\vehicle_trajectory_prediction\traffic-analysis-detection-tracking\data\0212.mp4"


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
if not ret:
    print("Error: Could not read the first frame.")
    exit()

for (x, y) in pred_abs.numpy().astype(int):
    frame = cv2.circle(frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

cv2.imshow("Prediction on First Frame", frame)
cv2.waitKey(1)
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