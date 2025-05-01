#python traffic-analysis-detection-tracking/SDT-ATT/eval.py

####3###

import torch
from torch.utils.data import DataLoader
from model import SDTATTModel
from dataloader import SDTATTDataset 
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm 
import os 
import pandas as pd
import numpy as np

# ---- Config ----
BATCH_SIZE = 1
INPUT_DIM = 2
HIDDEN_DIM = 64
NUM_NEIGHBORS = 3
FUTURE_LEN = 90

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
N_DATA_PATH = os.path.join(BASE_DIR, "data", "sdtatt_data.npy")
N_DATA_PATH="/kaggle/input/sdtatt-np/sdtatt_data.npy"
TRACKING_CSV_PATH = os.path.join(PARENT_DIR, "data", "combined_tracking_data.csv")
TRACKING_CSV_PATH= "/kaggle/input/combined-tracking-data/combined_tracking_data.csv"
CHECKPOINT_PATH=os.path.join(BASE_DIR, "checkpoint","sdtatt_final.pt")
CHECKPOINT_PATH="/kaggle/input/sdtatt_np/pytorch/default/1/sdtatt_final.pt"

# Load tracking data
tracking_df = pd.read_csv(TRACKING_CSV_PATH)

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Dataset ----
dataset = SDTATTDataset(N_DATA_PATH)


# Set target frame and vehicle
frame_id = 320
track_id = 8


print(f"Predicting for frame_id: {frame_id}, track_id: {track_id}")
sample = dataset.get_sample_by_frame_and_track(frame_id, track_id)

def validate_data(tv_hist, nv_sp, nv_dp):
    # Check for NaN or Inf values
    if np.any(np.isnan(tv_hist)) or np.any(np.isnan(nv_sp)) or np.any(np.isnan(nv_dp)):
        raise ValueError("NaN values found in input data")
    
    if np.any(np.isinf(tv_hist)) or np.any(np.isinf(nv_sp)) or np.any(np.isinf(nv_dp)):
        raise ValueError("Infinite values found in input data")
    
    # Check for zero-padded neighbors
    valid_neighbors = np.sum(np.any(nv_sp != 0, axis=(1,2)))
    if valid_neighbors < 2:  # Require at least 2 valid neighbors
        raise ValueError("Insufficient valid neighbors for prediction")

def SDTATT_predict():
    model = SDTATTModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_neighbors=NUM_NEIGHBORS,
        future_len=FUTURE_LEN
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()
    tv_hist = sample['tv_hist'].numpy()
    nv_sp = sample['nv_sp'].numpy()
    nv_dp = sample['nv_dp'].numpy()

    # Convert back to tensors
    tv_hist = torch.from_numpy(tv_hist).float()
    nv_sp = torch.from_numpy(nv_sp).float()
    nv_dp = torch.from_numpy(nv_dp).float()
    
    # Move to device
    tv_hist = tv_hist.unsqueeze(0).to(device)
    nv_sp = nv_sp.unsqueeze(0).to(device)
    nv_dp = nv_dp.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tv_hist, nv_sp, nv_dp)
    
    # Denormalize the output
    pred_rel = output[0, :, :2]   # μₓ,μᵧ
    #pred_abs=torch.cumsum(pred_rel*tv_std+tv_mean,dim=0)+tv_hist[0,-1]
    pred_abs = torch.cumsum(pred_rel, dim=0) + tv_hist[0, -1]
    pred_abs= pred_abs.cpu()
    
    return pred_abs, sample['center_frame'], sample['tv_id']

pred_abs, center_frame, track_id = SDTATT_predict()

# Load video
VIDEO_PATH = os.path.join(PARENT_DIR, "data", "Lane_C_Video.mp4")
VIDEO_PATH="/kaggle/input/lne-c-video/Lane_C_Video.mp4"
video = cv2.VideoCapture(VIDEO_PATH)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Output video writer
out = cv2.VideoWriter(
    #os.path.join(PARENT_DIR, "data", "output_SDTATT2.mp4"),
    "/kaggle/working/vehicle_trajectory_prediction/sdtatt_video",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

START_BUFFER = 150
END_BUFFER = 300
start_frame = int(center_frame) + 1
end_frame = start_frame + FUTURE_LEN
frame_idx = 0

with tqdm(total=START_BUFFER+FUTURE_LEN+END_BUFFER, desc="Processing Frames") as pbar:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        if start_frame - START_BUFFER <= frame_idx < end_frame + END_BUFFER:
            # Get tracking data for current frame
            current_frame_data = tracking_df[tracking_df['frame_number'] == frame_idx]
            
            # Draw bounding boxes and labels
            for _, row in current_frame_data.iterrows():
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                det_id = int(row['tracker_id'])
                
                # Label Target Vehicle
                if det_id == track_id:
                    label = "TV"
                    color = (0, 255, 0)  # Green
                # Label Neighbor Vehicles
                elif det_id in sample.get("nv_ids", []):
                    label = "NV"
                    color = (0, 165, 255)  # Orange
                else:
                    label = f"ID:{det_id}"
                    color = (255, 255, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw prediction
            idx = frame_idx - start_frame
            if 0 <= idx < len(pred_abs):
                x, y = pred_abs[int(idx)].numpy().astype(int)
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

                for i in range(1, idx + 1):
                    pt1 = tuple(pred_abs[i - 1].numpy().astype(int))
                    pt2 = tuple(pred_abs[i].numpy().astype(int))
                    cv2.line(frame, pt1, pt2, color=(0, 200, 0), thickness=2)

            # Draw TV history up to current frame
            tv_hist_np = sample['tv_hist'].cpu().numpy().astype(int)
            # Only show history points after start_frame
            if frame_idx >= start_frame:
                # Calculate how many history points to show based on current frame
                history_points_to_show = frame_idx - start_frame + 1
                if history_points_to_show > 0 and history_points_to_show <= len(tv_hist_np):
                    for pt in tv_hist_np[:history_points_to_show]:
                        cv2.circle(frame, tuple(pt), 2, (255, 0, 0), -1)

            # Draw NV history
            nv_hist_np = sample['nv_sp'].cpu().numpy().astype(int)
            for n in range(nv_hist_np.shape[0]):
                for pt in nv_hist_np[n]:
                    cv2.circle(frame, tuple(pt), 2, (0, 140, 255), -1)

            # Add prediction info
            if idx >= 0:
                cv2.putText(frame, "SDT-ATT Prediction", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
                cv2.putText(frame, f"Predicted Maneuver: Lane Keep", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

            out.write(frame)
            pbar.update(1)

        elif frame_idx >= end_frame + END_BUFFER:
            break

        frame_idx += 1

video.release()
out.release()
print("Video with predictions saved!")

