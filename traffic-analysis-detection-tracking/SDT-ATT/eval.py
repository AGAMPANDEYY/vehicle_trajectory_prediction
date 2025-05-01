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
from scipy.spatial.distance import euclidean

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


model = SDTATTModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_neighbors=NUM_NEIGHBORS,
        future_len=FUTURE_LEN
    ).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

def get_predicted_maneuver(predicted_traj):
    return "Lane Change"


def calculate_trajectory_metrics(pred_trajectory, actual_trajectory):
    """
    Calculate trajectory prediction metrics
    Args:
        pred_trajectory: predicted trajectory points
        actual_trajectory: ground truth trajectory points
    Returns:
        ade: Average Displacement Error
        fde: Final Displacement Error
    """
    # Ensure same length of trajectories
    min_len = min(len(pred_trajectory), len(actual_trajectory))
    pred = pred_trajectory[:min_len]
    actual = actual_trajectory[:min_len]
    
    # Calculate displacement errors for each timestep
    displacement_errors = [euclidean(p, a) for p, a in zip(pred, actual)]
    
    # Average Displacement Error (ADE)
    ade = np.mean(displacement_errors)
    
    # Final Displacement Error (FDE)
    fde = displacement_errors[-1]
    
    return ade, fde


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


all_preds=[]

def SDTATT_predict_vehicle():
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


def SDTATT_predict():
  
    for sample in tqdm(dataset, desc="Predicting all trajectories"):
        # Extract identifiers
        center_frame = sample['center_frame']
        track_id     = sample['tv_id']
        
        # Move data onto device
        tv_hist = torch.from_numpy(sample['tv_hist']).float().unsqueeze(0).to(device)
        nv_sp   = torch.from_numpy(sample['nv_sp']).  float().unsqueeze(0).to(device)
        nv_dp   = torch.from_numpy(sample['nv_dp']).  float().unsqueeze(0).to(device)

        # Run model
        with torch.no_grad():
            output = model(tv_hist, nv_sp, nv_dp)
        # Build absolute trajectory ([FUTURE_LEN,2])
        pred_rel = output[0,:,:2]
        pred_abs = torch.cumsum(pred_rel, dim=0) + tv_hist[0,-1]
        pred_np  = pred_abs.cpu().numpy()  # shape (FUTURE_LEN, 2)

        # For each future timestep
        for i in range(pred_np.shape[0]):
            future_frame = int(center_frame) + 1 + i
            # Try to get the bounding box for this vehicle/frame
            bb = tracking_df[
                (tracking_df.frame_number == future_frame) &
                (tracking_df.tracker_id   == track_id)
            ]
            if bb.empty:
                # skip or fill with NaNs
                continue
            x1, y1, x2, y2 = bb[['x1','y1','x2','y2']].iloc[0]

            all_preds.append({
                'frame_id':   future_frame,
                'vehicle_id': track_id,
                'x_future':   float(pred_np[i,0]),
                'y_future':   float(pred_np[i,1]),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })
    
    # Convert to DataFrame & save
    pred_df = pd.DataFrame(all_preds)
    pred_df.to_csv(os.path.join(PARENT_DIR, "data", "future_trajectories.csv"),
                index=False)
    print(f"Saved {len(pred_df)} trajectory points to future_trajectories.csv")


choice="all trajectories"

if choice=="single vehicle":
    pred_abs, center_frame, track_id = SDTATT_predict_vehicle()
    
else:
    # Predict all trajectories
    SDTATT_predict()
    # Load the predicted trajectories
    pred_df = pd.read_csv(os.path.join(PARENT_DIR, "data", "future_trajectories.csv"))
    # Filter for the specific vehicle and frame
    pred_df = pred_df[(pred_df['frame_id'] == frame_id) & (pred_df['vehicle_id'] == track_id)]
    # Extract the predicted trajectory points
    pred_abs = torch.tensor(pred_df[['x_future', 'y_future']].values).float()
    center_frame = frame_id
    track_id = track_id

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
    "/kaggle/working/vehicle_trajectory_prediction/sdtatt_video/output_SDTATT.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

START_BUFFER = 150
END_BUFFER = 300
start_frame = int(center_frame) + 1
end_frame = start_frame + FUTURE_LEN
frame_idx = 0

# Initialize error collection variables
all_errors = []
frame_errors = {}


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

           
            # Get actual trajectory data for comparison
            actual_trajectory = []
            evaluation_frames = range(start_frame, start_frame + FUTURE_LEN)
            for eval_f  in evaluation_frames:
                frame_data = tracking_df[(tracking_df['frame_number'] == eval_f ) & 
                                       (tracking_df['tracker_id'] == track_id)]
                if not frame_data.empty:
                    x_center = (frame_data['x1'].iloc[0] + frame_data['x2'].iloc[0]) / 2
                    y_center = (frame_data['y1'].iloc[0] + frame_data['y2'].iloc[0]) / 2
                    actual_trajectory.append([x_center, y_center])

                        # Convert predictions to numpy array
            pred_trajectory = pred_abs.cpu().numpy()
            
            # Getting predicted maneuver
            predicted_maneuver= get_predicted_maneuver(pred_trajectory)
            # Add prediction info
            if idx >= 0:
                cv2.putText(frame, "SDT-ATT Prediction", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
                cv2.putText(frame, f"Predicted Maneuver: {predicted_maneuver}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)


            if len(actual_trajectory) > 0:
                # Calculate metrics
                ade, fde = calculate_trajectory_metrics(pred_trajectory, actual_trajectory)
                #print(f"\nTrajectory Prediction Metrics:")
                #print(f"Average Displacement Error (ADE): {ade:.2f} pixels")
                #print(f"Final Displacement Error (FDE): {fde:.2f} pixels")

            # After drawing predictions, add this code to draw actual trajectory
            if frame_idx >= start_frame and frame_idx < end_frame:
                current_actual_idx = frame_idx - start_frame
                if current_actual_idx < len(actual_trajectory):
                    # Draw actual trajectory points
                    for i in range(current_actual_idx + 1):
                        pt = tuple(map(int, actual_trajectory[i]))
                        cv2.circle(frame, pt, 2, (0, 0, 255), -1)  # Red color for actual trajectory
                    
                    # Draw error between prediction and actual position
                    if current_actual_idx < len(pred_trajectory):
                        pred_pt = tuple(map(int, pred_trajectory[current_actual_idx]))
                        actual_pt = tuple(map(int, actual_trajectory[current_actual_idx]))
                        cv2.line(frame, pred_pt, actual_pt, (255, 0, 255), 1)  # Magenta line showing error
                        
                        # Calculate and store current error
                        current_error = euclidean(pred_pt, actual_pt)
                        frame_errors[frame_idx] = current_error
                        all_errors.append(current_error)
                        
                        # Display current error on frame
                        cv2.putText(frame, f"Current Error: {current_error:.2f}px", 
                                  (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

            # Add ADE/FDE to video if available
            if 'ade' in locals():
                cv2.putText(frame, f"ADE: {ade:.2f}px", (30, 160), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                cv2.putText(frame, f"FDE: {fde:.2f}px", (30, 200), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
           

            out.write(frame)
            pbar.update(1)

        elif frame_idx >= end_frame + END_BUFFER:
            break

        frame_idx += 1

video.release()
out.release()

# Print final error statistics
print("\n=== Trajectory Prediction Error Statistics ===")
print(f"Average Displacement Error (ADE): {np.mean(all_errors):.2f} pixels")
print(f"Final Displacement Error (FDE): {all_errors[-1]:.2f} pixels")
print(f"Maximum Error: {np.max(all_errors):.2f} pixels")
print(f"Minimum Error: {np.min(all_errors):.2f} pixels")
print(f"Standard Deviation of Error: {np.std(all_errors):.2f} pixels")

# Save error data to CSV
error_df = pd.DataFrame({
    'frame': list(frame_errors.keys()),
    'error': list(frame_errors.values())
})
error_csv_path = os.path.join(PARENT_DIR, "data", "trajectory_errors.csv")
error_df.to_csv(error_csv_path, index=False)
print(f"\nError data saved to: {error_csv_path}")


print("Video with predictions saved!")

