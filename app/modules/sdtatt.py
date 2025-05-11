import torch
from torch.utils.data import DataLoader
from traffic_analysis_detection_tracking.SDT_ATT.model import SDTATTModel
from traffic_analysis_detection_tracking.SDT_ATT.dataloader import SDTATTDataset 
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
CHECKPOINT_PATH=os.path.join(BASE_DIR, "sdtatt_final.pt")
CHECKPOINT_PATH=r"C:\Agam\Work\vehicle_trajectory_prediction\app\sdtatt_final.pt"


# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SDTATTModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_neighbors=NUM_NEIGHBORS,
        future_len=FUTURE_LEN
    ).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH,  map_location=torch.device(device)))
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

def SDTATT_predict_vehicle(frame_id, track_id):
    print(f"Predicting for frame_id: {frame_id}, track_id: {track_id}")
    sample = dataset.get_sample_by_frame_and_track(frame_id, track_id)

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


def SDTATT_predict_all():
    """
    Predict future trajectories for all samples in the dataset.
    Returns path to saved CSV of predictions.
    """
    all_preds = []

    tracking_dict = {(row['frame_number'], row['tracker_id']): row for _, row in tracking_df.iterrows()}

    for data_sample in tqdm(dataset, desc="Predicting all trajectories"):
        center_frame = int(data_sample['center_frame'])
        track_id = int(data_sample['tv_id'])

        # Prepare inputs
        tv_hist = torch.from_numpy(data_sample['tv_hist'].numpy()).float().unsqueeze(0).to(device)
        nv_sp = torch.from_numpy(data_sample['nv_sp'].numpy()).float().unsqueeze(0).to(device)
        nv_dp = torch.from_numpy(data_sample['nv_dp'].numpy()).float().unsqueeze(0).to(device)

        # Model inference
        with torch.no_grad():
            output = model(tv_hist, nv_sp, nv_dp)

        pred_rel = output[0, :, :2]
        pred_abs = torch.cumsum(pred_rel, dim=0) + tv_hist[0, -1]
        pred_np = pred_abs.cpu().numpy()

        ref_bb = tracking_dict.get((center_frame, track_id), None)
        timestamp= tracking_dict.get((center_frame, track_id), None)['timestamp']
        if ref_bb is None:
            print(f"Skipping track_id={track_id}, no bbox info at frame={center_frame}")
            continue

        # Estimate width and height of bbox
        width = abs(ref_bb['x2'] - ref_bb['x1'])
        height = abs(ref_bb['y2'] - ref_bb['y1'])

        # Collect predictions
        for i in range(pred_np.shape[0]):
            future_frame = center_frame + i + 1

            x_center = float(pred_np[i, 0])
            y_center = float(pred_np[i, 1])

            # Compute bounding box around predicted center
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            all_preds.append({
                'timestamp': timestamp,
                'frame_id': future_frame,
                'vehicle_id': track_id,
                'x_future': float(pred_np[i, 0]),
                'y_future': float(pred_np[i, 1]),
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2
            })

    # DataFrame and dedupe
    pred_df = pd.DataFrame(all_preds)
    pred_df = pred_df.drop_duplicates(subset=['frame_id', 'vehicle_id', 'x_future', 'y_future'])

    # Save
    out_csv = os.path.join(PARENT_DIR, "data", "future_trajectories.csv")
    out_csv= "/kaggle/working/vehicle_trajectory_prediction/traffic-analysis-detection-tracking/data/future_trajectories.csv"
    pred_df.to_csv(out_csv, index=False)
    print(f"Saved {len(pred_df)} predictions to {out_csv}")