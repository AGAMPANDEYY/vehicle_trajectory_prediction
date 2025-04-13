
"""
we assume that each vehicle sample (from your dataset) has a predicted future trajectory (using your SDTATT model) for, say, 10 future frames. We then “simulate” each vehicle’s future position and—using a simple default boundary (for example, a rectangle or box around the vehicle)—check for overlaps between different vehicles at the same global (predicted) frame. When two or more vehicle boundaries overlap, we flag that frame as a potential conflict. Finally, we store in a CSV the conflict information with fields such as:

conflict_frame (or conflict timestamp if you convert frame to time)

conflict_x, conflict_y (the average conflict location)

track_ids (list of vehicle IDs in conflict)

In our sample code we use a fixed vehicle boundary (width50, height20 pixels) to compute the overlap; you can adjust this (or compute per vehicle if you have that data).
"""


import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from model import SDTATTModel
from dataloader import SDTATTDataset
from tqdm import tqdm
import os



import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# =====================================
# CONFIGURATION AND PATHS
# =====================================
INPUT_DIM = 2
HIDDEN_DIM = 64
NUM_NEIGHBORS = 5
FUTURE_LEN = 10  # Number of future frames predicted by SDTATT model
N_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sdtatt_data.npy")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "conflict_events.csv")

# For simplicity, define default vehicle dimensions (in pixels)
VEHICLE_WIDTH = 50   # width of bounding box (assumed constant)
VEHICLE_HEIGHT = 20  # height of bounding box

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# CUSTOM FUNCTIONS
# =====================================

def detect_lane_change(y_coords, threshold=1.0):
    """
    (Optional) Lane change detection logic.
    For your case (if y-axis is lateral), if cumulative abs change in y > threshold, return True.
    """
    dy = np.abs(np.diff(y_coords))
    total_change = np.sum(dy)
    return total_change > threshold

def detect_lane_change_rate_based(y_coords, rate_threshold=0.05, duration=3):
    """
    y_coords: array of y over time
    rate_threshold: minimum change between frames to consider lane change
    duration: how many consecutive frames to observe significant change
    """
    dy = np.diff(y_coords)
    lane_change_mask = np.abs(dy) > rate_threshold

    # Check for at least `duration` consecutive significant lateral movements
    count = 0
    for change in lane_change_mask:
        if change:
            count += 1
            if count >= duration:
                return True
        else:
            count = 0
    return False

def assign_zone(x, y, frame_width, frame_height, num_cols=10, num_rows=5):
    """
    Divides the frame into a grid (default 10 columns x 5 rows) and returns zone indices as (row, col).
    """
    cell_width = frame_width / num_cols
    cell_height = frame_height / num_rows
    col = min(int(x // cell_width), num_cols - 1)
    row = min(int(y // cell_height), num_rows - 1)
    return row, col

def boxes_overlap(pos1, pos2, width=VEHICLE_WIDTH, height=VEHICLE_HEIGHT):
    """
    Checks whether two vehicles (with identical bounding box dimensions) overlap.
    The boundaries are computed as center ± half width/height.
    Two boxes overlap if:
         |x1 - x2| < (width)   and   |y1 - y2| < (height)
    """
    return (abs(pos1[0] - pos2[0]) < width) and (abs(pos1[1] - pos2[1]) < height)

# =====================================
# SDTATT PREDICTION FUNCTION
# =====================================
def SDTATT_predict(sample, model, device):
    """
    Runs forward inference on one sample and returns the predicted future trajectory
    as absolute positions (shape [FUTURE_LEN, 2]).
    """
    tv_hist = sample['tv_hist'].unsqueeze(0).to(device)  # [1, T, 2]
    nv_sp = sample['nv_sp'].unsqueeze(0).to(device)        # [1, N, T, 2]
    nv_dp = sample['nv_dp'].unsqueeze(0).to(device)        # [1, N, T, 2]
    
    with torch.no_grad():
        output = model(tv_hist, nv_sp, nv_dp)  # [1, FUTURE_LEN, 2]
    last_pos = tv_hist[0, -1]  # Use the last history frame as reference
    pred_rel = output[0]       # [FUTURE_LEN, 2]
    pred_abs = torch.cumsum(pred_rel, dim=0) + last_pos  # [FUTURE_LEN, 2]
    return pred_abs.cpu().numpy()

# =====================================
# LOAD DATASET AND MODEL
# =====================================
from dataloader import SDTATTDataset 
from model import SDTATTModel        

dataset = SDTATTDataset(N_DATA_PATH)
print("Total samples in dataset:", len(dataset))

model = SDTATTModel(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_neighbors=NUM_NEIGHBORS,
    future_len=FUTURE_LEN
).to(DEVICE)
model.eval()

# =====================================
# COLLECT PREDICTIONS ACROSS SAMPLES
# =====================================
# We'll iterate over all (or a subset of) samples, run prediction, and store:
# - center_frame
# - tv_id
# - predicted future trajectory (for each frame offset)
# We'll assume that the "center_frame" of the sample is the reference; predictions are made for FUTURE_LEN frames ahead.

predictions = []  # list of dicts with keys: center_frame, track_id, pred (array [FUTURE_LEN,2])
for i in tqdm(range(len(dataset)), desc="Predicting trajectories"):
    sample = dataset[i]
    predictions.append({
        'center_frame': sample['center_frame'],
        'track_id': sample['tv_id'],
        'pred': SDTATT_predict(sample, model, DEVICE)
    })

# =====================================
# GROUP PREDICTIONS BY GLOBAL (Predicted) Frame
# =====================================
# For each sample, for each predicted future offset (0 to FUTURE_LEN-1),
# compute the global frame (e.g., sample['center_frame'] + offset) and store position.
time_dict = {}  # key: global frame number, value: list of tuples (track_id, (x, y))
for pred in predictions:
    base_frame = pred['center_frame']
    track = pred['track_id']
    for offset in range(FUTURE_LEN):
        global_frame = base_frame + offset
        position = pred['pred'][offset]  # (x, y)
        time_dict.setdefault(global_frame, []).append((track, position))

# =====================================
# CONFLICT ZONE DETECTION
# =====================================
# For each global frame, check for overlapping boundaries among vehicles.
# When two or more vehicle boundaries overlap, that is considered a conflict.
conflict_events = []  # list to store conflict events across frames

for frame, vehicle_list in time_dict.items():
    if len(vehicle_list) < 2:
        continue  # need at least 2 vehicles to have a conflict
    # We do pairwise checks to determine groups of vehicles that overlap.
    conflict_groups = []  # each group is a set of track IDs that are in conflict at this frame
    n = len(vehicle_list)
    for i in range(n):
        track_i, pos_i = vehicle_list[i]
        group = {track_i}
        for j in range(i + 1, n):
            track_j, pos_j = vehicle_list[j]
            if boxes_overlap(pos_i, pos_j, width=VEHICLE_WIDTH, height=VEHICLE_HEIGHT):
                group.add(track_j)
        if len(group) > 1:
            conflict_groups.append(group)
    # Optionally, merge overlapping groups. For simplicity, we record each group separately.
    for group in conflict_groups:
        # Compute an average conflict position (centroid) for the group at this frame.
        positions = [pos for (track, pos) in vehicle_list if track in group]
        avg_x = np.mean([p[0] for p in positions])
        avg_y = np.mean([p[1] for p in positions])
        conflict_events.append({
            'conflict_frame': frame,
            'conflict_x': avg_x,
            'conflict_y': avg_y,
            'track_ids': list(group)
        })

# =====================================
# SAVE CONFLICT EVENTS TO CSV
# =====================================
df_conflict = pd.DataFrame(conflict_events)
df_conflict.to_csv(OUTPUT_CSV, index=False)
print("Conflict events CSV saved at:", OUTPUT_CSV)