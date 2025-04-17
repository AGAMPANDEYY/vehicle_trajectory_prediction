
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


import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib import cm
import os

# Paths
CONFLICT_CSV = OUTPUT_CSV  # From earlier step
VIDEO_INPUT = r"C:\Agam\Work\vehicle_trajectory_prediction\traffic-analysis-detection-tracking\data\output_video.mp4" #"data/output_SDTATT.mp4"
VIDEO_OUTPUT = "data/video_with_conflicts.mp4"

# Frame dimensions
FRAME_WIDTH = 2046  # change based on your input video
FRAME_HEIGHT = 1080

tracked_csv_path = r"C:\Agam\Work\vehicle_trajectory_prediction\traffic-analysis-detection-tracking\data\processed_combined_tracking_data.csv"
tracked_csv=pd.read_csv(tracked_csv_path)

def get_vehicles_in_frame(frame_idx, tracked_csv):
    """
    Retrieves vehicles present in a specific frame from tracking data.
    
    Args:
        frame_idx (int): The frame number to retrieve vehicles for
        tracked_csv (DataFrame): Pandas DataFrame containing vehicle tracking data
        
    Returns:
        List of tuples, each containing (vehicle_id, (x, y, w, h))
    """
    # Filter the DataFrame for the specific frame
    vehicles = tracked_csv[tracked_csv['frame_number'] == frame_idx]
    
    # Create a list of tuples in the format: (vehicle_id, (x, y, w, h))
    vehicle_data = []
    for _, vehicle in vehicles.iterrows():
        vehicle_id = int(vehicle['vehicle_id'])
        
        # Calculate bounding box parameters
        x1, y1 = float(vehicle['x1']), float(vehicle['y1'])
        x2, y2 = float(vehicle['x2']), float(vehicle['y2'])
        
        # x, y are the top-left coordinates (not center)
        x, y = x1, y1
        
        # Calculate width and height
        w = x2 - x1
        h = y2 - y1
        
        vehicle_data.append((vehicle_id, (int(x), int(y), int(w), int(h))))
    
    return vehicle_data



# Load conflict data
conflict_df = pd.read_csv(CONFLICT_CSV)

# Build a frame-wise conflict heatmap dictionary
conflict_map = defaultdict(list)
# Track which vehicle IDs are involved in conflicts at each frame
conflict_vehicle_ids = defaultdict(set)
for _, row in conflict_df.iterrows():
    frame = int(row['conflict_frame'])
    x = int(row['conflict_x'])
    y = int(row['conflict_y'])
    conflict_map[frame].append((x, y))
    # Extract vehicle IDs involved in this conflict
    # Assuming 'track_ids' is stored as a string like "[1, 4, 5]" or as a list
    if 'track_ids' in row:
        if isinstance(row['track_ids'], str):
            # Parse string representation of list
            try:
                vehicle_ids = eval(row['track_ids'])
                for vid in vehicle_ids:
                    conflict_vehicle_ids[frame].add(vid)
            except:
                # If string parsing fails, try comma-separated format
                vehicle_ids = [int(id.strip()) for id in row['track_ids'].strip('[]').split(',')]
                for vid in vehicle_ids:
                    conflict_vehicle_ids[frame].add(vid)
        elif isinstance(row['track_ids'], list):
            # If already a list
            for vid in row['track_ids']:
                conflict_vehicle_ids[frame].add(vid)

# Load video
cap = cv2.VideoCapture(VIDEO_INPUT)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

FRAME_HEIGHT= 480
FRAME_WIDTH= 852

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

# Initialize an accumulated heatmap for the entire video
accumulated_heatmap = np.zeros((480, 852), dtype=np.float32)

# Conflict highlighting colors
NORMAL_BOX_COLOR = (255, 255, 255)   # White for normal vehicles
CONFLICT_BOX_COLOR = (0, 0, 255)     # Red for vehicles in conflict
WARNING_BOX_COLOR = (0, 165, 255)    # Orange for vehicles approaching conflict

with tqdm (total=frame_count, desc="Overlaying Conflict Zone Heatmap on video") as pbar:

    # Loop through video frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, channels = frame.shape
        #print(f"Frame dimensions: Width={width}, Height={height}")

        # Create a frame-specific heatmap based on actual frame dimensions
        frame_heatmap = np.zeros((height, width), dtype=np.float32)

        if frame_idx in conflict_map:
            for (x, y) in conflict_map[frame_idx]:
                # Ensure coordinates are within frame bounds
                if 0 <= x < width and 0 <= y < height:
                    # Create a larger Gaussian blob instead of a simple circle
                    radius = 60  # Larger radius for better visibility
                    sigma = radius / 3
                    
                    # Define region for Gaussian blob
                    y_min = max(0, y - radius)
                    y_max = min(height, y + radius + 1)
                    x_min = max(0, x - radius)
                    x_max = min(width, x + radius + 1)
                    
                    # Create y, x meshgrid for the region
                    y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
                    
                    # Calculate distance from center
                    dist_sq = ((y_coords - y) ** 2 + (x_coords - x) ** 2)
                    
                    # Create Gaussian mask
                    mask = np.exp(-dist_sq / (2 * sigma ** 2))
                    
                    # Add to frame heatmap
                    frame_heatmap[y_min:y_max, x_min:x_max] += mask
        
        # Add frame heatmap to accumulated heatmap with decay factor
        decay_factor = 0.95  # Reduces previous heatmap values
        accumulated_heatmap = accumulated_heatmap * decay_factor + frame_heatmap
        
        # Normalize and convert to color heatmap for visualization
        if np.max(accumulated_heatmap) > 0:
            # Copy to avoid modifying the accumulated heatmap
            vis_heatmap = accumulated_heatmap.copy()
            
            # Apply threshold to highlight more significant conflicts
            threshold = np.max(vis_heatmap) * 0.1  # 10% of max value
            vis_heatmap[vis_heatmap < threshold] = 0
            
            # Normalize to 0-255 range
            cv2.normalize(vis_heatmap, vis_heatmap, 0, 255, cv2.NORM_MINMAX)
            vis_heatmap = vis_heatmap.astype(np.uint8)
            
            # Apply colormap - JET provides good blue-to-red gradient
            heatmap_color = cv2.applyColorMap(vis_heatmap, cv2.COLORMAP_JET)
            
            # Ensure heatmap matches frame dimensions
            heatmap_color = cv2.resize(heatmap_color, (width, height))
            
            # Blend with original frame
            overlayed = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)
        else:
            overlayed = frame
        
        # NEW CODE: Highlight vehicles involved in conflicts
        # This assumes you have vehicle detection/tracking data for each frame
        # Assuming you have a function or data structure that gives you vehicle positions and IDs
        # For example: vehicles = get_vehicles_in_frame(frame_idx)
        
        # Example implementation - replace with your actual vehicle tracking code
        # For demonstration, I'll use placeholder code that assumes you have positions
        vehicles_in_frame = get_vehicles_in_frame(frame_idx,tracked_csv)  # This is a placeholder function
        
        # Check each vehicle to see if it's involved in a conflict
        for vehicle_id, (x, y, w, h) in vehicles_in_frame:
            box_color = NORMAL_BOX_COLOR  # Default white box
            label_text = f"ID:{vehicle_id}"
            
            # Check if this vehicle is involved in a conflict at this frame
            if frame_idx in conflict_vehicle_ids and vehicle_id in conflict_vehicle_ids[frame_idx]:
                box_color = CONFLICT_BOX_COLOR  # Red box for conflict vehicles
                label_text = f"ID:{vehicle_id} [CONFLICT]"
            
            # Draw the bounding box with appropriate color
            cv2.rectangle(overlayed, (x, y), (x + w, y + h), box_color, 2)
            
            # Add vehicle ID label
            cv2.putText(overlayed, label_text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        out.write(overlayed)
        frame_idx += 1
        pbar.update(1)
cap.release()
out.release()
print(f"Video saved with conflict heatmap: {VIDEO_OUTPUT}")
