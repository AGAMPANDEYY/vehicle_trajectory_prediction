
#####1####


import pandas as pd
import numpy as np 
from collections import defaultdict
from tqdm import tqdm 


#parameters 
TH= 90 #time horizon ----> number of frames for history  video is of 30fps so 1 sec has 30 frames.
NF=120 #number of frames for future
MAX_NEIGHBORS=3 #max number of neighbors to consider for each TV in 2-lane double direction scenario

#load data
def load_data(csv_path):
    df=pd.read_csv(csv_path)
    # Generate frame_id by ranking unique timestamps
    #df = df.sort_values(by=['vehicle_id'])
    print(df.head())
    print("Total number of frames in the dataset: ", df['frame_number'].nunique())
    #print(f" vehciles in frame 600", df[df['frame_number'] == 600]['vehicle_id'].unique())
    return df

def extract_velocity(positions, timestamps):
    velocities = []
    for i in range(1, len(positions)):
        dt = timestamps[i] - timestamps[i-1]
        if dt == 0:
            velocities.append([0, 0])
        else:
            dx = (positions[i][0] - positions[i-1][0]) / dt
            dy = (positions[i][1] - positions[i-1][1]) / dt
            velocities.append([dx, dy])
    velocities.insert(0, velocities[0])  # repeat first velocity
    return np.array(velocities)

def get_window_sequences(df):
    all_samples = []

    grouped = df.groupby('tracker_id')

    # Build a dict: {track_id: list of (frame_id, x, y, timestamp)}
    track_data = {
        tid: group[['frame_number','x', 'y', 'timestamp']].values.tolist()
        for tid, group in grouped
    }

    for tid, traj in tqdm(track_data.items()):
        traj = np.array(traj)
        for i in range(TH, len(traj)):
            hist_window = traj[i - TH:i]
            center_frame = hist_window[-1][0]

            tv_positions = hist_window[:, 1:3]
            tv_timestamps = hist_window[:, 3]
            tv_velocities = extract_velocity(tv_positions, tv_timestamps)

            # Get NVs in the same center frame
            nv_frame_data = df[df['frame_number'] == center_frame]
            nv_frame_data = nv_frame_data[nv_frame_data['tracker_id'] != tid]

            # Calculate relative positions and distances
            tv_current_pos = tv_positions[-1]  # Current position of target vehicle
            nv_positions = nv_frame_data[['x', 'y']].values
            
            # Calculate relative positions
            rel_positions = nv_positions - tv_current_pos
            
            # Calculate distances
            dists = np.linalg.norm(rel_positions, axis=1)
            
            # Create a DataFrame with distances and IDs
            dist_df = pd.DataFrame({
                'tracker_id': nv_frame_data['tracker_id'].values,
                'distance': dists,
                'relative_x': rel_positions[:, 0],
                'relative_y': rel_positions[:, 1]
            })
            
            # Group by tracker_id and take the minimum distance for each vehicle
            dist_df = dist_df.groupby('tracker_id').min().reset_index()
            
            # Sort by distance and select top MAX_NEIGHBORS unique vehicles
            selected_nv_ids = dist_df.sort_values('distance')['tracker_id'].values[:MAX_NEIGHBORS]
            
            # Extract NV trajectories
            valid_nv_sp = []
            valid_nv_dp = []

            for nv_id in selected_nv_ids:
                nv_traj = track_data.get(nv_id, [])
                nv_traj = np.array(nv_traj)
                nv_hist = nv_traj[nv_traj[:, 0] <= center_frame]

                if len(nv_hist) >= TH:
                    nv_hist = nv_hist[-TH:]
                    nv_pos = nv_hist[:, 1:3]
                    nv_ts = nv_hist[:, 3]

                    nv_vel = extract_velocity(nv_pos, nv_ts)
                    rel_pos = nv_pos - tv_positions  # spatial
                    valid_nv_sp.append(rel_pos)
                    valid_nv_dp.append(nv_vel)

            # Only keep samples where ≥ 2 valid neighbors are found
            if len(valid_nv_sp) >= 2:
                # Pad if less than MAX_NEIGHBORS
                while len(valid_nv_sp) < MAX_NEIGHBORS:
                    valid_nv_sp.append(np.zeros((TH, 2)))
                    valid_nv_dp.append(np.zeros((TH, 2)))

                sample = {
                    'tv_id': tid,
                    'nv_ids': selected_nv_ids,
                    'tv_hist': tv_positions,
                    'tv_vel': tv_velocities,
                    'nv_sp': np.array(valid_nv_sp),
                    'nv_dp': np.array(valid_nv_dp),
                    'center_frame': center_frame
                }

                all_samples.append(sample)

    return all_samples

if __name__ == "__main__": 

    import os 
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)
    csv_path= os.path.join(PARENT_DIR,"data", "combined_tracking_data.csv")
    df = load_data(csv_path)
    dataset = get_window_sequences(df)

    # Save as numpy
    np.save(os.path.join(BASE_DIR, "data", "sdtatt_data.npy"), dataset)
    print(f"Processed {len(dataset)} samples.")

