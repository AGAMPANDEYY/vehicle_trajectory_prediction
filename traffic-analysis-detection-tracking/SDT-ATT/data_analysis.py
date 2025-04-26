import numpy as np

# Load the raw npy file directly
data = np.load(r"D:\vehicle_trajectory_prediction\traffic-analysis-detection-tracking\SDT-ATT\data\sdtatt_data.npy", allow_pickle=True)

# Collect unique frame IDs and track IDs
unique_frames = set()
unique_tracks = set()
frame_track_pairs = set()

for sample in data:
    unique_frames.add(sample['center_frame'])
    unique_tracks.add(sample['tv_id'])
    frame_track_pairs.add((sample['center_frame'], sample['tv_id']))

print(f"Number of unique center frames: {len(unique_frames)}")
print(f"Number of unique track IDs: {len(unique_tracks)}")
print(f"Total unique (frame, track) pairs: {len(frame_track_pairs)}")


"""
USER@DESKTOP-9HMMGG6 MINGW64 /c/Agam/Work/vehicle_trajectory_prediction (main)
$  python traffic-analysis-detection-tracking/SDT-ATT/data_analysis.py
Number of unique center frames: 9005
Number of unique track IDs: 974
Total unique (frame, track) pairs: 336283

"""