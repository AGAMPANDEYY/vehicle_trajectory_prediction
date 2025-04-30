import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the numpy file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "sdtatt_data.npy")
data = np.load(data_path, allow_pickle=True)

print(f"Total number of samples: {len(data)}")

# Analyze the first sample to understand the structure
first_sample = data[0]
print("\nKeys in each sample:")
print(first_sample.keys())

# Print details of the first sample
print("\nFirst sample details:")
print(f"Target Vehicle ID: {first_sample['tv_id']}")
print(f"Center Frame: {first_sample['center_frame']}")
print(f"Neighbor Vehicle IDs: {first_sample.get('nv_ids', 'Not available')}")
print(f"TV History shape: {first_sample['tv_hist'].shape}")
print(f"NV Spatial shape: {first_sample['nv_sp'].shape}")
print(f"NV Dynamic shape: {first_sample['nv_dp'].shape}")

# Collect statistics about the dataset
unique_tv_ids = set()
unique_frames = set()
nv_counts = []
# Create lists to store all frame IDs and vehicle IDs
all_frame_ids = []
all_vehicle_ids = []
frame_vehicle_pairs = []

for sample in data:
    unique_tv_ids.add(sample['tv_id'])
    unique_frames.add(sample['center_frame'])
    nv_counts.append(len(sample.get('nv_ids', [])))
    all_frame_ids.append(sample['center_frame'])
    all_vehicle_ids.append(sample['tv_id'])
    frame_vehicle_pairs.append((sample['center_frame'], sample['tv_id']))


# Convert to sets to get unique values
unique_frame_ids = sorted(set(all_frame_ids))
unique_vehicle_ids = sorted(set(all_vehicle_ids))

# Print summary statistics
print("\nDataset Summary:")
print(f"Total number of samples: {len(data)}")
print(f"Number of unique frames: {len(unique_frame_ids)}")
print(f"Number of unique vehicles: {len(unique_vehicle_ids)}")

# Print available frame IDs
print("\nAvailable Frame IDs:")
print(unique_frame_ids)

# Print available vehicle IDs
print("\nAvailable Vehicle IDs:")
print(unique_vehicle_ids)



# Find and print information for the specific frame and vehicle
import numpy as np
import os

# Load the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(BASE_DIR, "data", "sdtatt_data.npy"), allow_pickle=True)

# Set the specific frame ID and vehicle ID you want to analyze
target_frame_id = 300.0  # Keep as float
target_vehicle_id = 8    # Keep as int

print(f"\nSearching for frame {target_frame_id} and vehicle {target_vehicle_id}...")

# Find and print information for the specific frame and vehicle
found = False
for i, sample in enumerate(data):
    # Use np.isclose for float comparison and exact match for vehicle ID
    if np.isclose(sample['center_frame'], target_frame_id) and sample['tv_id'] == target_vehicle_id:
        found = True
        print(f"\nFound at sample index {i}:")
        print(f"TV ID: {sample['tv_id']}")
        print(f"Center Frame: {sample['center_frame']}")
        print(f"Neighbor Vehicle IDs: {sample.get('nv_ids', [])}")
        print(f"Number of Neighbors: {len(sample.get('nv_ids', []))}")
        print(f"TV History shape: {sample['tv_hist'].shape}")
        print("\nTV History points:")
        for j, point in enumerate(sample['tv_hist']):
            print(f"Point {j}: {point}")
        
        print("\nNeighbor Vehicle Spatial Information:")
        for n, nv_sp in enumerate(sample['nv_sp']):
            print(f"\nNeighbor {n+1} (ID: {sample['nv_ids'][n]}) spatial trajectory:")
            for j, point in enumerate(nv_sp):
                print(f"Point {j}: {point}")
        
        print("\nNeighbor Vehicle Dynamic Information:")
        for n, nv_dp in enumerate(sample['nv_dp']):
            print(f"\nNeighbor {n+1} (ID: {sample['nv_ids'][n]}) dynamic trajectory:")
            for j, point in enumerate(nv_dp):
                print(f"Point {j}: {point}")
        break

if not found:
    print(f"No sample found for frame {target_frame_id} and vehicle {target_vehicle_id}")
    print("\nLet's check what samples exist for this frame:")
    frame_samples = [s for s in data if np.isclose(s['center_frame'], target_frame_id)]
    if frame_samples:
        print(f"\nFound {len(frame_samples)} samples for frame {target_frame_id}:")
        for s in frame_samples:
            print(f"Vehicle ID: {s['tv_id']}")
    else:
        print(f"No samples found for frame {target_frame_id}")