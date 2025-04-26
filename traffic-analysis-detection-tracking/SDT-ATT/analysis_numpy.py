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

for sample in data:
    unique_tv_ids.add(sample['tv_id'])
    unique_frames.add(sample['center_frame'])
    nv_counts.append(len(sample.get('nv_ids', [])))

print("\nDataset Statistics:")
print(f"Number of unique target vehicles: {len(unique_tv_ids)}")
print(f"Number of unique center frames: {len(unique_frames)}")
print(f"Average number of neighbors per sample: {np.mean(nv_counts):.2f}")
print(f"Minimum number of neighbors: {min(nv_counts)}")
print(f"Maximum number of neighbors: {max(nv_counts)}")


# Print detailed information about a few samples
print("\nDetailed information for first 5 samples:")
for i in range(min(5, len(data))):
    sample = data[i]
    print(f"\nSample {i}:")
    print(f"TV ID: {sample['tv_id']}")
    print(f"Center Frame: {sample['center_frame']}")
    print(f"Number of Neighbors: {len(sample.get('nv_ids', []))}")
    print(f"TV History shape: {sample['tv_hist'].shape}")
    print(f"TV History first point: {sample['tv_hist'][0]}")
    print(f"TV History last point: {sample['tv_hist'][-1]}")


import numpy as np
import os

# Load the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(BASE_DIR, "data", "sdtatt_data.npy"), allow_pickle=True)

# Set the specific frame ID and vehicle ID you want to analyze
target_frame_id = 171  # Change this to your desired frame ID
target_vehicle_id = 8  # Change this to your desired vehicle ID

print(f"\nSearching for frame {target_frame_id} and vehicle {target_vehicle_id}...")

# Find and print information for the specific frame and vehicle
found = False
for i, sample in enumerate(data):
    if sample['center_frame'] == target_frame_id and sample['tv_id'] == target_vehicle_id:
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