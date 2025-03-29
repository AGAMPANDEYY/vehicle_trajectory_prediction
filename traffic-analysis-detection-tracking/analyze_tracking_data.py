import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def analyze_tracking_data(csv_path):
    """
    Analyze the combined tracking data CSV file and generate insights.
    
    Args:
        csv_path (str): Path to the combined_tracking_data.csv file
    """
    print("\n=== Starting Tracking Data Analysis ===\n")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Basic statistics
    print("Basic Statistics:")
    print(f"Total number of data points: {len(df)}")
    print(f"Number of unique vehicles (tracker_ids): {df['tracker_id'].nunique()}")
    print(f"Number of frames: {df['frame_number'].nunique()}")
    print(f"Time span: {df['timestamp'].max() - df['timestamp'].min():.2f} seconds")
    
    # Analyze prediction types
    print("\nPrediction Types Distribution:")
    prediction_counts = df['prediction_type'].value_counts()
    print(prediction_counts)
    
    # Analyze bounding box statistics
    print("\nBounding Box Statistics:")
    bbox_stats = df[['x1', 'y1', 'x2', 'y2']].describe()
    print(bbox_stats)
    
    # Calculate vehicle sizes
    df['width'] = df['x2'] - df['x1']
    df['height'] = df['y2'] - df['y1']
    
    print("\nVehicle Size Statistics:")
    size_stats = df[['width', 'height']].describe()
    print(size_stats)
    
    # Create visualizations
    output_dir = os.path.dirname(csv_path)
    plots_dir = os.path.join(output_dir, 'analysis_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Vehicle Trajectories
    plt.figure(figsize=(12, 8))
    for tracker_id in df['tracker_id'].unique()[:5]:  # Plot first 5 vehicles
        vehicle_data = df[df['tracker_id'] == tracker_id]
        plt.plot(vehicle_data['x'], vehicle_data['y'], label=f'Vehicle {tracker_id}')
    plt.title('Vehicle Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'vehicle_trajectories.png'))
    plt.close()
    
    # 2. Prediction Types Distribution
    plt.figure(figsize=(8, 6))
    prediction_counts.plot(kind='bar')
    plt.title('Distribution of Prediction Types')
    plt.xlabel('Prediction Type')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plots_dir, 'prediction_types.png'))
    plt.close()
    
    # 3. Vehicle Size Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['width', 'height']])
    plt.title('Vehicle Size Distribution')
    plt.ylabel('Pixels')
    plt.savefig(os.path.join(plots_dir, 'vehicle_sizes.png'))
    plt.close()
    
    # 4. Time Series of Vehicle Counts
    plt.figure(figsize=(12, 6))
    vehicle_counts = df.groupby('frame_number')['tracker_id'].nunique()
    plt.plot(vehicle_counts.index, vehicle_counts.values)
    plt.title('Number of Vehicles Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Vehicles')
    plt.savefig(os.path.join(plots_dir, 'vehicle_counts.png'))
    plt.close()
    
    # Save summary statistics to a text file
    with open(os.path.join(plots_dir, 'summary_statistics.txt'), 'w') as f:
        f.write("=== Tracking Data Analysis Summary ===\n\n")
        f.write(f"Total data points: {len(df)}\n")
        f.write(f"Unique vehicles: {df['tracker_id'].nunique()}\n")
        f.write(f"Total frames: {df['frame_number'].nunique()}\n")
        f.write(f"Time span: {df['timestamp'].max() - df['timestamp'].min():.2f} seconds\n\n")
        f.write("Prediction Types Distribution:\n")
        f.write(prediction_counts.to_string())
        f.write("\n\nBounding Box Statistics:\n")
        f.write(bbox_stats.to_string())
        f.write("\n\nVehicle Size Statistics:\n")
        f.write(size_stats.to_string())
    
    print(f"\nAnalysis complete! Plots and statistics saved to: {plots_dir}")
    print("Generated files:")
    print("1. vehicle_trajectories.png - Visualization of vehicle paths")
    print("2. prediction_types.png - Distribution of prediction types")
    print("3. vehicle_sizes.png - Distribution of vehicle sizes")
    print("4. vehicle_counts.png - Number of vehicles over time")
    print("5. summary_statistics.txt - Detailed numerical statistics")

def main():
    # Get the CSV file path from the command line or use default
    import argparse
    parser = argparse.ArgumentParser(description="Analyze combined tracking data CSV file")
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to the combined_tracking_data.csv file",
        default="combined_tracking_data.csv"
    )
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: File {args.csv_path} not found!")
        return
    
    analyze_tracking_data(args.csv_path)

if __name__ == "__main__":
    main() 