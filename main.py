import argparse
import os
import pandas as pd
import cv2
import numpy as np

# Import necessary classes and functions (assuming they are in separate modules)
from app.modules.video_processor import VideoProcessor
from app.modules.sdtatt import SDTATT_predict_vehicle, SDTATT_predict_all
from app.modules.pet import PETPipeline


def main(video_path, tracking_path, yolo_weights, lstm_model, sdtatt_data_path,
         sdtatt_checkpoint, choice, frame_id, vehicle_id, output_dir, zone_path):
  
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Generate tracking data if not provided
    if tracking_path is None:
        target_video_path = os.path.join(output_dir, "processed_video.mp4")
        processor = VideoProcessor(
            source_weights_path=yolo_weights,
            source_video_path=video_path,
            target_video_path=target_video_path,
            lstm_model_path=lstm_model
        )
        processor.process_video()
        tracking_path = os.path.join(os.path.dirname(target_video_path), "combined_tracking_data.csv")
    else:
        # Use provided tracking CSV
        pass

    # Step 2: Run SDTATT trajectory prediction
    if choice == "all":
        future_csv = os.path.join(data_dir, "future_trajectories.csv")
        SDTATT_predict_all(sdtatt_data_path, tracking_path, sdtatt_checkpoint, future_csv)
    elif choice == "single":
        if frame_id is None or vehicle_id is None:
            print("Error: For single vehicle prediction, frame_id and vehicle_id must be provided.")
            exit(1)
        pred_abs, center_frame, track_id = SDTATT_predict_vehicle(frame_id, vehicle_id, sdtatt_data_path, sdtatt_checkpoint)
        pred_df = pd.DataFrame(pred_abs.numpy(), columns=["x_future", "y_future"])
        pred_df["frame_id"] = range(frame_id + 1, frame_id + 1 + len(pred_abs))
        pred_df["vehicle_id"] = vehicle_id
        pred_df["x1"] = 0  # Dummy bounding box values (not used in PET)
        pred_df["y1"] = 0
        pred_df["x2"] = 0
        pred_df["y2"] = 0
        future_csv = os.path.join(data_dir, "future_trajectories_single.csv")
        pred_df.to_csv(future_csv, index=False)

    # Step 3: Prepare predicted trajectories for PET analysis
    pet_input_csv = os.path.join(data_dir, "pet_input.csv")
    future_df = pd.read_csv(future_csv)
    future_df['timestamp'] = future_df['frame_id']  # Use frame_id as a proxy for time
    future_df['frame_number'] = future_df['frame_id']
    future_df['tracker_id'] = future_df['vehicle_id']  # Map vehicle_id to tracker_id
    future_df['x'] = future_df['x_future']
    future_df['y'] = future_df['y_future']
    # Ensure bounding box columns are present (even if dummy values)
    if 'x1' not in future_df.columns:
        future_df['x1'] = 0
        future_df['y1'] = 0
        future_df['x2'] = 0
        future_df['y2'] = 0
    future_df = future_df[['timestamp', 'frame_number', 'tracker_id', 'x', 'y', 'x1', 'y1', 'x2', 'y2']]
    future_df.to_csv(pet_input_csv, index=False)

    # Step 4: Run PET Pipeline using predicted trajectories
    pet_pipeline = PETPipeline(
        tracking_path=pet_input_csv,  # Use prepared predicted trajectories
        zone_path=zone_path,
        video_path=video_path  # For metadata like width, height, fps
    )
    pet_pipeline.run()
    pet_csv = os.path.join(data_dir, "pet_results.csv")
    pet_pipeline.pet_df.to_csv(pet_csv, index=False)

    # Step 5: Generate and save heatmap
    heatmap = pet_pipeline.generate_heatmap()

    heatmap_video= pet_pipeline.generate_heatmap_video(heatmap)
    
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_normalized.astype(np.uint8)
    heatmap_path = os.path.join(data_dir, "heatmap.png")
    cv2.imwrite(heatmap_path, heatmap_uint8)

    # Print completion messages
    print(f"PET results saved to {pet_csv}")
    print(f"Heatmap saved to {heatmap_path}")

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)
    # Define input variables manually
    VIDEO_PATH = r"C:\Agam\Work\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\data\Lane_C_Video.mp4"  # Path to the input video
    YOLO_WEIGHTS = r"C:\Agam\Work\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\data\best.pt"  # Path to YOLO weights
    LSTM_MODEL = r"C:\Agam\Work\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\trajectory_prediction_models\deep_learning_based\trajectory_predictor_120.pth"  # Path to LSTM model for trajectory prediction
    CHOICE = "all"  # Choose "all" for all vehicles or "single" for a specific vehicle
    FRAME_ID = 320  # Frame ID for single vehicle prediction (only if choice is "single")
    VEHICLE_ID = 8  # Vehicle ID for single vehicle prediction (only if choice is "single")
    OUTPUT_DIR = r"C:\Agam\Work\vehicle_trajectory_prediction\output"  # Directory to save outputs
    ZONE_PATH = r"C:\Agam\Work\vehicle_trajectory_prediction\output"  # Path to conflict zones CSV
   
    SDTATT_NUMPY_DATA_PATH = os.path.join(BASE_DIR, "data", "sdtatt_data.npy")
    SDTATT_NUMPY_DATA_PATH="/kaggle/input/sdtatt-np/sdtatt_data.npy"
    TRACKING_CSV_PATH = os.path.join(PARENT_DIR, "data", "combined_tracking_data.csv")
    TRACKING_CSV_PATH= "/kaggle/input/combined-tracking-data/combined_tracking_data.csv"
    SDTATT_CHECKPOINT_PATH=os.path.join(BASE_DIR, "checkpoint","sdtatt_final.pt")
    SDTATT_CHECKPOINT_PATH="/kaggle/input/sdtatt_np/pytorch/default/1/sdtatt_final.pt"

    main(VIDEO_PATH, TRACKING_CSV_PATH, YOLO_WEIGHTS, LSTM_MODEL, SDTATT_NUMPY_DATA_PATH,
         SDTATT_CHECKPOINT_PATH, CHOICE, VEHICLE_ID, OUTPUT_DIR, ZONE_PATH)
    