import argparse
import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
from traffic_analysis_detection_tracking.trajectory_prediction_models.deep_learning_based.trajectory_prediction import TrajectoryPredictor
from traffic_analysis_detection_tracking.trajectory_prediction_models.physics_based.kalman_filter import KalmanFilterPredictor

# To run python -m traffic_analysis_detection_tracking.metrics.lstm_kalman

class CombinedPredictor:
    def __init__(self, lstm_model_path, sequence_length=90, prediction_length=120):
        # Initialize LSTM predictor
        self.lstm_predictor = TrajectoryPredictor(lstm_model_path, sequence_length, prediction_length)
        
        # Initialize Kalman filter
        self.kalman_filters = {}
        self.prediction_length = prediction_length
        
    def initialize_kalman_filter(self, tracker_id):
        return KalmanFilterPredictor()

class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str,
        lstm_model_path: str,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        print("\nInitializing VideoProcessor...")
        self.csv_path=r"C:\Agam\Work\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\data\combined_tracking_data.csv"
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        
        print("Loading YOLO model...")
        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()
        
        print("Loading video information...")
        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        
        print("Initializing predictors...")
        self.predictor = CombinedPredictor(lstm_model_path)
        
        print("Setting up annotators...")
        self.box_annotator = sv.BoxAnnotator()
        self.trace_annotator = sv.TraceAnnotator(
            position=sv.Position.CENTER,
            trace_length=100,
            thickness=2
        )
        
        # Initialize trajectory storage
        self.trajectories = {}
        self.current_sequences = {}
        self.bbox_sequences = {}
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize data storage for CSV
        self.tracking_data = []
        
        # Colors for different visualizations
        self.colors = {
            'actual': (0, 255, 0),    # Green for actual trajectory
            'lstm': (255, 0, 0),      # Red for LSTM predictions
            'kalman': (0, 0, 255),    # Blue for Kalman predictions
            'current': (255, 255, 0)  # Yellow for current position
        }
        print("Initialization complete!\n")

    import os, numpy as np, pandas as pd

    def process_video(self):
        df = pd.read_csv(self.csv_path)
        df_actual = df[df.prediction_type=='actual']
        df_lstm   = df[df.prediction_type=='lstm']
        df_kalman = df[df.prediction_type=='kalman']

        # block sizes
        HIST_LEN = df_actual.groupby('tracker_id')['sequence_index'].nunique().min()
        PRED_LEN = df_lstm.groupby('tracker_id')['sequence_index'].nunique().min()
        print(f"HIST_LEN={HIST_LEN}, PRED_LEN={PRED_LEN}")

        for source_df, pred_type in [(df_lstm,'lstm'), (df_kalman,'kalman')]:
            print(f"\n=== {pred_type.upper()} Error Stats ===")
            all_ade, all_fde, frame_errors = [], [], {}

            for tid in sorted(df['tracker_id'].unique()):
                act  = df_actual[df_actual['tracker_id']==tid]
                pred = source_df[source_df['tracker_id']==tid]
                cycles = len(act) // HIST_LEN

                for c in range(cycles):
                    hist_block = act.iloc[c*HIST_LEN:(c+1)*HIST_LEN]
                    pred_block = pred.iloc[c*PRED_LEN:(c+1)*PRED_LEN]

                    if len(hist_block)==HIST_LEN and len(pred_block)==PRED_LEN:
                        last_xy     = hist_block[['x','y']].iloc[-1].to_numpy(np.float32)
                        coords_pred = pred_block[['x','y']].to_numpy(np.float32)
                        coords_act  = np.tile(last_xy, (PRED_LEN,1))

                        errs = np.linalg.norm(coords_pred - coords_act, axis=1)
                        all_ade.append(errs.mean())
                        all_fde.append(errs[-1])
                        for fn, e in zip(pred_block['frame_number'], errs):
                            frame_errors[int(fn)] = e
                    else:
                        print(f"Tracker {tid} cycle {c}: missing data "
                            f"(got hist={len(hist_block)}, pred={len(pred_block)})")

            if all_ade:
                print(f"Overall ADE: {np.mean(all_ade):.2f}")
                print(f"Overall FDE: {np.mean(all_fde):.2f}")
                # save frame_errors to CSV as before…
            else:
                print(f"⚠ No complete segments for {pred_type}")


    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Process frame with YOLO
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)
        
        # Process each detection
        for tracker_id, xyxy in zip(detections.tracker_id, detections.xyxy):
            x1, y1, x2, y2 = xyxy
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Initialize trajectory storage for new tracker
            if tracker_id not in self.trajectories:
                self.trajectories[tracker_id] = []
                self.current_sequences[tracker_id] = []
                self.bbox_sequences[tracker_id] = []  # Store historical bounding boxes
                self.predictor.kalman_filters[tracker_id] = self.predictor.initialize_kalman_filter(tracker_id)
            
            # Update current sequence with both center points and bounding boxes
            self.current_sequences[tracker_id].append([center_x, center_y])
            self.bbox_sequences[tracker_id].append([x1, y1, x2, y2])
            
            if len(self.current_sequences[tracker_id]) > 20:  # Keep last 20 positions
                self.current_sequences[tracker_id].pop(0)
                self.bbox_sequences[tracker_id].pop(0)
            
            # Update Kalman filter
            kf = self.predictor.kalman_filters[tracker_id]
            if len(self.current_sequences[tracker_id]) == 1:
                # Initialize Kalman filter with first position
                kf.initialize(np.array([center_x, center_y]))
            else:
                kf.predict()
                kf.update(np.array([center_x, center_y]))
            
            # Make predictions if we have enough data
            if len(self.current_sequences[tracker_id]) == 20:
                # LSTM prediction
                lstm_predictions = self.predictor.lstm_predictor.predict_trajectory(
                    np.array(self.current_sequences[tracker_id])
                )
                
                # Kalman prediction
                kalman_predictions = kf.predict_future(self.predictor.prediction_length)
                
                # Store data for CSV
                # Store actual positions with their corresponding bounding boxes
                for i, ((actual_x, actual_y), (x1, y1, x2, y2)) in enumerate(zip(self.current_sequences[tracker_id], self.bbox_sequences[tracker_id])):
                    self.tracking_data.append({
                        'frame_number': self.frame_count - len(self.current_sequences[tracker_id]) + i,
                        'timestamp': time.time() - self.start_time,
                        'tracker_id': tracker_id,
                        'x': actual_x,
                        'y': actual_y,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'prediction_type': 'actual',
                        'sequence_index': i
                    })
                
                # Calculate average bounding box size from historical data
                avg_width = np.mean([x2 - x1 for x1, y1, x2, y2 in self.bbox_sequences[tracker_id]])
                avg_height = np.mean([y2 - y1 for x1, y1, x2, y2 in self.bbox_sequences[tracker_id]])
                
                # Store LSTM predictions with estimated bounding boxes
                for i, (pred_x, pred_y) in enumerate(lstm_predictions):
                    # Estimate bounding box for prediction
                    pred_x1 = pred_x - avg_width/2
                    pred_y1 = pred_y - avg_height/2
                    pred_x2 = pred_x + avg_width/2
                    pred_y2 = pred_y + avg_height/2
                    
                    self.tracking_data.append({
                        'frame_number': self.frame_count + i,
                        'timestamp': time.time() - self.start_time,
                        'tracker_id': tracker_id,
                        'x': pred_x,
                        'y': pred_y,
                        'x1': pred_x1,
                        'y1': pred_y1,
                        'x2': pred_x2,
                        'y2': pred_y2,
                        'prediction_type': 'lstm',
                        'sequence_index': i
                    })
                
                # Store Kalman predictions with estimated bounding boxes
                for i, (pred_x, pred_y) in enumerate(kalman_predictions):
                    # Estimate bounding box for prediction
                    pred_x1 = pred_x - avg_width/2
                    pred_y1 = pred_y - avg_height/2
                    pred_x2 = pred_x + avg_width/2
                    pred_y2 = pred_y + avg_height/2
                    
                    self.tracking_data.append({
                        'frame_number': self.frame_count + i,
                        'timestamp': time.time() - self.start_time,
                        'tracker_id': tracker_id,
                        'x': pred_x,
                        'y': pred_y,
                        'x1': pred_x1,
                        'y1': pred_y1,
                        'x2': pred_x2,
                        'y2': pred_y2,
                        'prediction_type': 'kalman',
                        'sequence_index': i
                    })
                
                # Draw actual trajectory
                actual_points = np.array(self.current_sequences[tracker_id], dtype=np.int32)
                cv2.polylines(frame, [actual_points], False, self.colors['actual'], 2)
                
                # Draw LSTM predictions
                lstm_points = np.array(lstm_predictions, dtype=np.int32)
                cv2.polylines(frame, [lstm_points], False, self.colors['lstm'], 2)
                
                # Draw Kalman predictions
                kalman_points = np.array(kalman_predictions, dtype=np.int32)
                cv2.polylines(frame, [kalman_points], False, self.colors['kalman'], 2)
                
                # Draw current position
                cv2.circle(frame, (int(center_x), int(center_y)), 5, self.colors['current'], -1)
                
                # Add legend
                legend_y = 30
                for name, color in self.colors.items():
                    cv2.putText(frame, name, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    legend_y += 20
        
        # Draw bounding boxes and tracking IDs
        frame = self.box_annotator.annotate(frame, detections)
        frame = self.trace_annotator.annotate(frame, detections)
        
        return frame

def main():
    print("\n=== Starting Traffic Analysis Pipeline ===\n")

    source_video_path=r"C:\Agam\Work\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\data\Lane_C_Video.mp4"
    source_weights_path=r"C:\Agam\Work\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\data\traffic_analysis.pt"
    target_video_path=r"C:\Agam\Work\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\data\lstm_kalman_output.py"
    lstm_model_path=r"C:\Agam\Work\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\trajectory_prediction_models\deep_learning_based\trajectory_predictor_120.pth"
    confidence_threshold=0.6
    iou_threshold=0.6


    print("Starting video processing pipeline...")
    
    # Initialize and run video processor
    processor = VideoProcessor(
        source_weights_path=source_weights_path,
        source_video_path=source_video_path,
        target_video_path=target_video_path,
        lstm_model_path=lstm_model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    
    processor.process_video()
    print("\n=== Processing Complete! ===")
    print(f"✓ Output video saved to: {target_video_path}")
    print(f"✓ Tracking data saved to: {os.path.join(os.path.dirname(target_video_path), 'combined_tracking_data.csv')}")

if __name__ == "__main__":
    main()