import torch
from collections import deque
import numpy as np

import argparse
import os
import time
import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv

from traffic_analysis_detection_tracking.trajectory_prediction_models.deep_learning_based.trajectory_prediction import TrajectoryPredictor
from traffic_analysis_detection_tracking.trajectory_prediction_models.physics_based.kalman_filter import KalmanFilterPredictor

# --- New SDT-ATT wrapper ---
class SDTATTPredictor:
    def __init__(self, checkpoint_path, input_dim=2, hidden_dim=64, 
                 num_neighbors=3, future_len=120, history_len=90, device=None):
        from SDT_ATT.model import SDTATTModel
        self.future_len = future_len
        self.history_len = history_len
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model
        self.model = SDTATTModel(input_dim, hidden_dim, num_neighbors, future_len)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device).eval()

        # buffer to store each vehicle's history of centers
        self.histories = {}  # tracker_id -> deque of (x,y)

    def update_history(self, tracker_id, x, y):
        if tracker_id not in self.histories:
            self.histories[tracker_id] = deque(maxlen=self.history_len)
        self.histories[tracker_id].append((x, y))

    def predict(self, tracker_id):
        hist = self.histories.get(tracker_id)
        if not hist or len(hist) < self.history_len:
            return None

        tv_hist = torch.tensor([list(hist)], dtype=torch.float32, device=self.device)
        nv_sp = torch.zeros((1, self.model.num_neighbors, self.history_len, 2), device=self.device)
        nv_dp = torch.zeros_like(nv_sp)

        with torch.no_grad():
            output = self.model(tv_hist, nv_sp, nv_dp)
        pred_rel = output[0, :, :2]
        pred_abs = torch.cumsum(pred_rel, dim=0) + tv_hist[0, -1]
        return pred_abs.cpu().numpy()  # shape (future_len,2)

# --- CombinedPredictor extended ---
class CombinedPredictor:
    def __init__(self, lstm_model_path, kalman_params, sdtatt_ckpt,
                 sequence_length=90, prediction_length=120):
        self.lstm = TrajectoryPredictor(lstm_model_path, sequence_length, prediction_length)
        self.kalman = {}
        self.kalman_params = kalman_params
        self.pred_len = prediction_length

        # SDT-ATT setup
        self.sdtatt = SDTATTPredictor(
            checkpoint_path=sdtatt_ckpt,
            future_len=prediction_length,
            history_len=sequence_length
        )

    def initialize_kalman_filter(self, tracker_id):
        kf = KalmanFilterPredictor(**self.kalman_params)
        self.kalman[tracker_id] = kf
        return kf

    def update_and_predict(self, tracker_id, history_xy):
        # LSTM
        lstm_pred = self.lstm.predict(history_xy)

        # Kalman
        if tracker_id not in self.kalman:
            self.initialize_kalman_filter(tracker_id)
        kalman_pred = self.kalman[tracker_id].predict(history_xy)

        # SDT-ATT: update with latest point then predict
        self.sdtatt.update_history(tracker_id, *history_xy[-1])
        sdtatt_pred = self.sdtatt.predict(tracker_id)

        return {'lstm': lstm_pred, 'kalman': kalman_pred, 'sdtatt': sdtatt_pred}

class VideoProcessor:
    def __init__(
        self,
        tracking_csv_path: str,
        source_video_path: str,
        target_video_path: str,
        lstm_model_path: str,
        kalman_params: dict,
        sdtatt_ckpt: str,
        history_len: int = 20,
        pred_len: int = 90,
    ) -> None:
        # Load saved detections/tracking
        self.tracking_df = pd.read_csv(tracking_csv_path)
        self.frame_ids = sorted(self.tracking_df['frame_number'].unique())

        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.video = cv2.VideoCapture(source_video_path)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.video.get(cv2.CAP_PROP_FPS)

        self.predictor = CombinedPredictor(
            lstm_model_path=lstm_model_path,
            kalman_params=kalman_params,
            sdtatt_ckpt=sdtatt_ckpt,
            sequence_length=history_len,
            prediction_length=pred_len
        )

        # Storage
        self.current_sequences = {}   # tracker_id -> deque
        self.bbox_sequences = {}
        self.tracking_data = []
        self.frame_count = 0
        self.start_time = time.time()

        # Colors
        self.colors = {
            'actual': (0,255,0),
            'lstm':   (255,0,0),
            'kalman': (0,0,255),
            'sdtatt': (0,200,255),
            'current':(255,255,0)
        }

    def process_video(self):
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.target_video_path, fourcc, self.fps, (self.width, self.height))

        for frame_number in tqdm(self.frame_ids, desc="Processing frames"):
            ret, frame = self.video.read()
            if not ret: break

            # Get detections for this frame
            rows = self.tracking_df[self.tracking_df['frame_number']==frame_number]
            for _, row in rows.iterrows():
                tid = int(row['tracker_id'])
                x1,y1,x2,y2 = row[['x1','y1','x2','y2']]
                cx,cy = (x1+x2)/2, (y1+y2)/2

                # Init buffers
                if tid not in self.current_sequences:
                    self.current_sequences[tid] = deque(maxlen=self.predictor.sdtatt.history_len)
                    self.bbox_sequences[tid] = deque(maxlen=self.predictor.sdtatt.history_len)
                    self.predictor.initialize_kalman_filter(tid)

                seq = self.current_sequences[tid]; bbox_seq = self.bbox_sequences[tid]
                seq.append((cx,cy)); bbox_seq.append((x1,y1,x2,y2))

                # Kalman update
                kf = self.predictor.kalman[tid]
                if len(seq)==1:
                    kf.initialize(np.array([cx,cy]))
                else:
                    kf.predict(); kf.update(np.array([cx,cy]))

                # Predict when enough history
                if len(seq)==self.predictor.sdtatt.history_len:
                    preds = self.predictor.update_and_predict(tid, list(seq))

                    # record actual
                    for i,(ax,ay) in enumerate(seq):
                        x1_,y1_,x2_,y2_ = bbox_seq[i]
                        self.tracking_data.append({
                            'frame_number': frame_number - len(seq)+i,
                            'tracker_id': tid,
                            'x': ax,'y': ay,
                            'prediction_type':'actual','sequence_index':i
                        })
                    # avg box
                    widths=[b[2]-b[0] for b in bbox_seq]; heights=[b[3]-b[1] for b in bbox_seq]
                    aw,ah = np.mean(widths), np.mean(heights)

                    # record preds
                    for mode in ['lstm','kalman','sdtatt']:
                        p = preds[mode]
                        if p is None: continue
                        for i,(px,py) in enumerate(p):
                            self.tracking_data.append({
                                'frame_number': frame_number + i,
                                'tracker_id':tid,
                                'x':px,'y':py,
                                'prediction_type':mode,'sequence_index':i
                            })

                    # draw
                    for mode,color in self.colors.items():
                        if mode=='current': continue
                        dfm = pd.DataFrame([d for d in self.tracking_data if d['prediction_type']==mode and d['tracker_id']==tid])
                        pts = dfm.sort_values('sequence_index')[['x','y']].to_numpy().astype(int)
                        if len(pts)>1: cv2.polylines(frame,[pts],False,color,2)

            out.write(frame)

        out.release(); self.video.release()

          # Organize actual and predicted data by tracker_id
        df = pd.DataFrame(self.tracking_data)
        df_actual  = df[df['prediction_type']=='actual']
        df_lstm    = df[df['prediction_type']=='lstm']
        df_kalman  = df[df['prediction_type']=='kalman']
        df_sdtatt  = df[df['prediction_type']=='sdtatt']   # <— new

        # Now loop through all three predictors
        for pred_df, pred_type in zip(
            [df_lstm, df_kalman, df_sdtatt], 
            ['lstm', 'kalman', 'sdtatt']
        ):
            print(f"\n=== {pred_type.upper()} Trajectory Error Statistics ===")
            errors = []
            frame_errors = {}

            for tracker_id in df_actual['tracker_id'].unique():
                actual_pts    = df_actual [df_actual ['tracker_id']==tracker_id] \
                                    .sort_values('sequence_index')
                predicted_pts = pred_df   [pred_df   ['tracker_id']==tracker_id] \
                                    .sort_values('sequence_index')

                # both should have the same length (= prediction_length)
                if (len(actual_pts)==self.predictor.pred_len 
                and len(predicted_pts)==self.predictor.pred_len):
                    # ground-truth future is last actual point repeated
                    last_actual = actual_pts.iloc[-1][['x','y']].to_numpy(dtype=np.float32)
                    actual_future = np.tile(last_actual, (self.predictor.pred_len,1))
                    
                    pred_coords   = predicted_pts[['x','y']].to_numpy(dtype=np.float32)
                    actual_coords = actual_future

                    # Euclidean distance per step
                    step_errors = np.linalg.norm(pred_coords - actual_coords, axis=1)
                    errors.extend(step_errors.tolist())

                    # record per-frame errors if you like
                    for i, err in enumerate(step_errors):
                        frame_no = int(predicted_pts.iloc[i]['frame_number'])
                        frame_errors[frame_no] = err

            if errors:
                ade = np.mean(errors)
                fde = errors[-1]
                print(f"ADE: {ade:.2f} px")
                print(f"FDE: {fde:.2f} px")
                print(f"Max Error: {np.max(errors):.2f} px")
                print(f"Min Error: {np.min(errors):.2f} px")
                print(f"Std Dev: {np.std(errors):.2f} px")

                # Save errors to CSV
                error_df = pd.DataFrame({
                    'frame': list(frame_errors.keys()),
                    'error': list(frame_errors.values())
                })
                error_csv = os.path.join(
                    os.path.dirname(self.target_video_path),
                    f"{pred_type}_trajectory_errors.csv"
                )
                error_df.to_csv(error_csv, index=False)
                print(f"✓ {pred_type} errors saved to {error_csv}")
            else:
                print(f"⚠ No {pred_type} predictions found.")
            
        print("\nEvaluating trajectory prediction accuracy...")
        all_errors = []
        frame_errors = {}

        # Organize actual and predicted data by tracker_id
        df = pd.DataFrame(self.tracking_data)
        df_actual = df[df['prediction_type'] == 'actual']
        df_lstm = df[df['prediction_type'] == 'lstm']
        df_kalman = df[df['prediction_type'] == 'kalman']

        for pred_df, pred_type in zip([df_lstm, df_kalman], ['lstm', 'kalman']):
            print(f"\n=== {pred_type.upper()} Trajectory Error Statistics ===")

            errors = []

            for tracker_id in df_actual['tracker_id'].unique():
                actual_points = df_actual[df_actual['tracker_id'] == tracker_id].sort_values('sequence_index')
                predicted_points = pred_df[pred_df['tracker_id'] == tracker_id].sort_values('sequence_index')

                if len(actual_points) == 20 and len(predicted_points) == 10:
                    last_actual = actual_points.iloc[-1][['x', 'y']].values.astype(np.float32)

                    actual_future = []
                    for i in range(10):
                        actual_future.append([last_actual[0], last_actual[1]])  # Assuming stationary after last actual

                    pred_coords = predicted_points[['x', 'y']].values.astype(np.float32)
                    actual_coords = np.array(actual_future)

                    # Euclidean distance
                    frame_errors_tracker = np.linalg.norm(pred_coords - actual_coords, axis=1)
                    errors.extend(frame_errors_tracker.tolist())

                    # Optionally log individual errors per frame
                    for i, err in enumerate(frame_errors_tracker):
                        frame_errors[int(predicted_points.iloc[i]['frame_number'])] = err

            if errors:
                ade = np.mean(errors)
                fde = errors[-1] if len(errors) > 0 else 0
                print(f"ADE: {ade:.2f} pixels")
                print(f"FDE: {fde:.2f} pixels")
                print(f"Max Error: {np.max(errors):.2f} pixels")
                print(f"Min Error: {np.min(errors):.2f} pixels")
                print(f"Std Dev: {np.std(errors):.2f} pixels")

                # Save to CSV
                error_df = pd.DataFrame({
                    'frame': list(frame_errors.keys()),
                    'error': list(frame_errors.values())
                })
                error_csv_path = os.path.join(os.path.dirname(self.target_video_path), f"{pred_type}_trajectory_errors.csv")
                error_df.to_csv(error_csv_path, index=False)
                print(f"✓ Error data saved to: {error_csv_path}")
            else:
                print(f"⚠ No predictions found for {pred_type}")


def main():
    print("\n=== Starting Traffic Analysis Pipeline ===\n")

    source_video_path=r"D:\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\data\Lane_C_Video.mp4"
    source_weights_path=r"D:\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\data\traffic_analysis.pt"
    target_video_path=r"D:\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\data\lstm_kalman_output.py"
    lstm_model_path=r"D:\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\trajectory_prediction_models\deep_learning_based\trajectory_predictor.pth"
    confidence_threshold=0.6
    iou_threshold=0.6
    sdtatt_model_path=r"D:\vehicle_trajectory_prediction\traffic_analysis_detection_tracking\SDT-ATT\chkpt\sdtatt_final.pt"


    from collections import defaultdict

    # initialize
    comb = CombinedPredictor(
        lstm_model_path=lstm_model_path, 
        kalman_params={'R':1.0,'Q':0.01},     # example params
        sdtatt_ckpt=sdtatt_model_path,
        sequence_length=90,
        prediction_length=120
    )

    # maintain per-tracker history
    histories = defaultdict(lambda: deque(maxlen=90))

    # per-frame storage of errors
    errors = {'lstm':[], 'kalman':[], 'sdtatt':[]}

if __name__ == "__main__":
    main()