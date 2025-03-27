import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import cv2
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
import time
import os

"""
This is a simple LSTM-based trajectory prediction model.
It uses an LSTM network to predict the future trajectory of a vehicle based on its past positions.
Data format: {tracker_id, Time, x, y}
"""

class TrajectoryDataset(Dataset):
    def __init__(self, data, sequence_length, prediction_length):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.data = data
        
        # Group data by tracker_id and keep only those with enough data points
        self.trajectories = []
        for tracker_id, group in data.groupby('tracker_id'):
            if len(group) >= sequence_length + prediction_length:
                self.trajectories.append(group[['x', 'y']].values)
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        max_start = len(trajectory) - self.sequence_length - self.prediction_length
        # If max_start is 0 or negative, force start_idx to 0
        start_idx = 0 if max_start <= 0 else np.random.randint(0, max_start)
        
        input_seq = trajectory[start_idx : start_idx + self.sequence_length]
        target_seq = trajectory[start_idx + self.sequence_length : start_idx + self.sequence_length + self.prediction_length]
        
        return torch.FloatTensor(input_seq), torch.FloatTensor(target_seq)

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2, prediction_length=10):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_length = prediction_length
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * prediction_length)
    
    def forward(self, x):
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions.view(batch_size, self.prediction_length, 2)

def train_model(model, train_loader, num_epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

class TrajectoryPredictor:
    def __init__(self, model_path, sequence_length=20, prediction_length=10):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.model = LSTMPredictor(prediction_length=prediction_length)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Initialize scaler for coordinate normalization
        # Note: For consistency, consider fitting the scaler on training data and saving it.
        self.scaler = MinMaxScaler()
        
    def predict_trajectory(self, current_trajectory):
        # Normalize the input sequence
        normalized_trajectory = self.scaler.fit_transform(current_trajectory)
        
        # Prepare input tensor
        input_tensor = torch.FloatTensor(normalized_trajectory).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Denormalize predictions
        predictions = predictions.squeeze(0).numpy()
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions

def visualize_predictions(video_path, model_path, output_path, sequence_length=20, prediction_length=10):
    # Initialize YOLO and tracker
    model = YOLO('yolov8n.pt')
    tracker = sv.ByteTrack()
    
    # Initialize trajectory predictor
    predictor = TrajectoryPredictor(model_path, sequence_length, prediction_length)
    
    # Initialize video capture and writer
    cap = cv2.VideoCapture(video_path)
    video_info = sv.VideoInfo.from_video_path(video_path)
    
    # Initialize trajectory storage
    trajectories = {}
    current_sequences = {}
    predictions_data = []
    frame_count = 0
    
    with sv.VideoSink(output_path, video_info) as sink:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame with YOLO
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)
            
            # Update trajectories and make predictions
            for tracker_id, xyxy in zip(detections.tracker_id, detections.xyxy):
                x1, y1, x2, y2 = xyxy
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                if tracker_id not in trajectories:
                    trajectories[tracker_id] = []
                    current_sequences[tracker_id] = []
                
                current_sequences[tracker_id].append([center_x, center_y])
                
                # Keep only the last sequence_length points
                if len(current_sequences[tracker_id]) > sequence_length:
                    current_sequences[tracker_id].pop(0)
                
                # Make prediction if we have enough points
                if len(current_sequences[tracker_id]) == sequence_length:
                    predictions = predictor.predict_trajectory(np.array(current_sequences[tracker_id]))
                    
                    # Store predictions in DataFrame format
                    for i, (pred_x, pred_y) in enumerate(predictions):
                        predictions_data.append({
                            'timestamp': frame_count + i,
                            'tracker_id': tracker_id,
                            'x': pred_x,
                            'y': pred_y,
                            'is_prediction': True,
                            'frame_number': frame_count
                        })
                    
                    # Store actual positions
                    for i, (actual_x, actual_y) in enumerate(current_sequences[tracker_id]):
                        predictions_data.append({
                            'timestamp': frame_count - len(current_sequences[tracker_id]) + i,
                            'tracker_id': tracker_id,
                            'x': actual_x,
                            'y': actual_y,
                            'is_prediction': False,
                            'frame_number': frame_count
                        })
                    
                    # Draw actual trajectory
                    actual_points = np.array(current_sequences[tracker_id], dtype=np.int32)
                    cv2.polylines(frame, [actual_points], False, (0, 255, 0), 2)
                    
                    # Draw predicted trajectory
                    predicted_points = np.array(predictions, dtype=np.int32)
                    cv2.polylines(frame, [predicted_points], False, (255, 0, 0), 2)
                    
                    # Draw current position
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            
            # Write frame
            sink.write_frame(frame)
            frame_count += 1
    
    cap.release()
    
    # Save predictions to CSV
    if predictions_data:
        predictions_df = pd.DataFrame(predictions_data)
        data_dir = os.path.dirname(video_path)
        output_csv = os.path.join(data_dir, 'lstm_predictions.csv')
        predictions_df.to_csv(output_csv, index=False)
        print(f"LSTM predictions saved to {output_csv}")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir = 'C:/Agam/Work/cen-300/supervision/examples/traffic_analysis/'
    os.makedirs(data_dir, exist_ok=True)
    
    # First, train the model
    # Load your tracking data
    data = pd.read_csv(os.path.join(data_dir, 'data/tracking_data.csv'))
    
    # Create dataset and dataloader
    dataset = TrajectoryDataset(data, sequence_length=20, prediction_length=10)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize and train model
    model = LSTMPredictor(prediction_length=10)
    train_model(model, train_loader)
    
    # Save the trained model
    model_path = os.path.join(data_dir, 'trajectory_prediction_models/deep_learning_based/trajectory_predictor.pth')
    torch.save(model.state_dict(), model_path)
    
    # Then, visualize predictions on video
    video_path = os.path.join(data_dir, 'data/0212.mp4')
    output_path = os.path.join(data_dir, 'trajectory_prediction_models/deep_learning_based/output_with_predictions.mp4')
    
    visualize_predictions(
        video_path=video_path,
        model_path=model_path,
        output_path=output_path
    )
