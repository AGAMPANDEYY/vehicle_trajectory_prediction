import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
import os

"""
This is a simple Kalman filter implementation for trajectory prediction.
It assumes a constant velocity model and uses a 2D state vector [x, y, vx, vy].

The filter estimates the position and velocity of a target based on noisy measurements.
"""

class KalmanFilterPredictor:
    def __init__(self, dim_x=4, dim_z=2):
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.dt = 1.0  # Time step
        
        # State transition matrix: [x, y, vx, vy]
        self.kf.F = np.array([[1, 0, self.dt, 0],
                             [0, 1, 0, self.dt],
                             [0, 0, 1,  0],
                             [0, 0, 0,  1]])
        
        # Measurement function: we measure x and y only
        self.kf.H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]])
        
        # Measurement noise covariance
        self.kf.R = np.eye(2) * 0.5
        
        # Estimate error covariance matrix
        self.kf.P *= 1000.
        
        # Process noise covariance
        self.kf.Q = np.eye(4) * 0.01

    def initialize(self, initial_pos):
        """Initialize the filter with first position"""
        self.kf.x = np.array([initial_pos[0], initial_pos[1], 0, 0]).reshape(4, 1)

    def predict(self):
        """Make a prediction step"""
        self.kf.predict()

    def update(self, measurement):
        """Update the filter with a new measurement"""
        self.kf.update(measurement)

    def get_state(self):
        """Get current state estimate"""
        return self.kf.x[:2].flatten()

    def predict_future(self, steps=10):
        """Predict future positions"""
        future_positions = []
        # Create a deep copy of the Kalman filter state
        kf_copy = KalmanFilter(dim_x=self.kf.dim_x, dim_z=self.kf.dim_z)
        kf_copy.F = self.kf.F.copy()
        kf_copy.H = self.kf.H.copy()
        kf_copy.R = self.kf.R.copy()
        kf_copy.P = self.kf.P.copy()
        kf_copy.Q = self.kf.Q.copy()
        kf_copy.x = self.kf.x.copy()
        
        for _ in range(steps):
            kf_copy.predict()
            future_positions.append(kf_copy.x[:2].flatten())
        
        return np.array(future_positions)

def process_tracking_data(data_path, tracker_id, save_plots=True):
    """Process tracking data for a specific tracker ID"""
    # Load tracking data
    df = pd.read_csv(data_path)
    
    # Select data for specific tracker_id and sort by timestamp
    tracker_data = df[df['tracker_id'] == tracker_id].sort_values('timestamp')
    
    # Initialize Kalman filter
    kf_predictor = KalmanFilterPredictor()
    
    # Initialize with first position
    initial_pos = tracker_data[['x', 'y']].iloc[0].values
    kf_predictor.initialize(initial_pos)
    
    predicted_positions = []
    
    # Process each measurement
    for _, row in tracker_data.iterrows():
        z = np.array([row['x'], row['y']])
        kf_predictor.predict()
        kf_predictor.update(z)
        predicted_positions.append(kf_predictor.get_state())
    
    predicted_positions = np.array(predicted_positions)
    
    # Predict future positions
    future_positions = kf_predictor.predict_future()
    
    if save_plots:
        # Create plots directory if it doesn't exist
        data_dir = os.path.dirname(data_path)
        plots_dir = os.path.join(data_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot results
        plt.figure(figsize=(8,6))
        plt.plot(tracker_data['x'], tracker_data['y'], 'bo-', label='Measured')
        plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], 'go-', label='Kalman Estimate')
        plt.plot(future_positions[:, 0], future_positions[:, 1], 'ro--', label='Future Prediction')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Trajectory Prediction using Kalman Filter (Tracker {tracker_id})')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'kalman_prediction_tracker_{tracker_id}.png'))
        plt.close()
    
    return predicted_positions, future_positions

if __name__ == "__main__":
    # Example usage
    data_path = 'data/tracking_data.csv'
    tracker_id = 1
    predicted_positions, future_positions = process_tracking_data(data_path, tracker_id)
