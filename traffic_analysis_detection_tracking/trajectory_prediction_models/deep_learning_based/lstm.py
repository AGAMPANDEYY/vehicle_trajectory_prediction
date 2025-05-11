import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load tracking data and select data for a specific tracker
df = pd.read_csv('tracking_data.csv')
tracker_id = 1
tracker_data = df[df['tracker_id'] == tracker_id].sort_values('timestamp')
positions = tracker_data[['x', 'y']].values

# Normalize the positions for training stability
scaler = MinMaxScaler()
positions_scaled = scaler.fit_transform(positions)

# Function to create sequences: use `seq_length` timesteps to predict the next point
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # For instance, use 10 timesteps
X, y = create_sequences(positions_scaled, seq_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 2)),
    Dense(2)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)

# Predict the next position for a sample sequence from the test set
sample_sequence = X_test[0]
predicted_point = model.predict(sample_sequence[np.newaxis, ...])[0]
true_point = y_test[0]

# Inverse transform to get original scale
sample_sequence_orig = scaler.inverse_transform(sample_sequence)
predicted_point_orig = scaler.inverse_transform(predicted_point.reshape(1, -1))[0]
true_point_orig = scaler.inverse_transform(true_point.reshape(1, -1))[0]

plt.figure(figsize=(8,6))
plt.plot(sample_sequence_orig[:, 0], sample_sequence_orig[:, 1], 'bo-', label='Input Sequence')
plt.plot(true_point_orig[0], true_point_orig[1], 'go', label='True Next Point')
plt.plot(predicted_point_orig[0], predicted_point_orig[1], 'ro', label='Predicted Next Point')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Trajectory Prediction using LSTM')
plt.legend()
plt.show()
