import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----- Step 1: Create Dataset -----
class TrajectoryDataset(Dataset):
    def __init__(self, positions, seq_length):
        """
        positions: numpy array of shape (num_points, 2) with columns [x, y]
        seq_length: number of timesteps used as input sequence
        """
        self.positions = positions
        self.seq_length = seq_length

    def __len__(self):
        return len(self.positions) - self.seq_length

    def __getitem__(self, idx):
        # Input sequence and the target (the next position)
        seq = self.positions[idx : idx + self.seq_length]
        target = self.positions[idx + self.seq_length]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# Load your tracking CSV (adjust the file path as needed)
df = pd.read_csv('tracking_data.csv')
# For this example, we select data for a specific vehicle (tracker_id == 1) and sort by timestamp.
tracker_id = 1
tracker_data = df[df['tracker_id'] == tracker_id].sort_values('timestamp')
positions = tracker_data[['x', 'y']].values

# Normalize positions (optional but recommended for stability)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
positions_norm = scaler.fit_transform(positions)

# Create dataset and dataloader
seq_length = 10  # Number of timesteps in input sequence
dataset = TrajectoryDataset(positions_norm, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ----- Step 2: Define Positional Encoding -----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        """
        Implements positional encoding as in "Attention is All You Need".
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, d_model) with positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ----- Step 3: Define the Transformer Model -----
class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=2, model_dim=64, num_heads=4, num_layers=3, dropout=0.1):
        """
        A simple Transformer encoder that takes a sequence of (x, y) and outputs the next (x, y).
        """
        super(TrajectoryTransformer, self).__init__()
        self.model_dim = model_dim
        # Embed the 2D positions to model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final fully connected layer to map back to 2D coordinates
        self.fc_out = nn.Linear(model_dim, input_dim)

    def forward(self, src):
        """
        src: shape (batch_size, seq_length, input_dim)
        """
        # Embed input and apply positional encoding
        src = self.embedding(src) * np.sqrt(self.model_dim)
        src = self.pos_encoder(src)
        # Transformer expects input shape (seq_length, batch_size, model_dim)
        src = src.permute(1, 0, 2)
        transformer_output = self.transformer_encoder(src)
        # Use the last output token as the summary representation
        last_token = transformer_output[-1, :, :]  # shape (batch_size, model_dim)
        out = self.fc_out(last_token)  # shape (batch_size, input_dim)
        return out

# ----- Step 4: Training the Model -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TrajectoryTransformer(input_dim=2, model_dim=64, num_heads=4, num_layers=3, dropout=0.1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for seq, target in dataloader:
        seq, target = seq.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * seq.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# ----- Step 5: Prediction and Visualization -----
model.eval()
# Choose a sample sequence from the dataset
sample_seq, true_target = dataset[0]
sample_seq = sample_seq.unsqueeze(0).to(device)  # shape (1, seq_length, 2)
with torch.no_grad():
    pred_norm = model(sample_seq).cpu().numpy()[0]

# Inverse transform to get original scale
pred = scaler.inverse_transform(pred_norm.reshape(1, -1))[0]
true_point = scaler.inverse_transform(true_target.reshape(1, -1))[0]
sample_seq_orig = scaler.inverse_transform(sample_seq.cpu().numpy().squeeze())

plt.figure(figsize=(8,6))
plt.plot(sample_seq_orig[:, 0], sample_seq_orig[:, 1], 'bo-', label='Input Sequence')
plt.plot(true_point[0], true_point[1], 'go', markersize=10, label='True Next Point')
plt.plot(pred[0], pred[1], 'ro', markersize=10, label='Predicted Next Point')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Trajectory Prediction using Transformer')
plt.legend()
plt.show()
