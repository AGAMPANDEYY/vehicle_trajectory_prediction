import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import SDTATTModel
from dataloader import SDTATTDataset
from tqdm import tqdm


# --- Hyperparameters (from the paper) ---

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "sdtatt_data.npy")
DATA_PATH="/kaggle/input/sdtatt-np/sdtatt_data.npy"
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 50
HISTORY_LEN = 90       # 3s @30Hz
FUTURE_LEN = 90        # 5s @30Hz
EMB_DIM = 32
HIST_HIDDEN = 64      # Bi-LSTM hidden units
DE_HIDDEN = 128       # Decoder LSTM hidden units
NUM_LAYERS_ENC = 4    # Bi-LSTM layers
NUM_LAYERS_DEC = 1    # Decoder LSTM layers
N_HEADS_TEMP = 2
N_HEADS_SPA = 3
N_HEADS_DYN = 3
TRAIN_SPLIT = 0.8
NUM_NEIGHBORS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_nll_loss(params, target):
    """
    Negative log-likelihood for bivariate Gaussian.
    params: [B, T, 5] -> mu_x, mu_y, sigma_x, sigma_y, rho
    target: [B, T, 2] -> true relative deltas
    """
    mu_x = params[..., 0]
    mu_y = params[..., 1]
    sigma_x = torch.clamp(params[..., 2], min=1e-3)
    sigma_y = torch.clamp(params[..., 3], min=1e-3)
    rho = torch.tanh(params[..., 4])
    x = target[..., 0]
    y = target[..., 1]

    norm_x = (x - mu_x) / sigma_x
    norm_y = (y - mu_y) / sigma_y
    z = norm_x**2 + norm_y**2 - 2 * rho * norm_x * norm_y
    denom = 2 * (1 - rho**2)
    numerator = torch.exp(-z / denom)
    coefficient = 1.0 / (2 * math.pi * sigma_x * sigma_y * torch.sqrt(1 - rho**2))
    pdf = coefficient * numerator + 1e-6
    nll = -torch.log(pdf)
    return nll.mean()


def train():
    # 1) Load dataset
    full_dataset = SDTATTDataset(DATA_PATH)
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2) Build model
    model = SDTATTModel(input_dim=2, hidden_dim=HIST_HIDDEN,
                        num_neighbors=NUM_NEIGHBORS,
                        future_len=FUTURE_LEN)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 3) (Optional) load means/stds if dataset normalizes inside
    # tv_mean, tv_std = full_dataset.tv_mean.to(DEVICE), full_dataset.tv_std.to(DEVICE)

    # 4) Training loop
    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            tv_hist = batch['tv_hist'].to(DEVICE)     # [B, T_hist, 2]
            nv_sp = batch['nv_sp'].to(DEVICE)         # [B, N, T_hist, 2]
            nv_dp = batch['nv_dp'].to(DEVICE)         # [B, N, T_hist, 2]
            tv_fut = batch['tv_fut_rel'].to(DEVICE)   # [B, T_fut, 2]

            optimizer.zero_grad()
            output = model(tv_hist, nv_sp, nv_dp)    # [B, T_fut, 5]
            loss = gaussian_nll_loss(output, tv_fut)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * tv_hist.size(0)

        avg_train_loss = total_loss / train_size

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tv_hist = batch['tv_hist'].to(DEVICE)
                nv_sp = batch['nv_sp'].to(DEVICE)
                nv_dp = batch['nv_dp'].to(DEVICE)
                tv_fut = batch['tv_fut_rel'].to(DEVICE)
                output = model(tv_hist, nv_sp, nv_dp)
                val_loss += gaussian_nll_loss(output, tv_fut).item() * tv_hist.size(0)
        avg_val_loss = val_loss / val_size

        print(f"Epoch {epoch}/{EPOCHS}  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint every few epochs
        if epoch % 10 == 0:
            ckpt_path = f"checkpoints/sdtatt_epoch{epoch}.pt"
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

    # Final save
    ckpt_path = os.path.join(BASE_DIR, "checkpoints", "sdtatt_final.pt")
    torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    train()
