import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.dropout(out)

class SDTATTModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_neighbors=3, future_len=30, n_heads=4):
        super(SDTATTModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.future_len = future_len
        self.num_neighbors = num_neighbors
        emb_dim = hidden_dim * 2

        # --- Hierarchical Embedding & Bi-LSTM Extractors ---
        # TV embedding: pos only; velocity/accel can be concatenated if available citeturn3file0turn3file1
        self.tv_embed = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.tv_encoder = BiLSTMEncoder(emb_dim, hidden_dim)

        # NV spatial embedding (position coords) citeturn3file0turn3file1
        self.nv_sp_embed = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.nv_sp_encoder = BiLSTMEncoder(emb_dim, hidden_dim)

        # NV dynamic embedding (vel & acc if available; here pos diff) citeturn3file0turn3file1
        self.nv_dyn_embed = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.nv_dyn_encoder = BiLSTMEncoder(emb_dim, hidden_dim)

        # --- Multivariate Interaction: Multi-Head Attention ---
        # Temporal MHA over TV history citeturn3file7turn3file5
        self.temporal_mha = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, batch_first=True)
        self.temporal_fc = nn.Linear(emb_dim, emb_dim)

        # Spatial interaction: social tensor + MHA + residual citeturn3file8turn3file11
        self.spatial_mha = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, batch_first=True)
        self.spatial_fc = nn.Linear(emb_dim, emb_dim)

        # Dynamic interaction: social tensor + MHA + residual citeturn3file8turn3file11
        self.dynamic_mha = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, batch_first=True)
        self.dynamic_fc = nn.Linear(emb_dim, emb_dim)

        # Fusion & Direct Multi-Step Decoder citeturn3file2turn3file18
        # Combine HTI, HSI, HDI into HMVI
        self.de_embed = nn.Linear(emb_dim * 3, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # Output 5 parameters for bivariate Gaussian (μx, μy, σx, σy, ρ) per step
        self.output = nn.Linear(hidden_dim, 5)

    def forward(self, tv_hist, nv_sp, nv_dp):
        B, N, T, _ = nv_sp.shape

        # --- TV branch ---
        tv_e = self.tv_embed(tv_hist)                    # [B, T, emb_dim]
        tv_feat = self.tv_encoder(tv_e)                   # [B, T, 2*hidden]
        # Temporal attention
        temp_out, _ = self.temporal_mha(tv_feat, tv_feat, tv_feat)  # [B, T, emb_dim]
        HTI = F.relu(self.temporal_fc(temp_out.mean(dim=1)))       # [B, emb_dim]

        # --- NV spatial branch ---
        # Flatten neighbors for embedding+encoding
        sp_flat = nv_sp.view(B * N, T, -1)
        sp_e = self.nv_sp_embed(sp_flat)                  # [B*N, T, emb_dim]
        sp_feat = self.nv_sp_encoder(sp_e).view(B, N, T, -1)
        # Social tensor: sum over neighbors of last time-step
        Hnbrs_l = sp_feat[:, :, -1, :]                    # [B, N, emb_dim]
        Hspa_tensor = Hnbrs_l.sum(dim=1).unsqueeze(1)     # [B, 1, emb_dim]
        spa_out, _ = self.spatial_mha(Hspa_tensor, Hspa_tensor, Hspa_tensor)
        HSI = F.relu(self.spatial_fc(spa_out.squeeze(1))) + Hspa_tensor.squeeze(1)

        # --- NV dynamic branch ---
        dp_flat = nv_dp.view(B * N, T, -1)
        dp_e = self.nv_dyn_embed(dp_flat)
        dp_feat = self.nv_dyn_encoder(dp_e).view(B, N, T, -1)
        Hnbrs_d = dp_feat[:, :, -1, :]
        Hdyn_tensor = Hnbrs_d.sum(dim=1).unsqueeze(1)
        dyn_out, _ = self.dynamic_mha(Hdyn_tensor, Hdyn_tensor, Hdyn_tensor)
        HDI = F.relu(self.dynamic_fc(dyn_out.squeeze(1))) + Hdyn_tensor.squeeze(1)

        # --- Fusion & decoding ---
        HMVI = torch.cat([HTI, HSI, HDI], dim=-1)        # [B, 3*emb_dim]
        de_in = F.relu(self.de_embed(HMVI)).unsqueeze(1).repeat(1, self.future_len, 1)
        dec_out, _ = self.decoder(de_in)
        params = self.output(dec_out)                      # [B, future_len, 5]

        return params


if __name__=="__main__":
   
   from dataloader import SDTATTDataset
   import os 

   BASE_DIR = os.path.dirname(os.path.abspath(__file__))
   tracking_data_numpy_path= os.path.join(BASE_DIR, "data", "sdtatt_data.npy")
   dataset= SDTATTDataset(tracking_data_numpy_path)
   
   model= SDTATTModel()

   output = model(dataset[0]['tv_hist'].unsqueeze(0), dataset[0]['nv_sp'].unsqueeze(0), dataset[0]['nv_dp'].unsqueeze(0))
   print(output.shape)  