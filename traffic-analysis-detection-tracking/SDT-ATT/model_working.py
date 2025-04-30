
import torch 
import torch.nn as nn 
import torch.nn.functional as F

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(BiLSTMEncoder,self).__init__()
        self.lstm= nn.LSTM(input_dim,hidden_dim, batch_first=True, bidirectional=True)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
            out, _= self.lstm(x)
            return self.dropout(out)
        
class TemporalAttention(nn.Module):
    def __init__(self, input_dim):
        super(TemporalAttention,self).__init__() #super calls the initialisation of the base class super calss nn.module
        self.query= nn.Linear(input_dim, input_dim)
        self.key= nn.Linear(input_dim, input_dim)
        self.value= nn.Linear(input_dim, input_dim)

    def forward(self, x):
            Q=self.query(x)
            K= self.key(x)
            V=self.value(x)

            attn_weights= F.softmax(torch.bmm(Q, K.transpose(1, 2)) / (x.size(-1) ** 0.5), dim=-1)
            attented= torch.bmm(attn_weights, V)
            return attented.mean(dim=1)
        
class SpattialDynamicAttention(nn.Module):
    def __init__(self, input_dim):
        super(SpattialDynamicAttention,self).__init__()
        self.query=nn.Linear(input_dim, input_dim)
        self.key= nn.Linear(input_dim, input_dim)
        self.value= nn.Linear(input_dim, input_dim)

    def forward(self, nv_encoded): #shape [B, N, T, D]
        B,N,T,D = nv_encoded.shape
        nv_flat=nv_encoded.view(B*N, T, D) #shape [B*N, T, D]

        Q=self.query(nv_flat)
        K=self.key(nv_flat)
        V=self.value(nv_flat)

        atttn_weights= F.softmax(torch.bmm(Q,K.transpose(1,2))/(D**0.5), dim=1)
        attended= torch.bmm(atttn_weights,V).mean(dim=1) #shape [B*N, D]

        return attended.view(B,N,D)
    
#CHGNAGE NUMBER OF NEIGHBORS ACCORDINGLY 
class SDTATTModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_neighbors=3, future_len=30):
       super(SDTATTModel,self).__init__()
       self.num_neighbors=num_neighbors
       self.future_len=future_len
       
       #encoders
       self.tv_encoder=BiLSTMEncoder(input_dim, hidden_dim)
       self.nv_spatial_encoder=BiLSTMEncoder(input_dim, hidden_dim)
       self.nv_dynamic_encoder=BiLSTMEncoder(input_dim, hidden_dim)

       #Multivariate Interaction Modelling
       #Attention

       self.temporal_attention=TemporalAttention(hidden_dim*2)
       self.spatial_attention=SpattialDynamicAttention(hidden_dim*2)
       self.dynamic_attention=SpattialDynamicAttention(hidden_dim*2)

       #Fully connected layer after Temporal, Spatial and Dynamic Attention
       self.tv_fc=nn.Linear(hidden_dim*2, hidden_dim)
       self.nv_spatial_fc=nn.Linear(hidden_dim*2*num_neighbors, hidden_dim)
       self.nv_dynamic_fc=nn.Linear(hidden_dim*2*num_neighbors, hidden_dim)

       #Direction-aware component for 2-lane double direction
       self.direction_encoder = nn.Linear(2, hidden_dim)  # Encode direction information
       self.direction_fusion = nn.Linear(hidden_dim*4, hidden_dim)  # Fuse direction with other features

       #Fully Connected layer plus fusion and decoder

       self.fc_fusion=nn.Sequential(
           
           nn.Linear(hidden_dim*3, hidden_dim),
           nn.ReLU(),
           nn.Dropout(0.2)
       )

       self.decoder=nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
       self.output= nn.Linear(hidden_dim, 2) #predicts x,y of TV

    def forward(self, tv_hist, nv_sp, nv_dp, direction=None):
        """
        tv_hist:     [B, T, 2]  
        nv_sp:       [B, N, T, 2]
        nv_dp:       [B, N, T, 2]
        direction:   [B, 2] - Optional direction vector (e.g., [1,0] for right, [-1,0] for left)
        """

        B, N, T, _ = nv_sp.shape

        #Encode target vehicle TV
        tv_feat= self.tv_encoder(tv_hist)
        tv_att=self.temporal_attention(tv_feat)

        #encode neighbors (spatial and dynamic)
        nv_sp_encoded= self.nv_spatial_encoder(nv_sp.view(B * N, T, 2)).view(B, N, T, -1)
        nv_dp_encoded= self.nv_dynamic_encoder(nv_dp.view(B * N, T, 2)).view(B, N, T, -1)

        nv_sp_att= self.spatial_attention(nv_sp_encoded) # [B, N, H*2]
        nv_dp_att= self.dynamic_attention(nv_dp_encoded)  # [B, N, H*2]

        #skip connection in Spatial and Dynamic Attention
        nv_sp_att= nv_sp_att + nv_sp_encoded[:, :, -1, :]  
        nv_dp_att= nv_dp_att + nv_dp_encoded[:, :, -1, :]  

        #fully connected layer after each attention 
        tv_att_fc= self.tv_fc(tv_att)
        nv_sp_att_fc= self.nv_spatial_fc(nv_sp_att.view(B,-1))
        nv_dp_att_fc= self.nv_dynamic_fc(nv_dp_att.view(B,-1))

        # Process direction information if provided
        if direction is not None:
            direction_feat = self.direction_encoder(direction)
            # Fuse direction with other features
            fused = torch.cat([tv_att_fc, nv_sp_att_fc.view(B, -1), nv_dp_att_fc.view(B, -1), direction_feat], dim=-1)
            fused = self.direction_fusion(fused)
        else:
            # Original fusion without direction
            fused = torch.cat([tv_att_fc, nv_sp_att_fc.view(B, -1), nv_dp_att_fc.view(B, -1)], dim=-1)
            fused = self.fc_fusion(fused)
            
        fused = fused.unsqueeze(1).repeat(1, self.future_len, 1)

        # Decode trajectory
        decoded, _ = self.decoder(fused)
        out = self.output(decoded)  # [B, future_len, 2]

        return out
 


if __name__=="__main__":
   
   from dataloader import SDTATTDataset
   import os 

   BASE_DIR = os.path.dirname(os.path.abspath(__file__))
   tracking_data_numpy_path= os.path.join(BASE_DIR, "data", "sdtatt_data.npy")
   dataset= SDTATTDataset(tracking_data_numpy_path)
   
   model= SDTATTModel()

   output = model(dataset[0]['tv_hist'].unsqueeze(0), dataset[0]['nv_sp'].unsqueeze(0), dataset[0]['nv_dp'].unsqueeze(0))
   print(output.shape)  