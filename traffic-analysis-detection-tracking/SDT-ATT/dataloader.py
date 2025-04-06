import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader

class SDTATTDataset(Dataset):
    def __init__ (self, numpy_path):
        self.data=np.load(numpy_path, allow_pickle=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]

        tv_hist = torch.tensor(sample['tv_hist'], dtype=torch.float32)       # [TH, 2]
        nv_sp = torch.tensor(sample['nv_sp'], dtype=torch.float32)          # [N, TH, 2]
        nv_dp = torch.tensor(sample['nv_dp'], dtype=torch.float32)          # [N, TH, 2]

        return {
            'tv_hist': tv_hist,
            'nv_sp': nv_sp,
            'nv_dp': nv_dp,
            'center_frame': sample['center_frame'],
            'tv_id': sample['tv_id']
        }