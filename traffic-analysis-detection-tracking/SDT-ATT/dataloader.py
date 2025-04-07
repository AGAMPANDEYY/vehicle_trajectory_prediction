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
    
    def get_sample_by_frame_and_track(self, frame_id=None, track_id=None):
        if track_id is None:
            for i in range(len(self.data)):
                if self.data[i]['center_frame'] == frame_id:
                    sample = self[i]
                    return sample
                
        if frame_id is None:
            for i in range(len(self.data)):
                if self.data[i]['tv_id'] == track_id:
                    sample = self[i]
                    return sample
        if frame_id is not None and track_id is not  None:
            for i in range(len(self.data)):
                if self.data[i]['center_frame'] == frame_id and self.data[i]['tv_id'] == track_id:
                    sample = self[i]  # Reuse __getitem__ to get tensors
                    return sample
        else:
         raise ValueError(f"No sample found for Frame ID {frame_id}, Track ID {track_id}")
