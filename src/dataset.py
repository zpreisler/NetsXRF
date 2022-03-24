import torch
from torch.utils.data import Dataset

import h5py

class SpectraDataset(Dataset):
    
    def __init__(self,data,labels):
        super().__init__()

        self.data = data
        self.labels = labels
        
    def __getitem__(self,idx):
        
        return self.data[idx],self.labels[idx]
    
    def __len__(self):
        return self.data.shape[0]
