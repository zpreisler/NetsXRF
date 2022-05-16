import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class SpectraDataset(Dataset):
    
    def __init__(self,data,labels):
        super().__init__()

        self.data = data
        self.labels = labels
        
    def __getitem__(self,idx):
        return self.data[idx],self.labels[idx]
    
    def __len__(self):
        return self.data.shape[0]

class SynthSpectraDataset(Dataset):
    
    def __init__(self,data,labels,scales=[None]):
        super().__init__()
        if scales ==[None]:self.scales = [0.1,0.2,0.3,0.4,0.5,0.7,1.0,1.1]
        else:self.scales = scales

        self.data = data
        self.labels = labels
        
    def __getitem__(self,idx):
        scale_factor = np.random.choice(self.scales)
        spec = self.data[idx]*scale_factor
        print(spec,scale_factor, idx)
        return torch.poisson(spec), self.labels[idx]*scale_factor
    
    def __len__(self):
        return self.data.shape[0]
 