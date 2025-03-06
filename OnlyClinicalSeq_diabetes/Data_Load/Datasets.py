from torch.utils.data import Dataset
import numpy as np

"""
Make a tabular dataset
X: row data
Y: class score
"""
class TabSeqDataset(Dataset):
    def __init__(self, X, y, dis_id):
        self.X = X
        self.y = y
        self.len = len(self.y)
        self.dis_id = dis_id
        
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        X = self.X[idx].astype('float32')  
        y = self.y[idx]# .astype('float32')  
    
        return X, y, self.dis_id.iloc[idx]