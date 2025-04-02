from torch.utils.data import Dataset
import numpy as np

"""
Make a tabular dataset
X: row data
Y: class score

같은 사람에 속하는 여러 윈도우가 동일한 라벨을 갖기 때문에, 사람이 아닌 윈도우 단위의 예측을 수행합니다.

"""
class TabSeqDataset(Dataset):
    def __init__(self, X, y, window_size, stride):
        self.samples, self.sample_labels = [], []
        N = X.shape[0]
        total_length, _ = X[0].shape
        self.dis_id = []
        for i in range(N):
            self.dis_id.extend([X.index[i]]*len(range(0, 1 + total_length - window_size, stride)))
            person_data = X[i]
            person_label = y[i]

            for start in range(0, 1 + total_length - window_size, stride):
                window = person_data[start:start + window_size]
                self.samples.append(window)
                self.sample_labels.append(person_label)
            # self.dis_id.extend([X.index[i]]*len(range(0, 1 + total_length - window_size, stride)))
            
        self.len = len(self.dis_id)
        
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        X = self.samples[idx].astype('float32')  
        y = self.sample_labels[idx]# .astype('float32')  
    
        return X, y, self.dis_id[idx]