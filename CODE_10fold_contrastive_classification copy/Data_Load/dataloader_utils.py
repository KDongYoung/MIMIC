import torch
import numpy as np
from collections import Counter

from Utils.Preprocess.Normalization import Normalize
from Utils.Preprocess.Balancing import Balance

# normalize and balancing dataset for Machine Learning
def make_dataset4ml(name, df, target, normalize, balance, seed):
    WEIGHT=None
    X, Y = df.loc[:, df.columns != target], df.loc[:, target]
    X, Y = Normalize(normalize, X, Y) 
    if name=="train":
        X, Y, WEIGHT = Balance(balance, seed, X, Y)     ## balancing
    return X, Y, WEIGHT


# get weights (use the number of samples)
def make_weights_for_balanced_classes(dataset):  
    counts = Counter()
    classes = []
    
    for y in dataset:
        y = int(y[1])
        counts[y] += 1 
        classes.append(y) 
    n_classes = len(counts)

    # calculate weight
    weight_per_class = {}
    for y in counts: 
        weight_per_class[y] = 1 / (counts[y] * n_classes)
    weights = np.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights



'''
#############################################################
Split dataset
#############################################################
'''
## split dataset into train, valid
class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)
    
## split dataset into train, valid in a balanced way for classes
def class_split_dataset(dataset, n_classes, n, seed=0):
    classes=list(range(n_classes))
    c_idx=[[] for _ in range(n_classes)] # n_classes class

    for idx, y in enumerate(dataset):
        for i in range(n_classes): 
            if int(y['result']) == classes[i]: # class
                c_idx[i].append(idx)
            
    for i in range(n_classes):
        np.random.RandomState(seed).shuffle(c_idx[i])

    valid=[]
    train=[]
    for i in range(n_classes):
        valid += c_idx[i][:int(n*len(c_idx[i]))]
        train += c_idx[i][int(n*len(c_idx[i])):]
        
    return _SplitDataset(dataset, valid), _SplitDataset(dataset, train) # valid, train

## split dataset into train, valid (randomly train valid split)
def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    n=int(len(dataset)*n)
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    valid = keys[:n]
    train = keys[n:]
    return _SplitDataset(dataset, valid), _SplitDataset(dataset, train) # valid, train