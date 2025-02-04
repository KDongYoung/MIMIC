import os
import datetime
import torch
import numpy as np
from collections import Counter

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

def make_new_dataset(path):
    if not os.path.exists(path) or os.listdir(path)==[] or "args.txt" not in os.listdir(path):
        return True, 0        
    
    ## 이전 데이터셋 구축 후 7일 지난 뒤에 다시 만들기
    with open(path + '/args.txt', 'r') as f:
        line=f.read().split("\n")
    
    ## 특정 파일 재 생성
    valid_files = [csv.split(".")[0] for csv in os.listdir(path)
                   if all(exclude not in csv for exclude in ["column_info", "sequence_dataset", "args"]) and csv.endswith(".csv") and not csv.startswith("unique_ICD")]
    if line[1:] != valid_files:
        return True, 0    
    
    if (datetime.datetime.strptime(line[0],'%Y%m%d')+datetime.timedelta(days=7))<datetime.datetime.now():
        return True, 1
    else:
        return False, 0

def make_to_sequence(path):
    if not os.path.exists(f"{path}/sequence"):
        return True
    elif not os.path.exists(path) or "sequence_0.csv" not in os.listdir(f"{path}/sequence"):
        return True
    else:
        return False   
    
def custom_collate_fn(batch):
    # Unpack the batch into separate lists
    X_batch, y_batch, dis_ids, groups = zip(*batch)
    X_batch = torch.tensor(np.stack(X_batch), dtype=torch.float32)  # Stack features
    y_batch = torch.tensor(y_batch, dtype=torch.int64)    # Stack targets
    dis_ids = list(dis_ids)  
    groups = list(groups)    
    
    return X_batch, y_batch, dis_ids, groups

def ml_loader(train_dataset, test_set, args):
    train_splits = [] # TRAIN SET
    holdout_fraction = 0

    for _, env in enumerate(train_dataset): # train_dataset, dataset
        _, train_set = split_dataset_ratio(env, holdout_fraction, args['seed']) 
        train_splits.append(train_set)
                       
    train_set = torch.utils.data.ConcatDataset(train_splits)    
    train_loaders = torch.utils.data.DataLoader(train_set, batch_size=len(train_set),
                                            shuffle=True, pin_memory=True, collate_fn=custom_collate_fn, 
                                            num_workers=args['num_workers']) # valid loader
    
    test_set = torch.utils.data.ConcatDataset(test_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['test_batch_size'], 
                                            shuffle=False, pin_memory=True, 
                                            num_workers=args['num_workers']) # test loader
    return train_loaders, None, test_loader
    
def dl_infinite_loader(train_dataset, test_set, args):
    # in_split: train set / out_split: val set
    train_splits = [] # TRAIN SET
    valid_splits = [] # VALID SET
    holdout_fraction = 0.1 # train:valid = 9:1

    for _, env in enumerate(train_dataset): # train_dataset, dataset
        if args["class_balance"]:
            valid_set, train_set = class_split_dataset(env, args['n_classes'], holdout_fraction, args['seed']) 
        else:
            valid_set, train_set = split_dataset_ratio(env, holdout_fraction, args['seed']) 
            
        # for class balance
        train_set_weights = make_weights_for_balanced_classes(train_set)
        valid_set_weights = make_weights_for_balanced_classes(valid_set)
        train_splits.append((train_set, train_set_weights))
        valid_splits.append((valid_set, valid_set_weights))
            
    holdout_fraction = 0.1 # train:valid = 9:1
    
    ### subject wise loader
    train_loaders = [InfiniteDataLoader(
                        dataset=env,
                        weights=env_weights,
                        batch_size=args['batch_size'],
                        num_workers=args['num_workers'])
                    for _, (env, env_weights) in enumerate(train_splits)] # make a train loader for each source subject 

    valid_set = torch.utils.data.ConcatDataset([env for _, (env, _) in enumerate(valid_splits)])
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args['valid_batch_size'],
                                            shuffle=False, pin_memory=True, 
                                            num_workers=args['num_workers']) # valid loader
    
    test_set = torch.utils.data.ConcatDataset(test_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['test_batch_size'], 
                                            shuffle=False, pin_memory=True, 
                                            num_workers=args['num_workers']) # test loader
            
    return train_loaders, valid_loader, test_loader

def dl_loader(train_dataset, test_set, args):
    # in_split: train set / out_split: val set
    train_splits = [] # TRAIN SET
    valid_splits = [] # VALID SET
    holdout_fraction = 0.1 # train:valid = 9:1

    for _, env in enumerate(train_dataset): # train_dataset, dataset
        if args["class_balance"]:
            valid_set, train_set = class_split_dataset(env, args['n_classes'], holdout_fraction, args['seed']) 
        else:
            valid_set, train_set = split_dataset_ratio(env, holdout_fraction, args['seed']) 
            
        # for class balance
        train_set_weights = make_weights_for_balanced_classes(train_set)
        valid_set_weights = make_weights_for_balanced_classes(valid_set)
        train_splits.append((train_set, train_set_weights))
        valid_splits.append((valid_set, valid_set_weights))
                
    ### loader
    sampler = torch.utils.data.WeightedRandomSampler([weights for (_, weights) in train_splits], 
                                                     replacement=True, 
                                                     num_samples=args['batch_size'])
    train_set = torch.utils.data.ConcatDataset([env for (env, _) in train_splits])
    train_loaders = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'],
                                            shuffle=True, pin_memory=True, sampler=sampler, 
                                            num_workers=args['num_workers']) # valid loader

    valid_set = torch.utils.data.ConcatDataset([env for (env, _) in enumerate(valid_splits)])
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args['valid_batch_size'],
                                            shuffle=False, pin_memory=True, 
                                            num_workers=args['num_workers']) # valid loader
    
    test_set = torch.utils.data.ConcatDataset(test_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['test_batch_size'], 
                                            shuffle=False, pin_memory=True, 
                                            num_workers=args['num_workers']) # test loader
            
    return train_loaders, valid_loader, test_loader


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
            if int(y[1]) == classes[i]: # class
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
def split_dataset_ratio(dataset, n, seed=0):
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


## split dataset into train, valid (randomly train valid split) sample 비율이 아니라 갯수 지정
def split_dataset_num(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    # n=int(len(dataset)*n)
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    valid = keys[:n]
    train = keys[n:]
    return _SplitDataset(dataset, valid), _SplitDataset(dataset, train) # valid, train

'''
#############################################################
Infinite_DataLoader
#############################################################
'''
"""Infinite Dataloder for each subject
Reference:
      Gulrajani et al. In Search of Lost Domain Generalization. ICLR 2021.
"""

class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=True)

        if weights is None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch
