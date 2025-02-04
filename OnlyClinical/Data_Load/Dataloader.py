import numpy as np
import pandas as pd
import datetime
import os
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

from Data_Load.dataloader_utils import *
import shutil

def init_dataloader(args, DOMAIN_LIST, domain_id, model_type):
    test_envs = np.r_[domain_id].tolist() 
    dataset = make_dataset(args)
        
    ############################################################### for Trainning: Train, Valid, Test loader
    if args['mode'] == "train":  
        train_envs = list(range(len(DOMAIN_LIST)))
        # id가 여러개인 경우 for id in domain_id:
        train_envs.remove(domain_id)
        
        train_dataset=[dataset[i] for i in train_envs]
        test_set=[dataset[i] for i in test_envs] 

        if model_type == "ML":
            train_loaders, valid_loader, test_loder = ml_loader(train_dataset=train_dataset, test_set=test_set, args=args)
        elif args['domain_balance']:
            train_loaders, valid_loader, test_loder = dl_infinite_loader(train_dataset=train_dataset, test_set=test_set, args=args)
        else: 
            train_loaders, valid_loader, test_loder = dl_loader(train_dataset=train_dataset, test_set=test_set, args=args)
            
        return train_loaders, valid_loader, test_loder 
    ############################################################### for Inference: only Test loader
    elif args['mode'] == "infer":
        test_set = torch.utils.data.ConcatDataset(test_set)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['test_batch_size'], 
                                                  shuffle=False, pin_memory=True, 
                                                  num_workers=args['num_workers'])

        return test_loader
 
def make_dataset(args):
    name=args["internal"]
    data_set = load_dataset()
    make, exist = make_new_dataset(args['save_data']) 

    if make:
        start_time=datetime.datetime.now()
        if exist: shutil.rmtree(f"{args['save_data']}") # deleter before saved data folder
        dataset=data_set(args)
        print(f"Dataset make: {datetime.datetime.now()-start_time}")
    
    print("Clinical data load!")
    dataset=pd.read_csv(f"{args['save_data']}/{name}_{args['target']}.csv") 
    
    dataset, n_class=class_division(dataset, args['target'], args['month'], args['n_classes'])
    args['n_classes']=n_class
    
    ## Preprocess (feature selection과 dataset 재정리)
    dataset = load_preprocess(dataset, args) 
        
    return dataset
       
def load_dataset():    
    from Make_Dataset.MIMIC4_Dataset import MIMIC4_Dataset        
    dataset=MIMIC4_Dataset
    return dataset

def load_preprocess(dataset, args):
    from Preprocess.MIMIC_preprocess import main      
    dataset = main(args, dataset)
    return dataset

def class_division(df, target, month=0, c=0):
    
    if target=="mortality":
        ## 입원 후 데이터를 기반으로 퇴원 후 'month'개월 내 사망, 'month'개월 후 생존
        if c==2:
            df.loc[(0<df["mortality"]) & (df["mortality"]<month*30), "mortality"]=0 # 퇴원 후 'month'개월 내 사망
            df.loc[df["mortality"]>=month*30, "mortality"]=1 # 퇴원 후 'month'개월 후 생존
            n_class=2
        elif c==5:
            c=0
            before=0    
            for i in range(month, 12+1, 3):
                print(before, i, c, month)
                df.loc[(before*30<=df["mortality"]) & (df["mortality"]<i*30), "mortality"]=c # 퇴원 후 'month'개월 내 사망
                before=i
                c+=1
            df.loc[df["mortality"]>=i*30, "mortality"]=c # 퇴원 후 'month'개월 후 생존
            print(f'#class: {c+1}, {month} month term')
            n_class=c+1
        
    return df, n_class

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

    