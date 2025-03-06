import numpy as np
import pandas as pd
import datetime
import os
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

from Data_Load.dataloader_utils import *
from Data_Load.Datasets import *
import shutil
from Make_Dataset import data2sequence

def init_dataloader(args, train_set, test_set):
    
    holdout_fraction = 0.1 # train:valid = 9:1
    
    valid_set, train_set = split_dataset_ratio(train_set, holdout_fraction, args['seed']) 

    ############################################################### for Trainning: Train, Valid, Test loader
    if args['mode'] == "train":  
        
        ### loader
        train_set = torch.utils.data.ConcatDataset([train_set])
        train_loaders = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'],
                                                shuffle=True, pin_memory=True, 
                                                num_workers=args['num_workers']) # valid loader

        valid_set = torch.utils.data.ConcatDataset([valid_set])
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args['valid_batch_size'],
                                                shuffle=False, pin_memory=True, 
                                                num_workers=args['num_workers']) # valid loader
        
        test_set = torch.utils.data.ConcatDataset([test_set])
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['test_batch_size'], 
                                                shuffle=False, pin_memory=True, 
                                                num_workers=args['num_workers']) # test loader
                
        return train_loaders, valid_loader, test_loader  
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
    
    ## sequence화 하기
    make = make_to_sequence(args['save_data']) 
    if make:
        data2sequence.data2seq(args, dataset)
        print(f"Dataset convert into sequence: {datetime.datetime.now()}")
    
    ## Preprocess (feature selection과 dataset 재정리)
    dataset = load_preprocess(args) 
        
    return dataset
      
def load_dataset():    
    from Make_Dataset.MIMIC4_Dataset import MIMIC4_Dataset        
    dataset=MIMIC4_Dataset
    return dataset

def load_preprocess(args):
    from Preprocess.MIMIC_preprocess import main      
    dataset = main(args)
    return dataset

