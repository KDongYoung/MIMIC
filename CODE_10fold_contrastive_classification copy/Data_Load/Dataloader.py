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

def init_dataloader(args, train_set, test_set):

    holdout_fraction = 0.1 # train:valid = 9:1
    
    if args["train_valid_class_balance"]:
        valid_set, train_set = class_split_dataset(train_set, 2, holdout_fraction, args['seed']) # 성공, 실패
    else:
        valid_set, train_set = split_dataset(train_set, holdout_fraction, args['seed']) 

    if args['model_type']=="ML":
        args['batch_size'] = len(train_set)
        args['valid_batch_size'] = len(valid_set)
        args['test_batch_size'] = len(test_set)
    
    # # for class balance
    # in_weights = make_weights_for_balanced_classes(train_set)
    # sampler=torch.utils.data.WeightedRandomSampler(in_weights, replacement=True, num_samples=len(train_set)) # num_sample을 전체 크기로 정해야함, 아니면 batch_size로 하는 경우 batch가 1개가 될수도
    
    ############################################################### for Trainning: Train, Valid, Test loader
    if args['mode'] == "train": 
        if args['model_type'] == "DL":
            CON_PAIRS, person_id = create_contrastive_pairs(train_set)
            TRIP_PAIRS = create_triplet_data(train_set, person_id)
            train_set  = CrossSubjectContrastiveTripletDataset(CON_PAIRS, TRIP_PAIRS)
        else: # ML
            train_set = SimpTabDataset(train_set, args['selected_feature_name'], args['target'])
            train_set = torch.utils.data.ConcatDataset([train_set])
                    
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'], pin_memory=True,
                                                   num_workers=args['num_workers']) # train loader

        valid_set = SimpTabDataset(valid_set, args['selected_feature_name'], args['target'])
        valid_set = torch.utils.data.ConcatDataset([valid_set])
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args['valid_batch_size'],
                                                   shuffle=False, pin_memory=True, 
                                                   num_workers=args['num_workers']) # valid loader

        test_set = SimpTabDataset(test_set, args['selected_feature_name'], args['target'])
        test_set = torch.utils.data.ConcatDataset([test_set])
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['test_batch_size'], 
                                                  shuffle=False, pin_memory=True, 
                                                  num_workers=args['num_workers'])

        return train_loader, valid_loader, test_loader
    
    ############################################################### for Inference: only Test loader
    elif args['mode'] == "infer": 
        test_set = SimpTabDataset(test_set, args['selected_feature_name'], args['target'])
        test_set = torch.utils.data.ConcatDataset([test_set])
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['test_batch_size'], 
                                                  shuffle=False, pin_memory=True, 
                                                  num_workers=args['num_workers'])

        return test_loader
 
def make_dataset(args):
    
    name=args["internal"]
    data_set = load_dataset()
    
    make, exist = make_new_dataset(args['save_data'], name) 

    if make:
        start_time=datetime.datetime.now()
        if exist: shutil.rmtree(f"{args['save_data']}") # deleter before saved data folder
        dataset=data_set(args)
        print(f"Dataset make: {datetime.datetime.now()-start_time}")
    
    print("Dataset load!")
    dataset=pd.read_csv(f"{args['save_data']}/{name}.csv", low_memory=False) 
    
    ## Preprocess (feature selection과 dataset 재정리)
    dataset = load_preprocess(dataset, args) 
    
    return dataset
    
   
def load_dataset():
    from Make_Dataset.CHA_INFERTILITY_Dataset import CHA_INFERTILITY_Dataset
    dataset=CHA_INFERTILITY_Dataset
    return dataset

    
def load_preprocess(dataset, args):
    from Preprocess.CHA_INFERTILITY_preprocess import main
    dataset = main(args, dataset)
    return dataset


def make_new_dataset(path, name):
    if not os.path.exists(path) or os.listdir(path)==[] or "args.txt" not in os.listdir(path):
        return True, 0        
    
    ## 이전 데이터셋 구축 후 7일 지난 뒤에 다시 만들기
    with open(path + '/args.txt', 'r') as f:
        line=f.read().split("\n")
    
    if name not in line[1:] and name not in [csv.split(".")[0] for csv in os.listdir(path) if "args" not in csv]:
        return True, 0
    
    ## 특정 파일 재 생성
    if line[1:] != [csv.split(".")[0] for csv in os.listdir(path) if "args" not in csv]:
        return True, 0    
    
    if (datetime.datetime.strptime(line[0],'%Y%m%d')+datetime.timedelta(days=7))<datetime.datetime.now():
        return True, 1
    else:
        return False, 0


