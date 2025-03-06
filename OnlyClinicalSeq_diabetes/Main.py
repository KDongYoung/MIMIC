import os
import pandas as pd
import numpy as np
import random
import datetime

import Data_Load.Dataloader as Dataloader
from Utils.Load_model import load_model, find_model_type
import Trainer 
from sklearn.model_selection import KFold, StratifiedKFold

def Experiment(args, path):
    
    args['total_path'] = f"{path}/{args['result_dir']}"
    # make a directory to save results, models    
    if not os.path.isdir(args['total_path']):
        os.makedirs(args['total_path'])
        if args['eval_metric'] != None:
            for metric in args['eval_metric']:
                os.makedirs(f"{args['total_path']}/models/{metric}")
        else: 
            os.makedirs(f"{args['total_path']}/models/")
        os.makedirs(f"{args['total_path']}/confusion_matrix")
        os.makedirs(f"{args['total_path']}/test_predicts")

    # connect GPU/CPU
    import torch.cuda
    args['cuda'] = torch.cuda.is_available()
    # check if GPU is available, if True chooses to use it
    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## fix seed
    seed = args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args['cuda']:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    with open(args['total_path'] + '/args.txt', 'a') as f:
        f.write('Start: '+ datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
    
    # if len(domainList)==1:
    #     args['mode']="infer"
    #     print("No Source Subject.... Change to Inference Phase")

    if args['mode']=="train":
        t_metrics = start_Training(args) # Train
    
    # save ARGUEMENT => 코드가 진행되면서 새로 선언되는 값들이 있다
    with open(args['total_path'] + '/args.txt', 'a') as f:
        f.write(str(args)+"\n\n")
   
    return t_metrics


def start_Training(args):
    
    DATASET = Dataloader.make_dataset(args)
        
    total_valid_best = [] 
    
    kfold = KFold(n_splits=args['kfold'], shuffle=True, random_state=args['seed'])
    kfold_split = kfold.split(DATASET)
    
    for i, (train_index, test_index) in enumerate(kfold_split): 
        print(f"{len(train_index)} Train 데이터, {len(test_index)} Test 데이터")
        
        ## Load Dataset
        train_loaders, valid_loader, test_loader = Dataloader.init_dataloader(args, [DATASET[idx] for idx in train_index], [DATASET[idx] for idx in test_index])
        args["feature_num"] = len(args["selected_feature_name"])
        
        #### Train ####
        model = load_model(args)
        if args['model_type'] =="DL" and args['cuda']:    
            model.cuda(device=args['device']) # connect DEVICE

        trainer = Trainer.Trainer(args, model, i+1, model_name=args['model_name'])
        valid_best = trainer.training(train_loaders, valid_loader, test_loader) # train, valid
        
        #### Evaluation ####
        result = []
        
        for metric in args['eval_metric']:
            # MODEL
            best_model = load_model(args) 
            if args['model_type'] =="DL" and args['cuda']: best_model.cuda(device=args['device']) # connect DEVICE
            
            best_trainer=Trainer.Trainer(args, best_model, i+1, model_name=args['model_name'])
            r = best_trainer.prediction(test_loader, metric)
            r[-1] = np.mean(r[-1][1:])
            result.append([valid_best[args['metric_dict'][metric]]] + r)
        
        total_valid_best.append(result)
        
    return total_valid_best
        
        
def Main(args, model_name):
    
    args["model_name"]=model_name
    args['model_type'] = find_model_type(args["model_name"])    

    exp_type=(
        f"{args['model_name']}_"
        f"c{args['train_valid_class_balance']}_batch{args['batch_size']}_{args['icustay_day']*24}ICU"
    )
    args['result_dir']=exp_type
        
    ## 결과 입력 폴더 생성
    dir = f"{args['steps']}_{args['batch_size']}_{args['optimizer']}_{args['lr']}_{args['imputation']}"
    path = f"{args['save_model']}/{dir}"
    
    if not os.path.isdir(path):
        os.makedirs(f"{path}/Results", exist_ok=True)
        os.makedirs(f"{path}/Logs", exist_ok=True)
        if args['model_type'] == 'ML':
            os.makedirs(f"{path}/Results/ml", exist_ok=True)
        else:
            for metric in args['eval_metric']:
                os.makedirs(f"{path}/Results/{metric}", exist_ok=True)
    
    print(f"{args['seed']} / {args['lr']} / {args['dataset_name']} {path} / {exp_type}")   
    
    results = Experiment(args, path)

    df = [[results[fold][metric_i] for fold in range(args['kfold'])] for metric_i, _ in enumerate(args['eval_metric'])]
    
    for metric_i, metric_name in enumerate(args['eval_metric']):
        part_df = pd.DataFrame(df[metric_i])
                    
        with open(f"{path}/Results/{metric_name}/{exp_type}_Performance.txt", 'a') as f:
            for fold in range(args['kfold']):
                f.write(f"{fold+1} FOLD: RMSE: {part_df.iloc[fold, 1]:.4f}, MAPE: {part_df.iloc[fold, 2]:.4f}\n") # save test performance   
        
        part_df.columns = ['Valid_best', 'RMSE', 'MAPE', 'Cost']
        part_df.index = [metric_name] * args['kfold']
        
        result = pd.concat([pd.DataFrame(part_df.mean()).T.set_axis([f"Avg_{col}" for col in part_df.columns], axis=1),
                            pd.DataFrame(part_df.max()).T.set_axis([f"Max_{col}" for col in part_df.columns], axis=1),
                            pd.DataFrame(part_df.min()).T.set_axis([f"Min_{col}" for col in part_df.columns], axis=1), 
                            pd.DataFrame(part_df.std()).T.set_axis([f"Std_{col}" for col in part_df.columns], axis=1)], axis=1)
        result.index=[exp_type]
        print(pd.DataFrame(part_df.mean()).T)
            
        csv_file = f"{path}/Results/{metric_name}_{dir}.csv"
        # part_df.to_csv(f"{path}/Results/{metric_name}/{exp_type}.csv", mode="a")
        
        if os.path.exists(csv_file):
            result.to_csv(csv_file, mode='a', header=False)  # 🚀 헤더 없이 추가
        else:
            result.to_csv(csv_file, mode='w', header=True)  # 🚀 새 파일 생성
            
    print("\n")
        
        