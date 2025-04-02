import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

import Main
from Utils.Set_ARGs import set_args
import os
from itertools import product
import copy


if __name__ == '__main__':
    
    args=set_args()
    
    args['dataset_name']="mimic4"
    args['internal']=args['dataset_name']
    args['target']="HbA1c" # bash로 입력 받음
    args['imputation'] = 'simpleimputer_zero'
    args['labevents'] = True # hemoglobin 정보 얻기
        
    # save root, save data, save model은 여기서 선언하자 (defualt는 ''으로 놔두고)
    args['save_root']='data/'
    args["save_data"]=f"Preprocessed_Data_hemoglobin_250308/"
    args["save_model"]=f"250318/{args['target']}/" # /{args['target']}
    
    args["run_code_folder"]=os.path.realpath(__file__) # folder name of running code    
    
    if args['dataset_name']=="mimic4":  
        args['data_root']=args['data_root']+"mimic4_2.2/"
        args['save_data']="MIMIC/"+args['save_data']
        args['save_model']="MIMIC/"+args['save_model']

        args["domain_balance_loader"]=True
        args["train_valid_class_balance"]=False
        args["icustay_day"]=1
       
    args["save_model"] = f"{args['save_root']}/{args['save_model']}"
    args["save_data"] = f"{args['save_root']}/{args['save_data']}"

    args['weight_decay']=0 
    args['domain'] = ['diabetes']
    
    args['eval_metric'] = ["rmse", "mape", "mae"]
    args['metric_dict'] = {"rmse": 0, "mape": 1, "mae":2}

    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    print(args['target'], args['disease_name'])
    
    MODEL_NAME=['lstm1', 'lstm2'] # 'transformer, 'lstm1', 
    args['model_type'] = 'DL'
    
    args['kfold'] = 5   
    
    if args['model_num'] == 0:     
        param_grid = {
            'lr': [0.001],
            'steps': [5],
            'stride': [6, 12, 24],
            'seq_length': [24, 48, 72],
            'batch_size': [128],
            'lstm_hidden_unit_factor': [1.5, 2, 2.5]        
            }
        
        all_combinations = list(product(
            param_grid['lr'], param_grid['steps'], param_grid['stride'],
            param_grid['seq_length'], param_grid['batch_size'], param_grid['lstm_hidden_unit_factor']
        ))

        for model_name in MODEL_NAME:
            for lr, steps, stride, seq_length, batch_size, lstm_hidden_unit_factor in all_combinations:
                # 기존 args 복사 후 업데이트
                new_args = copy.deepcopy(args)
                new_args.update({
                    'lr': lr,
                    'steps': steps,
                    'seq_length': seq_length,
                    'batch_size': batch_size,
                    'stride': stride,
                    'lstm_hidden_unit_factor': lstm_hidden_unit_factor
                })
            
                Main.Main(new_args, model_name)
    
    elif args['model_num'] == 1:     
        param_grid = {
            'lr': [0.01],
            'steps': [5],
            'stride': [6, 12, 24],
            'seq_length': [24, 48, 72],
            'batch_size': [128],
            'lstm_hidden_unit_factor': [1.5, 2, 2.5]        
            }
        
        all_combinations = list(product(
            param_grid['lr'], param_grid['steps'], param_grid['stride'],
            param_grid['seq_length'], param_grid['batch_size'], param_grid['lstm_hidden_unit_factor']
        ))

        for model_name in MODEL_NAME:
            for lr, steps, stride, seq_length, batch_size, lstm_hidden_unit_factor in all_combinations:
                # 기존 args 복사 후 업데이트
                new_args = copy.deepcopy(args)
                new_args.update({
                    'lr': lr,
                    'steps': steps,
                    'seq_length': seq_length,
                    'batch_size': batch_size,
                    'stride': stride,
                    'lstm_hidden_unit_factor': lstm_hidden_unit_factor
                })
            
                Main.Main(new_args, model_name)