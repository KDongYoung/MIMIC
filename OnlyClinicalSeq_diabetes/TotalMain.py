import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

import Main
from Utils.Set_ARGs import set_args
import os


if __name__ == '__main__':
    
    args=set_args()
    
    args['dataset_name']="mimic4"
    args['internal']=args['dataset_name']
    args['target']="HbA1c" # bash로 입력 받음
    args['imputation'] = 'simpleimputer_zero'
    # args['model_num']=0
    args['labevents'] = True # hemoglobin 정보 얻기
        
    # save root, save data, save model은 여기서 선언하자 (defualt는 ''으로 놔두고)
    args['save_root']='data/'
    args["save_data"]=f"Preprocessed_Data_hemoglobin_250225_2/"
    args["save_model"]=f"250214/{args['target']}/MODEL_DIR/" # /{args['target']}
    
    args["run_code_folder"]=os.path.realpath(__file__) # folder name of running code    
    
    if args['dataset_name']=="mimic4":  
        args['data_root']=args['data_root']+"mimic4_2.2/"
        args['save_data']="MIMIC/"+args['save_data']
        args['save_model']="MIMIC/"+args['save_model']
        
        args["select_feature"]="less"
        args["less_fraction"]=0.3
        args["domain_balance_loader"]=True
        args["train_valid_class_balance"]=False
        args["icustay_day"]=1
       
    args["save_model"] = f"{args['save_root']}/{args['save_model']}"
    args["save_data"] = f"{args['save_root']}/{args['save_data']}"

    args['weight_decay']=0 
    args['steps']=10
    args["lr"]=0.04
    args['domain'] = 'diabetes'
    
    args['eval_metric'] = ["rmse", "mape", "mae"]
    args['metric_dict'] = {"rmse": 0, "mape": 1, "mae":2}

    print(args['target'], args['disease_name'])
    
    MODEL_NAME=['lstm2', 'lstm3'] # 'transformer, 'lstm1', 
    args['model_type'] = 'DL'
    args['steps']=10
    args["lr"]=0.0005
    args['batch_size']=16
    
    for model_name in MODEL_NAME:
        Main.Main(args, model_name) 
    