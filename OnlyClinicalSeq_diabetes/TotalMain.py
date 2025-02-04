import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

import Main
from Utils.Set_ARGs import set_args
import os


if __name__ == '__main__':
    
    args=set_args()
    
    args['dataset_name']="mimic4"
    args['internal']=args['dataset_name']
    args['target']="mortality" # bash로 입력 받음
    args['imputation'] = 'simpleimputer_median'
    # args['model_num']=0
    args['labevents'] = True # 제외하자,,, 너무 양이 많고 chartevent의 정보만으로도 괜찮을 것 같다.
        
    # save root, save data, save model은 여기서 선언하자 (defualt는 ''으로 놔두고)
    args['save_root']='data/'
    args["save_data"]=f"Preprocessed_Data3_lab/"
    args["save_model"]=f"250103/{args['target']}/MODEL_DIR/" # /{args['target']}
    
    args["run_code_folder"]=os.path.realpath(__file__) # folder name of running code    
    
    if args['dataset_name']=="mimic4":  
        args['data_root']=args['data_root']+"mimic4_2.2/"
        args['save_data']="MIMIC/"+args['save_data']
        args['save_model']="MIMIC/"+args['save_model']
        args["month"]=3
        args["n_classes"]=2
        args['weight_decay']=0 
        args['steps']=1200
        args["lr"]=0.0004    
        args["select_feature"]="less"
        args["less_fraction"]=0.3
        args["domain_balance_loader"]=True
        args["train_valid_class_balance"]=False # 이것도 weight 맞춰서 계산...?
        args["icustay_day"]=1
        DOMAIN_LIST = [['COPD', 'diabetes'],
                ['hypertension', 'stroke', 'dyslipidemia']
                ]   
        
    args["save_model"] = f"{args['save_root']}/{args['save_model']}"
    args["save_data"] = f"{args['save_root']}/{args['save_data']}"

    args['weight_decay']=0 
    args['steps']=10
    args["lr"]=0.04
    args["domain"]=sum(DOMAIN_LIST, [])
    
    print(args['target'], args['disease_name'])
    
    

    if args['model_num'] == 0:    
        MODEL_NAME=["rf", "lightGBM", "xgb", "lr", 'dt']
        args['eval_metric'] = ['ML']
        args['model_type']='ML'
        args['batch_size']=2639
                    
        for model_name in MODEL_NAME:
            Main.Main(DOMAIN_LIST, args, model_name) 

    elif args['model_num'] == 1:
        MODEL_NAME=['mlp2', 'mlp3', 'mlp4', 'mlp4drop'] 
        args['model_type'] = 'DL'
        args['steps']=10
        args["lr"]=0.003
        args['batch_size']=64
        
        for model_name in MODEL_NAME:
            Main.Main(DOMAIN_LIST, args, model_name) 
            
    elif args['model_num'] == 2:
        MODEL_NAME=['resnet_like8', 'resnet_org8']
        args['model_type'] = 'DL'
        args['steps']=10
        args["lr"]=0.003
        args['batch_size']=64
        
        for model_name in MODEL_NAME:
            Main.Main(DOMAIN_LIST, args, model_name) 
    