import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

from Classification.Main import Main
from Utils.Set_ARGs import set_args
import os


if __name__ == '__main__':
    
    args=set_args()
    
    args['dataset_name']="cha_infertility"
    args['internal']=args['dataset_name']
    args['target']="volume" # bash로 입력 받음
    args["imputation"]="zero_median"
    # args['n_classes'] = 1 # regression
    
    # save root, save data, save model은 여기서 선언하자 (defualt는 ''으로 놔두고)
    args['save_root']='data/'
    args["save_data"]=f"Preprocessed_Data2/"
    args["save_model"]=f"250224/{args['target']}/{args['seed']}/"
    
    args["run_code_folder"]=os.path.realpath(__file__) # folder name of running code    

    args['data_root']=args['data_root']+"cha_infertility/"
    args['targets']= ["volume"]
       
    args["save_model"] = f"{args['save_root']}/{args['dataset_name'].upper()}/{args['save_model']}"
    args["save_data"] = f"{args['save_root']}/{args['dataset_name'].upper()}/{args['save_data']}"
    
    args['weight_decay']=0 
    args["train_valid_class_balance"] = True # 임신 성공, 실패 균형있게

    
    print(args['target'])
    
    args['loss'] = 'CELoss'
    args['aligh_weight'] = 0
    
    args['model_num'] = 0
    
    ## classification
    args['eval_metric'] = ["acc", "f1", "auroc"]
    args['metric_dict'] = {"acc": 0, "f1": 1, "auroc": 2}    
    
    # ## regression
    # args['eval_metric'] = ["rmse", "mape"]
    # args['metric_dict'] = {"rmse": 0, "mape": 1}
    
    if args['model_num'] == 0:    
        MODEL_NAME=["xgb", "lightGBM", "rf", "lr", 'svm', ''] # "xgb", "lightGBM", "rf",
        args['eval_metric'] = None
        args['model_type']='ML'
        
        for args['max_iter'] in [100]:
            for args['normalize'] in ['minmax','robust']: # , 'minmax', 'standard', ''
                for model_name in MODEL_NAME:
                    Main(args, model_name) 
                        
                        
    elif args['model_num'] == 1:    
        MODEL_NAME=['mlp2', 'mlp3', 'mlp4', 'mlp4drop'] 
        args['model_type'] = 'DL'
        args['steps']=30
        args['loss']='CELoss'
        
        for model_name in MODEL_NAME:
            for args["imputation"] in ["zero_median"]:
                for args['lr'] in [0.5]: 
                    for args['normalize'] in  ['robust', 'minmax']:
                        args['batch_size']=128
                        Main(args, model_name) 
    
    elif args['model_num'] == 2:    
        MODEL_NAME=['mlp4', 'mlp4drop', 'mlp2', 'mlp3'] 
        args['model_type'] = 'DL'
        args['steps']=30
        args['loss']='ConsLoss'
        
        for model_name in MODEL_NAME:
            for args["imputation"] in ["zero_median"]:
                for args['lr'] in [0.5]: 
                    for args['normalize'] in  ['robust', 'minmax']:
                        args['batch_size']=128
                        Main(args, model_name) 
                        
    elif args['model_num'] == 3:    
        MODEL_NAME=['mlp2', 'mlp3', 'mlp4', 'mlp4drop'] 
        args['model_type'] = 'DL'
        args['steps']=30
        args['loss']='ConsTripLoss'
        
        for args["imputation"] in ["zero_median"]:
            for args['lr'] in [0.5]: 
                for args['normalize'] in  ['robust', 'minmax']:
                    for model_name in MODEL_NAME:
                        args['batch_size']=128
                        Main(args, model_name) 

                    