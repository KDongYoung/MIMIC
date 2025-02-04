import os
import pandas as pd
import numpy as np
import random
import datetime

import Data_Load.Dataloader as Dataloader
from Utils.Load_model import load_model, find_model_type
import Trainer 


def Experiment(args, domain_id, domainList):
    
    args['total_path'] = f"{args['save_model']}/{args['seed']}_{args['steps']}_{args['lr']}_{args['imputation']}/{args['result_dir']}"
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
    
    if len(domainList)==1:
        args['mode']="infer"
        print("No Source Subject.... Change to Inference Phase")

    if args['mode']=="train":
        valid_best = start_Training(args, domainList, domain_id) # Train
        args['mode']="infer"
        loss, acc, bacc, f1score, specificity, sensitivity, mean_sensitivity, precision, auroc, auprc, cost = start_Inference(args, domainList, domain_id) # Leave-one-subject-out 
        args['mode']="train"
    if args['mode']=="infer":
        loss, acc, bacc, f1score, specificity, sensitivity, mean_sensitivity, precision, auroc, auprc, cost = start_Inference(args, domainList, domain_id) # Leave-one-subject-out 
        valid_best = [0]*len(args['eval_metric'])
    
    # save ARGUEMENT => 코드가 진행되면서 새로 선언되는 값들이 있다
    with open(args['total_path'] + '/args.txt', 'a') as f:
        f.write(str(args)+"\n\n")
   
    return valid_best, loss, acc, bacc, f1score, specificity, sensitivity, mean_sensitivity, precision, auroc, auprc, cost


def start_Training(args, domainList, domain_id):
    FLATTEN_DOMAIN_LIST=sum(domainList, [])
    train_loaders, valid_loader, test_loader = Dataloader.init_dataloader(args, FLATTEN_DOMAIN_LIST, domain_id, args['model_type'])
    args["feature_num"] = len(args["selected_feature_name"])
        
    # MODEL
    model = load_model(args)
    if args['model_type'] =="DL" and args['cuda']:    
        model.cuda(device=args['device']) # connect DEVICE

    trainer = Trainer.Trainer(args, domain_id, FLATTEN_DOMAIN_LIST, model)
    valid_best = trainer.training(train_loaders, valid_loader, test_loader) # train, valid
    
    result = []
    for metric in args['eval_metric']:
        # MODEL
        best_model = load_model(args) 
        if args['model_type'] =="DL" and args['cuda']: best_model.cuda(device=args['device']) # connect DEVICE
        
        best_trainer=Trainer.Trainer(args, domain_id, FLATTEN_DOMAIN_LIST, best_model)
        r = best_trainer.prediction(test_loader, metric)
        r[-1] = np.mean(r[-1][1:])
        result.append([valid_best[args['metric_dict'][metric]]] + r)

    return valid_best

def start_Inference(args, domainList, domain_id): # prediction  
    FLATTEN_DOMAIN_LIST=sum(domainList, [])

    metrics = ["loss", "acc", "bacc", "f1", "speci", "sens", "meansens", "preci", "rocauc", "auprc", "timecost"]
    t_metrics = {metric: [] for metric in metrics}
    
    test_loader = Dataloader.init_dataloader(args,  FLATTEN_DOMAIN_LIST, domain_id, args['model_type'])
        
    for metric in args['eval_metric']:
        # MODEL
        best_model = load_model(args)    
        if args['cuda']: best_model.cuda(device=args['device']) # connect DEVICE
       
        best_trainer=Trainer(args,  FLATTEN_DOMAIN_LIST, domain_id, best_model)
        
        result = best_trainer.prediction(test_loader, metric) 
        # loss, acc, bacc, f1score, specificity, sensitivity, mean_sensitivity, precision, auroc, auprc, cost
        cost = np.mean(result[-1][1:])
        
        result_dict = dict(zip(metrics[:-1], result[:-1]))
        result_dict["timecost"] = cost
        
        for key, value in result_dict.items():
            t_metrics[key].append(value)
            
    return tuple(t_metrics[metric] for metric in metrics)


def Main(domainList, args, model_name):
    
    args["model_name"]=model_name
    args['model_type'] = find_model_type(args["model_name"])    

    exp_type=(
        f"{args['model_name']}_"
        f"c{args['train_valid_class_balance']}_batch{args['batch_size']}_{args['icustay_day']*24}ICU"
    )
    args['result_dir']=exp_type
        
    ## 결과 입력 폴더 생성
    path = f"{args['save_model']}/{args['seed']}_{args['steps']}_{args['lr']}_{args['imputation']}"
    if not os.path.isdir(f"{path}"):
        for metric in args['eval_metric']:
            os.makedirs(f"{path}/Results/{metric}")

    before_sbj_num=0
    for i in range(args['domain_group']):
        before_sbj_num+=len(domainList[i])
    
    for id in range(len(domainList[args['domain_group']])):
        test_envs = domainList[args['domain_group']][id]
        print(f"{args['seed']} / {args['lr']} / {args['dataset_name']} {path} / {exp_type}")   
        print(f"{'~'*25} Test Domain {test_envs} {'~'*25}")
        
        valid_best, loss, acc, bacc, f1score, specificity, sensitivity, mean_sensitivity, precision, auroc, auprc, cost  = Experiment(args, before_sbj_num+id, domainList)

        if args["mode"]=="train":
            valid_best=[valid_best[args["metric_dict"][i]] for i in args["eval_metric"]]

        total_perf = np.array([["_".join([str(id) for id in test_envs])]*len(args['eval_metric']), valid_best, loss, acc, bacc, f1score, specificity, sensitivity, mean_sensitivity, precision, auroc, auprc, cost])
    
        for metric_i, metric_name in enumerate(args['eval_metric']):
            print(f"{metric_name} Valid: {valid_best[metric_i]:.2f}, TEST_DOMAIN: {test_envs} " \
                f"ACCURACY: {acc[metric_i]:.4f}%, SPECIFICITY: {specificity[metric_i]:.4f}%, SENSITIVITY: {sensitivity[metric_i]:.4f}%, AUROC: {auroc[metric_i]:.4f}%,")
            
            with open(f"{path}/Results/{metric_name}/{exp_type}_Performance.txt", 'a') as f:
                f.write(f"{test_envs} Loss: {loss[metric_i]:.4f}, Acc: {acc[metric_i]:.4f}, " \
                        f"Specificity: {specificity[metric_i]:.4f}, Sensitivity: {sensitivity[metric_i]:.4f}, "
                        f"AUROC: {auroc[metric_i]:.4f}\n") # save test performance   

        # save the performance result into '.csv' file
        df=pd.DataFrame(total_perf)
        for metric_i, metric_name in enumerate(args['eval_metric']):
            part=df.iloc[:, metric_i]
            df_part=part.to_frame().T
        
            print(df_part)
            df_part.to_csv(f"{path}/Results/{metric_name}/{exp_type}.csv", mode="a", index=False)
    
    
    
    
    
    
    # # save the performance result into '.txt' and '.csv file
    # if args['model_type'] =="DL":
    #     df = [[results[fold][metric_i] for fold in range(args['kfold'])] for metric_i, _ in enumerate(args['eval_metric'])]
        
    #     for metric_i, metric_name in enumerate(args['eval_metric']):
    #         part_df = pd.DataFrame(df[metric_i])
                        
    #         with open(f"{path}/Results/{metric_name}/{exp_type}_Performance.txt", 'a') as f:
    #             for fold in range(args['kfold']):
    #                 f.write(f"{fold+1} FOLD: Loss: {part_df.iloc[fold, 1]}, Accuracy: {part_df.iloc[fold, 2]}, " 
    #                         f"Bal_Accuracy: {part_df.iloc[fold, 3]}, F1: {part_df.iloc[fold, 4]}, "
    #                         f"Precision: {part_df.iloc[fold, 5]}, Recall: {part_df.iloc[fold, 6]}, "
    #                         f"AUROC: {part_df.iloc[fold, 7]}\n") # save test performance   
                    
    #         part_df.columns = ['Valid_best', 'Loss', 'Acc', 'BAcc', 'F1', 'Specificity', 'Sensitivity', 'Mean_Sensitivity', 'Precision', 'AUROC', 'AUPRC', 'Cost']
    #         # part_df = pd.concat([part_df, part_df.mean().to_frame().T], ignore_index=True)
    #         # part_df.index = [metric_name] * args['kfold'] + ['Average']
    #         average = part_df.mean().to_frame().T
    #         average.index=['Average']
            
    #         print(average)
    #         average.to_csv(f"{path}/Results/{metric_name}/{exp_type}.csv", mode="a")
    # else:
    #     df = pd.DataFrame(results)
    #     with open(f"{path}/Results/ml/{exp_type}_Performance.txt", 'a') as f:
    #         for fold in range(args['kfold']):
    #             f.write(f"{fold+1} FOLD: Loss: {df.iloc[fold, 0]}, Accuracy: {df.iloc[fold, 1]}," 
    #                         f"Bal_Accuracy: {df.iloc[fold, 2]}, F1: {df.iloc[fold, 3]},"
    #                         f"Precision: {df.iloc[fold, 4]}, Recall: {df.iloc[fold, 5]},"
    #                         f"AUROC: {df.iloc[fold, 6]}\n") # save test performance 
    
    #     df.columns = ['Loss', 'Acc', 'BAcc', 'F1', 'Specificity', 'Sensitivity', 'Mean_Sensitivity', 'Precision', 'AUROC', 'AUPRC', 'Cost']
        
    #     # df = pd.concat([df, df.mean().to_frame().T], ignore_index=True)
    #     # df.index = [f"FOLD {fold+1}"] * args['kfold'] + ['Average']
    #     average = df.mean().to_frame().T
    #     average.index=['Average']
        
    #     print(average)
    #     average.to_csv(f"{path}/Results/ml/{exp_type}.csv", mode="a")           
        
    print("\n")
        
        