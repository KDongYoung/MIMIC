import argparse

def set_args():
    """ Experiment Setting """ 
    # ARGUMENT
    parser = argparse.ArgumentParser(description='Stroke analysis')
    parser.add_argument('--data_root', default='data/DATASET/', help="name of the data folder") # /opt/workspace/Code_Stroke/dataset//
    parser.add_argument('--run_code_folder', default='')
    parser.add_argument('--save_root', default='data/', help="where to save the models and tensorboard records") # MODEL_SAVE_DIR
    parser.add_argument('--save_data', default='Preprocessed_Data/', help="name of the data folder") # DATASET_DIR/
    parser.add_argument('--save_model', default='MODEL_DIR_PATH/', help="where to save the models and tensorboard records") # MODEL_SAVE_DIR
    parser.add_argument('--result_dir', default="", help="save folder name") 
    parser.add_argument('--total_path', default="", help='total result path')
    parser.add_argument('--cuda', type=bool, default=True, help='cuda')
    parser.add_argument('--cuda_num', type=int, default=0, help='cuda number')
    parser.add_argument('--device', default="", help='device')
    parser.add_argument('--dataset_name', default='', help='dataset name: cha, mimic4, eicu')
    parser.add_argument('--internal', default='', help='dataset name: cha, mimic4, eicu')
    parser.add_argument('--external', default='', help='dataset name: cha, mimic4, eicu')
    parser.add_argument('--disease_name', default='Stroke')
    
    parser.add_argument('--optimizer', default="Adam", help='optimizer')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='for sgd optimizer')
    parser.add_argument('--nesterov', type=bool, default=False,  help='for sgd optimizer, fast convergence') 
    parser.add_argument('--scheduler', default="CosineAnnealingLR", help='scheduler')
    parser.add_argument('--n_classes', type=int, default=0, help='num classes')
    
    parser.add_argument('--column_info', default={}, help='which column is categorical and numerical') ######## cha dataset 사용할 때 수정하기
    parser.add_argument('--cat_num', default=0, help='')
     
    ## CHA_infertility
    parser.add_argument('--class', default=3, help='class')
    parser.add_argument('--target', default='', help='target feature')
    parser.add_argument('--dr_sel', default=['부인연령', 'BMI//차병원 검사', 'AMH//cycle직전 가장최근 수치', 
                                             'FSH//cycle직전 가장최근 수치', 'FSH제재_용량', '시술결과', '시술유형'],
                        help='dr select feature') 
    
    
    ## Modeling
    parser.add_argument('--disease_id', type=int, default=0, help='disease_id')
    parser.add_argument('--seed', type=int, default=2030, help='seed')
    parser.add_argument('--kfold', default=5, help='kfold cross validation')
    parser.add_argument('--kfold_type', default='skfold', help='which kfold, default: kfold, skfold')
    parser.add_argument('--model_name', default='', help='model_name')
    parser.add_argument('--model_type', default='DL', help='model_type: DL, ML, MDL')
    parser.add_argument('--dropout_rate', default=0.1, help='dropout_rate')
    parser.add_argument('--activation', default='gelu', help='activation function: gelu, relu, elu')
    parser.add_argument('--align_weight', default=0.1, help='regularization weight')
    parser.add_argument('--loss', default='MSELoss', help='LOSS TYPE')
    
    # Machine Learning
    parser.add_argument('--svm_kernel', default='linear', help='linear, rbf')
    parser.add_argument('--lr_threshold', default=0.5, help='linear regression threshold')
    parser.add_argument('--max_iter', default=100, help='max iteration')
    # TabNet
    parser.add_argument('--n_d', default=8, help='width of the decision prediction layer in tabnet, default: 8')
    parser.add_argument('--n_a', default=8, help='width of the attention embedding for each mask in tabnet, default: 8')
    parser.add_argument('--n_steps', default=3, help='number of steps in tabnet, default: 3')
    # ResNet-like & FT-Transformer
    parser.add_argument('--d', default=256, help='')
    parser.add_argument('--do_embedding', default=True, help='')
    parser.add_argument('--vocab_sizes', default=[], help='')    
    parser.add_argument('--embedding_dim', default=3, help='')
    parser.add_argument('--n_layers', default=6, help='ft-transformer, multi-head attention n_layer')
    parser.add_argument('--n_heads', default=4, help='ft-transformer, multi-head attention n_head (d_token % n_heads==0)')
    parser.add_argument('--d_token', default=64, help='ft-transformer, shape (d_token % n_heads==0)')
    parser.add_argument('--token_bias', default=True, help='toekn bias in feature tokenizer')
    parser.add_argument('--lambda_grl', default=0.3, help='weight regularization for gradient reversal layer')
    parser.add_argument('--tokenize_num', default=False, help='whether to tokenize numerical feature (num*weight)')
    parser.add_argument('--block_name', default=None, help='what do with the ResBlock')
    
    parser.add_argument('--select_feature', default='', help='all, atleast, less')    
    parser.add_argument('--less_fraction', default=0.3, help='select feature that consists at least certain amount of values')    
    parser.add_argument('--normalize', default='', help='standard, min max, robust')
    parser.add_argument('--balance', default='', help='random_oversampling (ROS), weighted_random_oversampling (WROS), SMOTE, SMOTE_Tomek, ADASYN')
    parser.add_argument('--feature_select', default='', help='lda, rfecv')
    parser.add_argument('--max_epoch', default=15, help='max iteration')
    parser.add_argument('--selected_feature_name', default=[], help='selected_feature_name by "feature_select"')
    parser.add_argument('--feature_num', default=53, help='selected_feature length')
    parser.add_argument('--imputation', default='zero', help='zero, median_median, median_mode, simpleimputer_mice') 
    
    parser.add_argument('--domain_balance_loader', default=True, help='domain balance data loader OR original loader')
    parser.add_argument('--train_valid_class_balance', default=True, help='train, valid class balance, default: False')
    parser.add_argument('--init_weight', default="he", help='xavier, he weight initialization')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size of each subject for training (default: 16)') 
    parser.add_argument('--valid_batch_size', type=int, default=1, metavar='N', help='valid batch size for training (default: 1)') 
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N', help='input batch size for testing (default: 1)')

    parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='number worker') 
    # parser.add_argument('--epochs', type=int, default=10, help='Number of epoch') 
    parser.add_argument('--steps', type=int, default=0, help='Number of steps') 
    parser.add_argument('--checkpoint', type=int, default=50, help='Checkpoint every N steps')
    parser.add_argument('--mode', default='train', help='train, infer')
    parser.add_argument('--eval_metric', default=["rmse", "mape"], help='evaluation metric for model selection ["loss", "acc", "bacc", "f1"]')
    parser.add_argument('--metric_dict', default={"rmse": 0, "mape": 1}, help='total evaluation metric')
    parser.add_argument('--save_confusion_matrix', default=False, help='confusion matrix')
    
    ## wandb
    parser.add_argument('--run_wandb', default=False, help='whether to run wandb')
    parser.add_argument('--wandb_key', default="", help='which project')
    parser.add_argument('--tensorboard', default="", help='which wandb run')
    
    
    
    parser.add_argument('--model_num', type=int, default=0, help='which model 임시')
    
    args = parser.parse_args()
    args=vars(args)
    
    return args