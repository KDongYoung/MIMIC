import torch

def load_model(args): # load model
    if args['model_name']=="rf":
        from sklearn.ensemble import RandomForestClassifier
        model=RandomForestClassifier(random_state=args['seed'])
    elif args['model_name']=="xgb":
        from xgboost import XGBClassifier
        model=XGBClassifier(n_estimators=args['max_iter'], random_state=args['seed'])
    elif args['model_name']=="lightGBM":
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(n_estimators=args['max_iter'], random_state=args['seed'])
    elif args['model_name']=="svm":
        import sklearn.svm as svm
        model=svm.SVC(kernel = args["svm_kernel"], random_state=args['seed'])
    elif args['model_name']=="lr":
        from sklearn.linear_model import LogisticRegression 
        model=LogisticRegression(max_iter=args['max_iter'], random_state=args['seed'], solver='saga')
    elif args['model_name']=="knn":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=10) 
    elif args['model_name']=="gbt":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(random_state=args['seed'], max_depth=10, learning_rate=0.05) 
    elif args['model_name']=='dt':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(max_depth=10)

    elif args['model_name']=="mlp4":
        from Networks.MLP_4layer import shallowLinearModel_4lowlayer
        model=shallowLinearModel_4lowlayer(args)
    elif args['model_name']=="mlp4drop":
        from Networks.MLP_4layer import shallowLinearModel_4lowlayer_dropout
        model=shallowLinearModel_4lowlayer_dropout(args)
    elif args['model_name']=="mlp3":
        from Networks.MLP_3layer import shallowLinearModel_3lowlayer
        model=shallowLinearModel_3lowlayer(args)
    elif args['model_name']=="mlp2":
        from Networks.MLP_2layer import shallowLinearModel_2lowlayer
        model=shallowLinearModel_2lowlayer(args)

    elif args['model_name']=="resnet_org8":
        from Networks.ResNet_org import resnet_8
        model=resnet_8(args)
    elif args['model_name']=="resnet_org18":
        from Networks.ResNet_org import resnet_18
        model=resnet_18(args)
    elif args['model_name']=="resnet_preact8":
        from Networks.ResNet_preActive import resnet_8
        model=resnet_8(args)
    elif args['model_name']=="resnet_preact18":
        from Networks.ResNet_preActive import resnet_18
        model=resnet_18(args)
    elif args['model_name']=="resnet_like8":
        from Networks.ResNet_like import resnet_8
        model=resnet_8(args)
    elif args['model_name']=="resnet_like18":
        from Networks.ResNet_like import resnet_18
        model=resnet_18(args)
    
    elif args['model_name']=="resnet8_dann":
        from Networks.ResNet_like_dann import resnet_8
        model=resnet_8(args)
    elif args['model_name']=="resnet18_dann":
        from Networks.ResNet_like_dann import resnet_18
        model=resnet_18(args)    
    
    
    elif args['model_name']==f"tabnet_{args['n_d']}_{args['n_a']}_{args['n_steps']}":        
        # from Networks.TabNet import TabNetClassifier
        from pytorch_tabnet.tab_model import TabNetClassifier
        model=TabNetClassifier(optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=args['lr']), 
                                scheduler_params={"T_max":100//args['checkpoint']}, 
                                scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,  
                                mask_type='entmax', n_d=args['n_d'], n_a=args['n_a'], n_steps=args['n_steps'], 
                                cat_idxs = [], cat_dims= [],
                                device_name="cuda") # 'entmax'
        
    elif args['model_name']=='ft_transformer':
        from Networks.FT_Transformer import Transformer
        model = Transformer(args)
    else:
        assert "Model not loaded....."
        
    return model

def find_model_type(model_name): # load model
    if model_name in ["rf", "xgb", "lightGBM", "svm", "lr", "knn", "gbt", 'dt']:
        return "ML"
        
    elif "resnet" in model_name or "mlp" in model_name or "tabnet" in model_name or model_name == 'ft_transformer':
        return "DL"
    else:
        assert "Unknown Model....."