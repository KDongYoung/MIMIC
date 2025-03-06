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
    
    elif args['model_name']=='transformer':
        from Networks.Transformer import Transformer
        model = Transformer(args)
    elif args['model_name']=='lstm1':
        from Networks.LSTM import lstm_1layers
        model = lstm_1layers(args)
    elif args['model_name']=='lstm2':
        from Networks.LSTM import lstm_2layers
        model = lstm_2layers(args)
    elif args['model_name']=='lstm3':
        from Networks.LSTM import lstm_3layers
        model = lstm_3layers(args)
    else:
        assert "Model not loaded....."
        
    return model

def find_model_type(model_name): # load model
    if model_name in ["rf", "xgb", "lightGBM", "svm", "lr", "knn", "gbt", 'dt']:
        return "ML"
        
    elif "lstm" in model_name or "transformer" in model_name:
        return "DL"
    else:
        assert "Unknown Model....."